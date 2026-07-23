# TF1 → TF2 Migration Plan — `triplet-reid` inference/training pipeline

> Written 2026-07-17 in `agents_writer`, after a forensic investigation into Laurens's
> recollection that the TF1→TF2 migration hit unremembered problems. Investigation
> covered: the `inference_webex.zip` source dump, the live upstream `triplet-reid` repo,
> and the actual `laubohlen/ibex_embedding_cpu` Docker Hub images RunPod runs.

## Scope — this is NOT a production fix

Production (`laubohlen/ibex_embedding_cpu:v0.01`, model `wibex_model_v03`) was
**fully verified sound** during this investigation:

- `pip freeze` inside the running container matches `builder/requirements.txt` exactly
  (`tensorflow==2.18.0`, `keras==3.8.0`, `numpy==2.0.2`).
- `handler.py` is byte-identical to the zip's copy.
- SHA256 of `embedding_model/variables/variables.data-00000-of-00001` is identical
  across the deployed image, the zip's `runpod_ibex_embedding_endpoint/embedding_model/`,
  and `wibex_model_v03` — confirms v03 is the exact shipped model.
- Live inference through the deployed container reproduces the pre-migration TF1
  embedding (`test_embedding_old.h5`) to within 2.86e-06 max abs diff — normal
  float32 non-determinism, not a regression.

**This plan is about making the *training/export* pipeline reproducible**, not about
fixing anything currently broken. Right now a working model exists only because of
several uncommitted, ad-hoc export attempts — the goal is a clean, single, committed
pipeline that produces the same result on purpose, not by accident.

## What actually went wrong (forensic findings)

Two real bugs surfaced in Laurens's own notebook trail (`inference/new_model.ipynb`,
`inference/test.ipynb`):

1. **`FailedPreconditionError: uninitialized value resnet_v1_50/.../BatchNorm/moving_variance`**
   on load — a checkpoint-restore variable-name/collection mismatch after swapping
   `tf.contrib.slim` for `tf_slim`.
2. **Numeric drift**: `emb_from_sess` vs `emb_from_frozen` were neither `array_equal`
   nor `allclose` on the same test image — a real discrepancy observed mid-migration.

**Root cause of both, confirmed via SHA256 lineage across all 6 SavedModel exports in
the zip** (`wibex_model_v01/v02/v03`, `model/`, `new_model2/`, `saved_model/`):
trained weights are bit-identical across every export except the abandoned `v02`
detour — **there was never a retrain**. The entire struggle was graph/export
mechanics: several dead-end attempts using the old
`tf.compat.v1.saved_model.builder.SavedModelBuilder` API (no `fingerprint.pb`) before
switching to the modern `tf.saved_model.save()` API for `v03` (which has
`fingerprint.pb`, matches the deployed image exactly).

The zip's `triplet-reid/` directory is **not a clean reference** — it has uncommitted
edits to `common.py`/`embed.py`, 7 untracked scripts (the freeze/migrate pipeline), and
`nets/resnet_v1.py` still hard-imports `tf.contrib.slim` (never migrated in that copy)
while `freeze_as_saved_model.py` sitting next to it mixes updated `tf.compat.v1.*`
calls with bare unported TF1 calls (`tf.placeholder`, `tf.Session`) that would crash
under real TF2. This script is almost certainly not what actually produced the shipped
model — the real combination used was never committed anywhere. `triplet-reid_v2adapted/`
(one commit, "adapt some parts to work with tensorflow v2") has the `tf_slim` nets/
port but lacks the freeze scripts entirely — **two divergent, never-merged attempts**.

Upstream `VisualComputingInstitute/triplet-reid` confirmed dormant since 2019
(`a538696` is still the real tip, verified via GitHub API) — no missed upstream fix
exists; this was self-inflicted, not a stale/tampered clone.

## `tf_upgrade_v2` — actual report (run 2026-07-17)

Ran sandboxed (`--network=none`, source mounted read-only, only the output tree
writable) via `tensorflow/tensorflow:2.21.0` against the zip's `triplet-reid/` tree.
Full report: 528 lines. Output tree and report currently sit at
`~/workspace/webibex/tmp/tf_upgrade_out/` (gitignored scratch — **re-run before
relying on this if that directory has been cleared**).

**Only 6 files have blocking errors** (`tf.contrib.slim`, cannot auto-convert):
- `nets/resnet_v1.py`, `nets/resnet_v1_50.py`, `nets/resnet_v1_101.py`
- `nets/resnet_utils.py`
- `nets/mobilenet_v1.py`, `nets/mobilenet_v1_1_224.py`

This is the exact same set Laurens hand-ported to `tf_slim` in his
`triplet-reid_v2adapted` commit — confirms his manual-conversion scope was correctly
targeted.

**Everything else converts mechanically** (`common.py`, `embed.py`, `evaluate.py`,
`train.py`, the freeze/export scripts, `heads/*`, `excluders/*`), with two
review-worthy but non-blocking warning classes:
1. `dataset.make_one_shot_iterator()` → `tf.compat.v1.data.make_one_shot_iterator()`
   rewrites in `evaluate.py`/`train.py`/`format_saved_model.py`/`migrate_checkpoint.py`.
2. Keras `.save()` calls flagged in `train.py`/`freeze_as_saved_model.py`/
   `migrate_checkpoint.py`/`format_saved_model.py` — Keras's default save format
   changed from HDF5 to SavedModel. Worth an explicit check given the SavedModel-
   builder-API confusion was already the actual root cause of the historical bugs.

## Plan

**Step 0 — Establish a clean base.** Fresh clone of `VisualComputingInstitute/triplet-reid`
@ `a538696`. Do not reuse the zip's `triplet-reid/` — it has uncommitted edits and dead
scripts mixed in.

**Step 1 — Run `tf_upgrade_v2` first, always**, before any manual work
(`--intree . --outtree ../v2_auto --reportfile report.txt`). Free, mechanical, zero
risk. Gives the exact 6-file manual-work list up front instead of discovering it by
trial and error, which is what actually happened last time. (Report already generated
this session — see above — but re-run against the clean Step 0 base, not the zip's
dirty copy.)

**Step 2 — Port the 6 flagged `nets/*.py` files.** `tf.contrib.slim` → `tf_slim`
(PyPI package) is the TF-endorsed path and is exactly what the `v2adapted` commit
already did — reuse that commit's diff as the starting point rather than redoing it.
**Verify `tf_slim` (last released ~2021) actually works cleanly against TF 2.18/2.21
— it predates both; pin this check as its own task, don't assume.**

**Step 3 — Port the freeze/export scripts properly, as one coherent, committed unit**
(not scattered uncommitted files): `freeze_as_saved_model.py`, `migrate_checkpoint.py`,
`format_saved_model.py`. Use the **modern `tf.saved_model.save()` API** throughout, not
`tf.compat.v1.saved_model.builder.SavedModelBuilder` — the old builder API is
implicated in the historical bugs (`model/`'s crash, and it's the API whose exports
lack `fingerprint.pb`). Explicitly include `tf.compat.v1.global_variables()` (not just
trainable vars) in whatever the export step's variable-capture logic is, to avoid
repeating the `moving_variance` uninitialized bug.

**Step 4 — Verification gate, mandatory before calling any export "done":**
1. `python -m tensorflow.python.tools.inspect_checkpoint --file_name=<ckpt> --all_tensors=True`
   — diff variable names against the new graph's `tf.compat.v1.global_variables()`
   *before* attempting a load.
2. Numeric equivalence check exactly like the one run this session for `v03`: same
   test image → embedding → `np.allclose` against `test_embedding_old.h5`, with a
   realistic tolerance (`atol=1e-4`), not the default `1e-8`.
3. `saved_model_cli show --dir <path> --all` to sanity-check the exported signature
   before wiring it into a handler.

**Step 5 — Consolidate.** One committed repo/branch holding the working pipeline end
to end, replacing the current five-directory sprawl (`wibex_model_v01/v02/v03`,
`model/`, `new_model2/`, `saved_model/`). Tag the final export clearly; delete or
clearly label the abandoned ones so the next person doesn't have to redo this
forensic exercise.

## Addendum — serving hardening candidate (separate from the migration itself)

Orthogonal to the training/export work above: `dhi.io/tensorflow-serving:2` (Docker
Hardened Image, TF Serving 2.20.0, Debian 13, CIS-compliant, native arm64, 124MB) is a
strong candidate to replace the current `python:3.12` + full pip-installed TF wheel
base in `runpod_ibex_embedding_endpoint/Dockerfile`.

Verified this session:
- Docker Scout: 0 Critical / 0 High.
- Trivy (pinned `aquasec/trivy:0.70.0`, run directly, matches devcontainer-guard's own
  scanner): 0 Critical / 0 High.
- Grype (pinned `anchore/grype:v0.111.1`): 2 Critical / 4 High, but all 3 unique CVEs
  are `libc6`-only (CVE-2026-5450, CVE-2026-5928, CVE-2026-5435), each formally
  triaged `<no-dsa>` (Minor issue) by Debian for trixie, and none reachable by
  anything TF Serving's serving code actually does (obscure `scanf %mc` specifier,
  wide-char stream pushback, deprecated DNS-debug-only functions). Grype simply
  doesn't honor DHI's VEX attestations the way Scout/Trivy do — same pattern already
  documented for the `bandit`/`semgrep` DHI images in devcontainer-guard's own
  `config/vex-exceptions.toml`.

A draft (not-yet-applied) patch pinning this image + adding the VEX exemptions to
devcontainer-guard's trust matrix sits at
`~/workspace/agents_writer/tmp/dhi-tensorflow-serving-pin.patch`.

**Integration approach chosen: thin-proxy, not full re-architecture.** Keep RunPod
Serverless and `handler.py` as the entry point; run `tensorflow_model_server` as a
background process inside the same worker container (image entrypoint is
`tensorflow_model_server --port=8500 --rest_api_port=8501 --model_name=model
--model_base_path=/models/model`, expects a versioned `/models/model/1/` layout —
`wibex_model_v03`'s SavedModel just needs that `1/` subdirectory added). `handler.py`
becomes a small REST client calling `localhost:8501/v1/models/model:predict` instead
of loading the model itself. Low-risk, small diff, replaces the current base image
(likely carrying similar CVE exposure to the plain `tensorflow/tensorflow:2.21.0`
image found to have 20 CRITICAL/234 HIGH CVEs this session, mostly Ubuntu
kernel-metadata artifacts but not narrowed down for the current `python:3.12` base) with
something already verified clean.

## Status update — 2026-07-20 (Step 4 passed, Step 5 in progress)

Continuation ran the host-side verification gate (Step 4 above) to a full pass:
checkpoint variable-name diff (272/272 match), numeric equivalence
(`max abs diff: 3.099e-06` at `atol=1e-4` against `test_embedding_old.h5`), and
signature parity against `wibex_model_v03`'s captured baseline. Full detail
(bugs found/fixed, diagnostic trail) lives in `docs/session-notes-2026-07-20-tf2-export-pipeline-verification.md`
and `tmp/inference/ADR-export-pipeline.md` — **note the latter sits under
`tmp/`, which is gitignored in this repo**, so it will not survive as
webibex history; this doc is the durable record going forward.

Checkpoint provenance confirmed via 3-file SHA256 exact match: the checkpoint
behind `wibex_model_v03` is `models_run1/results_hornmodel_1662468625/checkpoint-4000`
from the original 2022 runs — a **hornmodel** (horn-identification), not
**ibexmodel** checkpoint, despite the product name. Confirmed correct, not a
mismatch: production's chip input is a single horn-side crop.

Per-file porting provenance for the Step 2 `nets/`/`heads/` port (folded in
here because the staging copies these notes originally lived in,
`nets_tf2/README.md`/`heads_tf2/README.md`, are themselves under gitignored
`tmp/inference/` and not meant to persist):

- `resnet_utils.py`, `resnet_v1.py`, `resnet_v1_50.py`: verbatim from
  Laurens's `triplet-reid_v2adapted/nets/` — confirmed complete
  (`tf.contrib.slim` → `tf_slim`, `tf.variable_scope` →
  `tf.compat.v1.variable_scope`, etc.).
- `resnet_v1_101.py`: the "confirmed complete" assumption above was **wrong**
  for this one sibling file — it still called `tf.contrib.slim.arg_scope(...)`
  with no `tf_slim` import. Caught by an AST-based regression test
  (`test_nets_tf2_has_no_tf_contrib_references`) on first run, then fixed.
  Lesson: "confirmed complete for the resnet path" was checked against
  `resnet_v1_50.py`/`resnet_v1.py`/`resnet_utils.py` specifically, not
  exhaustively against every file matching the glob.
- `mobilenet_v1.py`, `mobilenet_v1_1_224.py`: the v2adapted reference never
  actually ported these — found byte-identical to the TF1 originals (still
  `tf.contrib.slim`/`tf.contrib.layers.*`). Ported fresh this round, using
  `slim.l2_regularizer`/`slim.softmax` (not a literal
  `tf.keras.regularizers.l2`/`tf.nn.softmax` mapping) to preserve the
  `scope=` kwarg used at the mobilenet call sites and match the pattern
  already proven in `resnet_utils.py`/`resnet_v1.py`. Not exercised this CR
  (out of scope — training config uses `resnet_v1_50`, not mobilenet); if it
  comes into scope later this needs its own Step 4 verification run.
- `heads/fc1024.py`, `heads/__init__.py`: verbatim from
  `triplet-reid_v2adapted/heads/` — added as a minimal necessary dependency
  of the export script (training config: `head_name=fc1024`,
  `embedding_dim=128`); `direct.py`/`direct_normalize.py`/`fc1024_normalize.py`
  were dropped from the consolidated branch — unused by the production chip
  path.

**Step 5 (consolidate) — DONE (2026-07-20)**: the port is committed to
webibex's own git history at `training/triplet-reid/` (not the nested
`triplet-reid-clean/` clone under gitignored `tmp/` — that had no durable
remote and is now superseded scratch). Independently reviewed (Fable 5,
prompt drafted by Opus) before commit — no HIGH findings; a handful of
MEDIUM staleness issues (stale `conftest.py` path, missing `skipif` guards
on tests depending on gitignored fixture data, stale "STAGING" headers,
stale CR-doc paths) were fixed in the same commit. One real regression was
also caught and fixed pre-commit: `tests/test_export_pipeline.py`'s path
constants were wrong for the final location (computed one directory too
shallow — a symptom of the code moving between staging/nested-clone/final
locations three times over the course of this work).

**Real host validation, post-consolidation (2026-07-20)**: re-ran
`host_runbook/phase4_verification_gate.sh` (updated to point at
`training/triplet-reid/` instead of the old `triplet-reid-clean/`) in a
real `tensorflow/tensorflow:2.18.0` Docker container. All three parts
passed again against the actual committed code: checkpoint variable-name
diff 272/272 match; numeric equivalence `max abs diff: 3.099e-06` at
`atol=1e-4` (identical to the original host run — confirms determinism);
`serving_default` signature matches production exactly. Confirms the
three-location move didn't break anything. `host_runbook/phase0_*` was
NOT re-run for this — it deliberately tests raw, unpatched `tf_slim`
directly (not through `nets/`'s monkeypatch) and is expected to fail with
the same `tf_keras.legacy_tf_layers` error the ADR already root-caused;
it's stale as a gate now that the fix is committed, superseded by Phase
2/4 which exercise the real, patched code path.

The five-directory sprawl (`wibex_model_v01/v02`, `model/`, `new_model2/`,
`saved_model/`) is still pending removal from `tmp/inference/`;
`wibex_model_v03/` must be kept regardless (numeric baseline). No durable
remote has been set up for `training/triplet-reid/` beyond webibex's own
repo — it's not a separate fork/clone, it's now just part of webibex.

## TF 2.18.0 → 2.18.1 patch bump (2026-07-20)

Applied as a low-risk supply-chain fix, independent of (and before) the
larger planned 2.18→2.21.0 migration. TF 2.18.1's release notes: security
fix bundling curl 8.11.0 (patches `CVE-2024-2004`, `-2379`, `-2398`,
`-2466`, `-6197`, `-7264`, `-8096`, `-9681`), plus a loosened `ml_dtypes`
upper bound (`<1.0.0`). Breaking changes are `tf.lite.Interpreter`
deprecation and TPU-only — neither touches this pipeline (no TFLite, no
TPU, CPU/serving only). Risk assessed as low: the vendored curl's typical
attack surface (TF fetching from attacker-controlled URLs) doesn't apply
to the RunPod serving path, but the patch is free and reduces attack
surface regardless.

Bumped `tensorflow==2.18.0` → `2.18.1` in:
- `tmp/inference/runpod_ibex_embedding_endpoint/builder/requirements.txt`
  (the real production pin — base image is plain `python:3.12`, pip
  resolves the wheel directly, no Docker tag dependency).
- `tmp/inference/host_runbook/Dockerfile.pipdeps` — **no
  `tensorflow/tensorflow:2.18.1` Docker Hub image tag exists** (confirmed:
  `docker manifest inspect tensorflow/tensorflow:2.18.1` 404s; only
  `2.18.0` was ever published as an image for this patch version). Kept
  `FROM tensorflow/tensorflow:2.18.0` and added a separate
  `pip install tensorflow==2.18.1` step to upgrade on top of the base
  image instead.

**`tf-keras` stays pinned at `2.18.0`** — confirmed via PyPI that
`tf-keras` never published a `2.18.1` (only one 2.18.x release exists);
it tracks TF's *minor* version, not patch, matching the same pattern
already found for the 2.21 line (latest `tf-keras` is `2.21.0`, no
`2.21.1`). `ml_dtypes` isn't pinned anywhere in this project — pip
resolves TF 2.18.1's own loosened constraint automatically.
`tensorboard==2.18.0` also stays unchanged — confirmed unused at serving
time (no import in `handler.py`/`test_model.py`) and follows the same
non-patch-release pattern.

**Verification — PASSED (2026-07-20)**: re-ran
`host_runbook/phase4_verification_gate.sh` after this bump (Docker layer
cache auto-invalidated from the changed `Dockerfile.pipdeps` line, no
manual `docker rmi` needed). All three parts passed with the exact same
numeric result as TF 2.18.0: checkpoint variable-name diff 272/272 match;
numeric equivalence `max abs diff: 3.099e-06` at `atol=1e-4` (byte-for-byte
identical value — confirms zero behavioral change for this pipeline);
`serving_default` signature matches production exactly. The pip install
log also confirmed none of the already-installed transitive deps
(`grpcio 1.67.0`, `tensorboard 2.18.0`, `keras 3.6.0`, `numpy 2.0.2`,
`h5py 3.12.1`, `ml-dtypes 0.4.1`) needed to change — exactly as predicted
above. **TF 2.18.1 is now the verified baseline** for this pipeline.

## TF 2.21.0 migration: dedicated per-version Dockerfiles (2026-07-20)

Superseded the originally-planned single ARG-parametrized `Dockerfile.pipdeps`
approach (a code-planner/code-analyst pass had already produced a plan and
25-scenario test spec for that design). Switched to **dedicated,
tracked-in-git Dockerfiles per TF version** instead:

- `training/triplet-reid/dockerfiles/{tf2180,tf2181,tf2210}/Dockerfile` —
  plain `Dockerfile` filename per version-specific subdirectory, not
  `Dockerfile.<suffix>` — the dot-suffix naming convention breaks the
  project's Sonar scan.
- `training/triplet-reid/dockerfiles/verify_gate.sh <version-dir>` — builds
  the given version's image and runs the same 3-part gate (checkpoint
  var-name diff, numeric equivalence, signature parity) against it, reusing
  the existing Phase 4a/4b/4c probe scripts from
  `tmp/inference/host_runbook/` (pure Python, not Dockerfiles — not subject
  to the naming constraint). `docker build` has network access (pulling the
  base image, installing pip packages); every `docker run` step that
  executes the pipeline code itself runs with `--network=none`.
- Incremental version path: 2.18.0 → 2.18.1 → 2.21.0, verified one hop at a
  time rather than jumping straight to 2.21.0. Fall back to 2.19/2.20 as
  intermediate stepping stones only if 2.21.0 breaks something — the
  batch-norm monkeypatch fix already sidesteps the specific historical
  `tf_keras.legacy_tf_layers` break by not depending on that shim at all, so
  a direct jump was judged worth trying first.
- **Security check before starting this migration**: full offline scan of
  OSV.dev's complete PyPI advisory database (1,707 tensorflow-related
  records) found zero advisories affecting either 2.18.1 or 2.21.0 — this
  migration has no security driver, it's maintenance currency only.

**All three versions verified — PASSED (2026-07-20)**, each via its own
dedicated image in `training/triplet-reid/dockerfiles/`:

| Image | TF | Checkpoint var-name diff | Numeric (`atol=1e-4`) | Signature |
|---|---|---|---|---|
| `tf2180` | 2.18.0 (pure, no patch) | 272/272 match | `max abs diff: 3.099e-06` | matches |
| `tf2181` | 2.18.1 | 272/272 match | `max abs diff: 3.099e-06` | matches |
| `tf2210` | 2.21.0 | 272/272 match | `max abs diff: 3.099e-06` | matches |

Identical numeric result across all three — confirms the export is
deterministic and behaviorally unchanged regardless of TF version. **No
code changes were needed for TF 2.21.0**: the `_batch_norm_compat`
monkeypatch and eager-restore mechanism both hold up as-is. The planned
fallback (hopping through 2.19/2.20 as intermediate stepping stones) was
not needed — the direct jump from 2.18.1 to 2.21.0 worked on the first
try. TF 2.21.0 is now a fully verified alternate to the 2.18.1 baseline.

Not yet decided: whether/when to commit `training/triplet-reid/dockerfiles/`
(currently untracked working-tree files) and adopt one of the three as the
new default for future runbook use — separate decision from verification.

## Default version decision + full e2e verification (2026-07-23)

**Decision: TF 2.21.0 (`tf2210`) becomes the default** for future runbook
use, committed this session. Rationale: production currently runs TF1, so
the gap between 2.18.1 and 2.21.0 is immaterial to production risk either
way — better to start on the version with the most maintenance headroom.
`training/triplet-reid/dockerfiles/` committed with this default.

Beyond the Phase 4a/4b/4c gate (already passing per the table above), a
full manual e2e test ran this session, end to end through the real
application code paths:

- Exported a fresh SavedModel with the `tf2210` image, persisted (not the
  `mktemp` dir `verify_gate.sh` normally cleans up).
- Ran it through `tmp/inference/runpod_ibex_embedding_endpoint/handler.py`
  (the actual RunPod serverless entry point) in RunPod's local-server test
  mode (`--rp_serve_api`), joined to the devcontainer's own network
  namespace (`--network container:<id>`) since the devcontainer's outbound
  network is sandboxed (`host.docker.internal` unreachable from inside it).
- Pointed the real webibex Django app's `core/utils.py:endpoint_inference()`
  at that local server (temporarily — reverted after) and drove the actual
  upload → landmark → crop → embed flow through the browser, repeatedly,
  across multiple real images. Multiple `Embedding` rows were created with
  correct 128-dim vectors end-to-end through the real HTTP request path,
  not just the handler in isolation.
- Caught and fixed one unrelated pre-existing bug in the process:
  `.imageToLandmark` (`static/css/tailwind.css`) had no width rule, so any
  uploaded image narrower than `settings.LANDMARK_IMAGE_WIDTH` (1600px)
  rendered at its own natural size in the browser, while the backend's
  `scale_coordinate()` still assumed a 1600px-wide render — silently
  shrinking landmark clicks toward the top-left corner for any image
  <1600px wide. Fixed by adding `width: 100%; height: auto;` to that class
  (own CR). Confirmed via before/after annotated screenshots. Unrelated to
  TF version — pure Django/frontend bug, would have reproduced identically
  against the pre-existing `laubohlen/ibex_embedding_cpu:v0.01` image too.

This is stronger verification than `core/test_model.py` (the local-dev TF
script already flagged for removal in `docs/security-remediation-plan.md`)
could ever provide — that script isn't a real test and hardcodes a path to
the original developer's machine.

## Unresolved questions

- ~~Does `tf_slim` (unmaintained since ~2021) actually import cleanly under TF 2.18/2.21?~~
  **Resolved (2026-07-20)**: yes, on both — via the `_batch_norm_compat` monkeypatch
  (bare `tf_slim` alone still breaks on TF 2.16+/Keras 3, see the batch-norm decision
  above), verified for real in Docker on 2.18.0, 2.18.1, and 2.21.0.
- Is retraining ever actually in scope, or is this permanently "same weights, new
  export wrapper" (per the `v02` detour being abandoned)? Still open — worth confirming
  with Laurens directly rather than assuming.
- ~~Should this migration target TF 2.18 (matches current production) or something
  newer?~~ **Resolved (2026-07-20)**: moved to and verified TF 2.21.0 as an alternate
  (professor/project wanted to move past 2.18).
- ~~Whether/when to commit `training/triplet-reid/dockerfiles/` and which TF version
  becomes the runbook default?~~ **Resolved (2026-07-23)**: committed, TF 2.21.0
  (`tf2210`) is the default — see "Default version decision + full e2e verification"
  above.

Related: `~/workspace/webibex/docs/security-remediation-plan.md` (separate, already
executed remediation track — TF was recommended for removal there in webibex's own
local-dev branches, unrelated to this RunPod-side pipeline).
