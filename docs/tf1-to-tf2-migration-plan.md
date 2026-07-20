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

**Step 5 (consolidate) — still open**: `triplet-reid-clean/` (clean clone of
`VisualComputingInstitute/triplet-reid @ a538696`, now on a checked-out
branch `tf2-export-pipeline` with the Step 2/3 port staged) still needs a
commit + tag. Its only remote is the upstream clone origin,
`https://github.com/VisualComputingInstitute/triplet-reid.git` — read-only
(no push access, and not an intended destination for this work); the
`tf2-export-pipeline` branch has no remote/tracking of its own, so right now
this consolidated code exists only locally, inside webibex's gitignored
`tmp/` tree. No durable remote home has been decided yet (fork upstream?
fold into webibex proper?) — worth resolving before this is considered truly
"consolidated." The five-directory sprawl (`wibex_model_v01/v02`, `model/`,
`new_model2/`, `saved_model/`) is pending removal from `tmp/inference/`;
`wibex_model_v03/` must be kept regardless (numeric baseline).

## Unresolved questions

- Does `tf_slim` (unmaintained since ~2021) actually import cleanly under TF 2.18/2.21?
  Untested — first thing to verify in Step 2.
- Is retraining ever actually in scope, or is this permanently "same weights, new
  export wrapper" (per the `v02` detour being abandoned)? Worth confirming with
  Laurens directly rather than assuming.
- Should this migration target TF 2.18 (matches current production) or something
  newer? No strong reason found to move past what's already deployed and verified.

Related: `~/workspace/webibex/docs/security-remediation-plan.md` (separate, already
executed remediation track — TF was recommended for removal there in webibex's own
local-dev branches, unrelated to this RunPod-side pipeline).
