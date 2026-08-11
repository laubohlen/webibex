# webibex Roadmap

Single entry point for this repo's active work tracks. Each track has its own
detailed plan doc (kept in place, not merged in full here — both grow with
forensic/session detail that would make one giant file harder to scan, not
easier). This doc is the short version: what each track is, current status,
and where to go for the full story.

## Track: Security & Tech Debt Remediation

Full detail: [`docs/security-remediation-plan.md`](security-remediation-plan.md)

Started 2026-07-07 — an ongoing security/tech-debt audit and remediation
effort for webibex itself (the live Django app on Railway).

**Done**: routine CVE bumps (`480607b`, 2026-07-09), JS build-tooling
cleanup, region-visibility fix (2026-07-24), auth/session hardening settings
(`95b2932`, 2026-07-25) + follow-up test coverage (`978f785`, 2026-07-26),
TensorFlow local-dev removal (`ee1839f`, 2026-07-26), `moto`-based S3-mock
test tier for `core/b2_utils.py` (`2cbfb84`, 2026-07-26) — all 4 functions
covered, verified with real execution (100% coverage on that file), plus a
`pytest_collection_modifyitems` hardening hook closing a marker-bypass gap
found during adversarial review.

**Blocked**: boto3/botocore/urllib3 version-bump triangle — needs a
dedicated B2 test bucket before it can start. The moto tier confirms
`moto==4.2.14` is compatible with the exact pinned triangle, but doesn't
unblock this itself (moto simulates S3, not Backblaze B2's actual
behavior) — see `security-remediation-plan.md`'s landmine section.

**Open / deferred**: CI scaffold, region-detail coordinate-exposure UX
question (still open; shared visibility itself confirmed intentional by the
professor 2026-08-08), IDOR fix — `location-id`/`oid` ownership checks, now
unblocked with confirmed owner-only semantics, ready to implement — ruff
baseline config, logging/observability pass, documentation gaps,
`allauth.mfa` evaluation. "Delete" tool: confirmed by the professor
(2026-08-08) to stay hidden, no real-implementation work needed now.

## Track: TF1 → TF2 Migration (`triplet-reid` pipeline)

Full detail: [`docs/tf1-to-tf2-migration-plan.md`](tf1-to-tf2-migration-plan.md)

Started 2026-07-17 — a separate effort making the `triplet-reid`
training/export pipeline (RunPod-side serving, `ibex_stambecchi`-related)
reproducible. **Not** a production fix — production was independently
verified sound on TF1 before this track began; the goal is a clean,
committed pipeline that reproduces the shipped model on purpose, not by
accident.

**Done**: root-caused the historical export bugs (graph/export mechanics,
not a retrain issue); ported `nets/`/`heads/` to TF2 via `tf_slim`;
consolidated into `training/triplet-reid/` (committed to webibex's own git
history); verified across TF 2.18.0/2.18.1/2.21.0 with identical numeric
results; **TF 2.21.0 (`tf2210`) is the default** (decided + full e2e
verified 2026-07-23, including driving the real Django upload → landmark →
crop → embed flow through the browser against a locally-served export).

**Open**: whether retraining is ever actually in scope, or this stays
permanently "same weights, new export wrapper" (needs confirming with
Laurens directly); the `dhi.io/tensorflow-serving:2` hardened-base-image
swap for the production RunPod serving container (draft patch not yet
applied, separate from the migration itself).

**New from a devcontainer-guard cross-project image-usage audit (2026-08-04)** — handoff
originated as gitignored `tmp/2026-08-04-dhi-python-hardening-and-subprocess-audit.md`;
folded in full here per that doc's own instruction ("fold into a real TODO tracker ... safe
to delete once read") so the findings survive independent of the untracked file:
- **Which worker Dockerfile is actually live is unresolved from repo content alone.** Two
  candidates exist, both under gitignored/untracked `tmp/inference/`, with no repo-level signal
  (CI, compose, docs) pointing at one over the other: `tmp/inference/runpod_ibex_embedding_endpoint/Dockerfile`
  (`python:3.12`) vs `tmp/inference/ibex_identification/Dockerfile` (`runpod/base:0.4.0-cuda11.8.0`,
  GPU/CUDA). Check the actual RunPod dashboard / endpoint config for which image the
  `RUNPOD_ENDPOINT_ID` env var (read by `core/utils.py`'s `endpoint_inference()`, called from
  `embed_new_chip()`) actually points at before any hardening work on either — and worth moving the confirmed-live one out of
  gitignored `tmp/` into a tracked location regardless, since a live prod image's source living
  only in scratch space is itself a gap. **If the live worker's inference actually routes
  through TF Serving rather than loading TensorFlow directly in the `python:3.12` image, this
  item and the `dhi.io/tensorflow-serving:2` swap above may be the same piece of work — check
  before doing both.**
- **DHI base migration, conditional on the above**: if the CPU-only `python:3.12` worker is
  live, same pattern as the `tensorflow-serving:2` swap — select a `dhi.io/python:3.12.x` tag,
  run it through a real CVE check (Scout + Trivy at minimum) before switching, don't assume
  "DHI = automatically fine." If the CUDA worker is live, DHI likely has no CUDA-capable Python
  base (needs checking) — the realistic path may be a distroless/minimal final stage layered on
  top of the CUDA runtime rather than a full base swap, since the CUDA driver/toolkit stack
  itself isn't something DHI is likely to cover. Cross-reference
  [`docs/tf1-to-tf2-migration-plan.md`](tf1-to-tf2-migration-plan.md), which already proposes
  `dhi.io/tensorflow-serving:2` (124MB) for the TF-serving side with a scan already done (Scout
  0C/0H, Trivy 0C/0H, Grype 3 libc6 CVEs triaged not-reachable) and a draft patch — check that
  plan before duplicating the base-swap work here.
- **subprocess/exec-capability audit**: can spawning subprocesses be blocked for the live worker?
  Literally removing the `subprocess` stdlib module isn't feasible (breaks `multiprocessing`,
  `asyncio` internals, and an unknown tail of dependencies that call it defensively). Real path:
  (1) static grep of `handler.py` + all installed deps for `subprocess`/`os.system`/`os.exec*`
  (GPU-driver-detection code — e.g. shelling out to `nvidia-smi` — is worth explicit suspicion,
  same as any ML-stack dependency); (2) dynamic verification via `strace -f -e
  trace=execve,fork,clone` (or seccomp audit mode) during a real inference request, to see what
  actually fires vs. what static grep merely finds; (3) if clean, enforce with a seccomp profile
  (`docker run --security-opt seccomp=<profile.json>`) denying `execve`/`fork`/`vfork`/relevant
  `clone` flags — testable, reversible, blocks the syscall regardless of API used — noting
  RunPod's serverless runtime may constrain how much control exists over container launch flags
  compared to a self-managed deploy, worth checking what RunPod actually exposes before
  committing to this as the enforcement mechanism; (4) cheaper complementary check — does the
  chosen base image ship a shell (`docker run --rm --entrypoint /bin/sh <image> -c true`)? GPU/
  CUDA bases may not strip shells the way CPU-only DHI images often do, so this may matter more
  here than elsewhere. If GPU-driver-detection code turns out to genuinely need exec, that's a
  real finding — the hardening decision becomes a judgment call on scoping a narrow seccomp
  allow-list, not something to force through.

## Related but out of scope for both tracks above

- `docs/changes/*.md` — individual completed-CR docs (what changed, why,
  how to verify/roll back), one per meaningful change. Not a roadmap, a
  changelog.
- `docs/session-notes-*.md` — dated, ephemeral working notes consumed by
  the next session that touches the same area; not meant to be a durable
  reference (superseded by the tracks above or by CR docs once their
  findings are folded in).
