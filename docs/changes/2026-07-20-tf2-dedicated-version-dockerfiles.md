# CR: TF export pipeline — dedicated per-version Dockerfiles + verification runner

**What changed:**
- New: `training/triplet-reid/dockerfiles/tf2180/Dockerfile` — TF 2.18.0, pure baseline (no patch).
- New: `training/triplet-reid/dockerfiles/tf2181/Dockerfile` — TF 2.18.1 (base image `2.18.0` + `pip install tensorflow==2.18.1`, since no `2.18.1` Docker Hub tag exists).
- New: `training/triplet-reid/dockerfiles/tf2210/Dockerfile` — TF 2.21.0 (`FROM tensorflow/tensorflow:2.21.0` directly, that tag does exist).
- New: `training/triplet-reid/dockerfiles/verify_gate.sh <version-dir>` — builds the given version's image, runs the 3-part verification gate (checkpoint variable-name diff, numeric equivalence at `atol=1e-4`, `serving_default` signature parity) against it, reusing the existing Phase 4a/4b/4c probe scripts from `tmp/inference/host_runbook/` (unmodified — pure Python, not Dockerfiles).
- Bumped: `tmp/inference/runpod_ibex_embedding_endpoint/builder/requirements.txt` (`tensorflow==2.18.0` → `2.18.1`, the real production pin) and `tmp/inference/host_runbook/Dockerfile.pipdeps` (same bump, via an added `pip install tensorflow==2.18.1` layer on top of the `2.18.0` base image).
- **Status: staged 2026-07-23, pending commit** — TF 2.21.0 (`tf2210`) adopted as the default version — see `docs/tf1-to-tf2-migration-plan.md`'s "Default version decision + full e2e verification" section for the full e2e test that informed the decision.

**Follow-up action — DONE (2026-07-23):** `training/triplet-reid/dockerfiles/` staged with `tf2210` as default, ready to commit.

**Do NOT:**
- Name any new per-version Dockerfile `Dockerfile.<suffix>` (e.g. `Dockerfile.pipdeps`-style) — that naming convention breaks this project's Sonar scan. Use a plain `Dockerfile` filename inside a version-specific subdirectory instead (the pattern already established here).
- Assume `tensorflow/tensorflow:<X.Y.Z>` Docker Hub tags exist for every PyPI-published TF version — confirmed both `2.18.1` and `2.21.1` have no matching image tag (`docker manifest inspect` 404s), while `2.18.0` and `2.21.0` do. Check with `docker manifest inspect` before writing a new version's `FROM` line; don't assume the pip-install-on-top-of-base-image workaround is always needed, or never needed.
- Assume `verify_gate.sh` persists its export the way the older `host_runbook/phase4_verification_gate.sh` does — it currently does NOT (exports to a `mktemp -d` temp dir cleaned up on exit). The 2026-07-23 e2e test worked around this with a separate one-off script (`tmp/inference/host_runbook/e2e_tf2210_manual_test.sh`, gitignored scratch) rather than patching `verify_gate.sh` itself — still an open gap if a persisted export becomes a recurring need.

**Trigger:** whenever adding/testing a new TF version for this pipeline, or re-running verification after a code change to `training/triplet-reid/{nets/__init__.py,export_saved_model.py}`.

**Why:** supersedes an earlier plan (code-planner + code-analyst had already produced a plan and 25-scenario test spec) for a single ARG-parametrized `Dockerfile.pipdeps`-style image. Switched mid-implementation once the Sonar naming constraint surfaced — dedicated per-version files satisfy the constraint directly and let each TF version be verified as an independent, git-tracked artifact rather than a parametrized build. All three versions (2.18.0, 2.18.1, 2.21.0) gate-passed with byte-identical numeric result (`max abs diff: 3.099e-06`), confirming the `_batch_norm_compat` monkeypatch and eager-restore mechanism hold up across the whole 2.18–2.21 range with zero code changes.

**Verify:** `cd training/triplet-reid/dockerfiles && ./verify_gate.sh <tf2180|tf2181|tf2210>` — all three parts (4a/4b/4c) must pass, matching the numeric result documented in `docs/tf1-to-tf2-migration-plan.md`'s "TF 2.21.0 migration" section.

**Rollback:** if still staged/uncommitted, `git restore --staged training/triplet-reid/dockerfiles/` (or delete the directory). If already committed, `git revert`. Neither affects the already-committed `training/triplet-reid/` code or production (RunPod loads a separate `embedding_model/` directory, unrelated to this).
