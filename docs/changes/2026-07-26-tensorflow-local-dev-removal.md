# CR: remove dead local-dev TensorFlow fallback + orphaned SavedModel

**What changed:**
- Modified: `core/utils.py` — deleted `get_tf()`/`_tf` (lazy TF import helper), the commented-out dead TF-import block, `chip_size` (became unused), the `model_is_local` variable, and the two `elif` branches in `embed_new_chip()` that loaded a local TF SavedModel. Branch condition simplified from `if (not database_is_local) and (not model_is_local):` to `if not database_is_local:` — proven logically equivalent, not just similar (see Why).
- Modified: `webibex/settings.py` — deleted `ENDPOINT_LOCALLY = True`.
- Deleted: `core/embedding_model/` — the orphaned ~104MB TF1 SavedModel binary tree (`saved_model.pb`, `variables/variables.data-00000-of-00001`, `variables/variables.index`) it fed; nothing in the codebase referenced it once the two branches above were gone.
- Modified: `.gitignore` — removed a dead commented-out `# core/embedding_model/` line.
- Modified: `.coveragerc` — removed `core/embedding_model/*` from the `omit =` list (nothing left to omit).
- Modified: `tests/core/test_utils_io.py` — edited existing T45/T46 to drop the now-dead `get_tf` patch/assert; added 8 new tests: a parametrized branch-selection-equivalence keystone, 3 error-path tests for the surviving branches (B2-download-returns-None, corrupt/undecodable image bytes, missing local file), a payload-argument-correctness test, and 2 structural `hasattr` regression guards proving `get_tf`/`ENDPOINT_LOCALLY` are actually gone.
- Modified: `docs/security-remediation-plan.md` — marked the "TensorFlow removal" item resolved (append-only, existing doc's own convention).
- Status: committed as `ee1839f`.

**Follow-up action:** none required — `core/test_model.py` (the standalone dev script mentioned in the original finding) was already removed in an earlier, unrelated commit (`2bde17e`); nothing further to clean up.

**Do NOT:**
- Assume this deletion needed "bump and verify" caution — it didn't. `ENDPOINT_LOCALLY` was hardcoded `True` (not `env()`-driven), so `model_is_local` evaluated to `False` in every environment, unconditionally. The deleted branches were logically unreachable dead code, not just rarely exercised.
- Re-add a local-model-load fallback to `embed_new_chip()` without first checking whether `INFERENCE_ENDPOINT_URL_OVERRIDE` (see `docs/changes/2026-07-25-runpod-endpoint-override-and-script-hardening.md`) already solves whatever local-testing need prompted it — that mechanism is what made this removal safe to do now (real e2e tests already exercise the actual RunPod inference container locally via the override, so a local in-process TF model is redundant).

**Trigger:** none expected — this is a closed removal. If a future need for local model inference re-emerges, evaluate `INFERENCE_ENDPOINT_URL_OVERRIDE` + a locally-running inference container first (the pattern that made this removal viable) before reintroducing an in-process TF load path.

**Why:** the deferred conversation with the original developer (Lauren) happened and RunPod access is confirmed; the user's call (common-sense, not a new investigation) was that since real e2e manual tests already exercise the actual RunPod inference container via `INFERENCE_ENDPOINT_URL_OVERRIDE`, the old local-dev TF1 fallback was no longer needed for anything. Verified safe via a full planning-TDD pipeline (code-planner → code-analyst → code-executioner) plus a Fable5 adversarial trace that independently re-derived the "always unreachable" claim from the original pre-deletion source (not the CR's framing) and traced all 6 adversarial candidates to `NOT-A-BYPASS` — including confirming the surviving branch bodies are byte-identical to the original via exhaustive diff-hunk accounting.

**Verify:** `.venv/bin/python -m pytest tests/core/test_utils_io.py -v` (19 pass, 1 xfailed); full suite `.venv/bin/python -m pytest -q` (167 passed, up from 159 — 1 skipped, 1 xfailed); `ruff check core/utils.py webibex/settings.py tests/core/test_utils_io.py` clean (0 new findings; 9 pre-existing findings elsewhere in `core/utils.py` confirmed unrelated).

**Rollback:** `git revert ee1839f` — restores `get_tf()`/`ENDPOINT_LOCALLY`/the local-TF branches and re-adds the 3 deleted binary files from git history (git revert restores deleted blobs from the parent commit).
