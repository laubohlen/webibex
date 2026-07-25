# CR: add INFERENCE_ENDPOINT_URL_OVERRIDE + harden start_local_rp_server.sh

**What changed:**
- Modified: `core/utils.py:endpoint_inference()` — reads `INFERENCE_ENDPOINT_URL_OVERRIDE` at call time (not a function-default-argument like the existing `endpoint_id`/`endpoint_api_key` params); when set to a non-empty string, used verbatim as the endpoint URL; otherwise falls back to the existing real `api.runpod.ai` URL construction, byte-identical to before.
- Modified: `core/tests/test_utils_io.py` — 3 new tests (T47-T49) asserting on the actual captured URL via the `mock_runpod` fixture.
- New: `training/triplet-reid/dockerfiles/start_local_rp_server.sh` — replaces a gitignored scratch script (`tmp/inference/host_runbook/start_local_rp_server.sh`, deleted) that had a hardcoded, ephemeral `DEVCONTAINER_ID`. Now takes positional args (`devcontainer_id` required, `image_tag`/`port` defaulted), full guard chain (docker missing/unreachable, container not-found/not-running, image not-found, invalid port), `--` option-injection defenses.
- New: `training/triplet-reid/dockerfiles/tests/test_start_local_rp_server.sh` — bash test harness (fake-docker-stub-on-PATH technique), 14 scenarios, run via `bash` directly (not wired into pytest — first shell-test file in this repo).
- Status: committed as `60082a8`.

**Follow-up action:** for local/manual e2e testing that needs real embeddings, set `INFERENCE_ENDPOINT_URL_OVERRIDE=http://localhost:8001/runsync` (or wherever `start_local_rp_server.sh` is listening) alongside the other dev env vars.

**Do NOT:**
- Add `endpoint_id`/`endpoint_api_key` as function-default-argument reads for `INFERENCE_ENDPOINT_URL_OVERRIDE` — those two existing params are read once at `core.utils` module-def time, which would freeze the override at import time and defeat its purpose (toggling per-test/per-session without reimporting). Keep it as a call-time `env()` read inside the function body.
- Guess a devcontainer's ID from the host via `docker ps -f name=...` — devcontainer-guard-managed containers carry no repo-identifying label, only a launcher-session name (`claude-devcontainer`/`claude-devcontainer-2`) that's ambiguous across concurrent sessions. Get the correct ID via `hostname` run *inside* the target devcontainer instead (exact, since Docker sets a container's hostname to its own short ID by default). Confirmed this exact failure mode: a `docker ps` name-based guess tunneled to the wrong container this session.
- Assume `ibex-embedding-tf2210:e2e-test` is already built locally — the hardened script's own guard will error clearly if not; build it first via `tmp/inference/host_runbook/e2e_tf2210_manual_test.sh` (host-side, needs real Docker).

**Trigger:** any future manual e2e/Playwright test that needs the real embedding pipeline exercised locally rather than hitting the production RunPod endpoint (which will fail with an SSL error from inside this devcontainer's sandboxed network regardless of credentials).

**Why:** `core/utils.py:embed_new_chip()` always calls the real `api.runpod.ai` cloud endpoint — `settings.ENDPOINT_LOCALLY` is hardcoded `True`, and the inverted boolean logic in `embed_new_chip()` means the "local model" branch never actually triggers regardless of `ENVIRONMENT`. This surfaced as a real `ValueError`/SSL error during a manual walkthrough. The override makes local testing possible without touching production code paths or their default behavior.

**Verify:** `.venv/bin/python -m pytest core/tests/test_utils_io.py -k endpoint_inference -q` (7 tests, all green); `bash training/triplet-reid/dockerfiles/tests/test_start_local_rp_server.sh` (14/14 pass); `shellcheck --enable=all` clean on both new `.sh` files.

**Rollback:** `git revert 60082a8` — restores the pre-override `endpoint_inference()` behavior and removes both new shell files (the deleted gitignored original is not restored by a revert, since it was never tracked).
