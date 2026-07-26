# CR: add moto-based in-process S3-mock test tier for core/b2_utils.py

**What changed:**
- Added: `tests/core/test_b2_utils_moto.py` — 15 new tests covering all 4 `core/b2_utils.py` functions (`get_b2_resource`, `download_file`, `delete_files`, `check_file_exists`) against an in-process moto S3 mock, distinct from the existing `unittest.mock`-based tier (`mock_b2` in `tests/core/test_utils_io.py`, which only stubs `download_file` at the function boundary). Exercises the real boto3/botocore client code paths instead.
- Modified: `requirements-dev.txt` — added `moto==4.2.14` (exact pin).
- Modified: `pytest.ini` — registered two new markers: `moto_s3` (gates the new tier) and `live_b2` (scaffold only, for a future real-Backblaze-B2 integration tier — zero tests use it yet).
- Modified: `conftest.py` (root) — two additions:
  - `MOTO_S3_CUSTOM_ENDPOINTS`/`AWS_DEFAULT_REGION` env defaults added to the existing `os.environ.setdefault` bootstrap block.
  - The autouse `no_network` fixture (blocks real `boto3.resource`/`requests.post` calls suite-wide) now conditionally skips only the `boto3.resource` patch when the current test carries `@pytest.mark.moto_s3` — the `requests.post` patch stays unconditional. Hardened with a `pytest_collection_modifyitems` hook that fails collection if a `moto_s3`-marked test doesn't also request the `moto_b2` fixture (closes a bypass-switch gap found during adversarial review — see Why).
- Modified: `tests/conftest.py` — added the `moto_b2` fixture (starts moto's `mock_s3()`, creates the `test-bucket` bucket pinned to `us-east-1`, yields a boto3 S3 client).
- Modified: `docs/security-remediation-plan.md` — dated note added to the boto3/botocore/urllib3 landmine section: moto confirms compatibility with the exact pinned triangle but doesn't unblock that landmine itself.
- Unchanged (verified): `requirements.txt`, `core/b2_utils.py` — the actual production code and its prod dependency manifest are untouched.
- Status: committed as `2cbfb84`.

**Follow-up action:** none required for this CR. Two items deferred to future work: (1) `boto3-stubs`/`botocore-stubs`/`django-stubs` are not installed — pyright can't resolve boto3's dynamically-generated resource/client attributes or Django's `.objects` manager, producing known false-positive diagnostics (7 in this diff, all confirmed stub-gap, not real bugs); (2) the real boto3/botocore/urllib3 version-bump triangle is still blocked on a dedicated B2 test bucket.

**Do NOT:**
- Assume `moto_s3` + `boto3.resource()` alone (without also requesting the `moto_b2` fixture) is safe just because the marker exists — the marker only tells `no_network` to stand down, it does not itself start moto. The new collection-time hook enforces this, but if it's ever removed or bypassed, a `moto_s3`-marked test with no active moto context would make a **real, unmocked network call**.
- Bump `moto` to 5.x independently — it's deliberately paired with the future boto3/botocore/urllib3 triangle bump (moto's simulated-S3 fidelity is calibrated to the botocore version it mocks; bumping moto alone while staying on the old pinned botocore risks moto simulating semantics the actual pinned client doesn't generate).
- "Fix" `check_file_exists`'s `None`-instead-of-`False` bug via this test file — `test_check_file_exists_error_paths_return_none_bug_pin` deliberately pins the current (buggy) behavior; changing the assertion to expect `False` would mask the bug, not fix it. A real fix is a separate, out-of-scope decision.

**Trigger:** whenever `core/b2_utils.py` gains new functions or new error branches — add corresponding moto-backed tests here rather than only unittest.mock-patching. Also relevant if the boto3/botocore/urllib3 triangle bump ever proceeds (see security-remediation-plan.md) — re-verify this tier against the bumped versions at the same time as the moto 5.x bump.

**Why:** `core/b2_utils.py` was only ~36% covered by the existing function-level mock (`mock_b2`), with real boto3 client construction, signature config, and `ClientError` branching entirely untested. moto gives realistic S3 IO mocking as a distinct tier without needing real network access or a real B2 bucket. Verified with real local execution after the sandbox's egress proxy was scoped to allow `pypi.org`+`files.pythonhosted.org` (same allowlist pattern already used for `pyright`/`ruff`) — all 15 tests pass, `core/b2_utils.py` coverage 36%→100%. The `no_network` guard change went through a two-stage adversarial review (Opus authors a targeted prompt, Fable5 executes the hunt against live code, per this project's established pattern) specifically targeting the marker-gated bypass; it found one real advisory (the bypass-switch-with-no-misuse-detection gap described above), fixed in the same session and re-verified (184/184 tests still pass, hook confirmed to reject a throwaway misuse test).

**Verify:** `source .venv/bin/activate && pytest -q` (184 passed, 1 skipped, 1 xfailed, 0 failed, 0 errors); `pytest --cov=core --cov-report=term-missing tests/core/test_b2_utils_moto.py` shows `core/b2_utils.py` at 100%; `ruff check tests/core/test_b2_utils_moto.py tests/conftest.py conftest.py` (0 new findings); `git diff --stat requirements.txt core/b2_utils.py` (empty — confirms R6).

**Rollback:** `git revert 2cbfb84` — restores the prior `conftest.py`/`tests/conftest.py`/`pytest.ini`/`requirements-dev.txt` state and removes the new test file. No production code or production dependency changes to unwind.
