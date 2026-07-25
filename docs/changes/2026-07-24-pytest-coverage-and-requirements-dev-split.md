# CR: add pytest/pytest-django test suite; split test deps into requirements-dev.txt

**What changed:**
- Added: root `conftest.py` (env-var setdefaults + `DJANGO_SETTINGS_MODULE`, `collect_ignore` for `db_management/test.py`, `no_network` autouse guard fixture), `pytest.ini` (`testpaths = core simple_landmarks`), `.coveragerc`.
- Added: `core/tests/` package (8 files, 109 tests total after the follow-up bug-fix commit) and `simple_landmarks/tests/__init__.py` — replaces the deleted `core/tests.py`/`simple_landmarks/tests.py` (unmodified `startapp` boilerplate).
- Deleted: `core/test_model.py` (no assertions, hardcoded path to another developer's machine).
- Added: `requirements-dev.txt` — `pytest==9.1.1`, `pytest-cov==7.1.0`, `pytest-django==4.12.0`.
- Modified: `requirements.txt` — unchanged from before this CR (the 3 test packages were briefly added here, then moved to `requirements-dev.txt` before commit).
- Status: committed as `2bde17e`. 104 tests initially, 109 after a follow-up bug-fix round (`6edb044`) same session.

**Follow-up action:** for local dev/testing, install both files: `pip install -r requirements.txt -r requirements-dev.txt` (or `uv pip install -r requirements.txt -r requirements-dev.txt`). Running `pytest` requires `ENVIRONMENT` to NOT be `"development"` — `conftest.py` defaults it to `"test"` already, no manual env setup needed for a normal `pytest` invocation from the repo root.

**Do NOT:**
- Add `pytest`/`pytest-django`/`pytest-cov` (or any future dev/test-only package) directly to `requirements.txt`. Railway's default Nixpacks builder (no `railway.toml`/`nixpacks.toml` in this repo) only installs `requirements.txt` for the actual deploy — anything test-only belongs in `requirements-dev.txt` instead. This was caught and corrected before the first commit; see `docs/security-remediation-plan.md`'s "Railway deployment hardening" TODO for the tracked follow-up on a more native dev/prod dependency-group split (`railway.toml`/`pyproject.toml`).
- Set `DJANGO_SETTINGS_MODULE` via the `[pytest]`/`[tool:pytest]` ini option in `pytest.ini`. This causes `pytest-django`'s `pytest_load_initial_conftests` hookimpl to eagerly touch `django.conf.settings` — which can run before the root `conftest.py`'s own `os.environ.setdefault(...)` calls have executed, depending on plugin hook-registration order, and crashes with `ImproperlyConfigured: Set the AWS_ACCESS_KEY_ID environment variable` (or similar) even though `conftest.py` looks correct. Keep `DJANGO_SETTINGS_MODULE` as an `os.environ.setdefault` call inside `conftest.py` instead — pytest-django then defers its settings check to `pytest_configure`, which always runs after all conftests are loaded.
- Trust `.venv`'s currently-installed package versions to match `requirements.txt`/`requirements-dev.txt` without checking — this environment's `.venv` has drifted from the pinned manifests before (see the `2026-07-24` dependency-bump CR/session notes for a concrete case where `.venv` had `setuptools==83.0.0` installed while `requirements.txt` still pinned `78.1.1`, and a test-only shim papered over the resulting `pkg_resources` import failure without anyone noticing until the pin was actually bumped to match).

**Trigger:** any new source file under `core/`/`simple_landmarks/` that needs test coverage (extend `core/tests/`, don't create a new top-level test directory); any new dev/test-only Python dependency (goes in `requirements-dev.txt`, not `requirements.txt`); any future `pytest.ini`/`conftest.py` edit that reintroduces the `DJANGO_SETTINGS_MODULE`-via-ini pattern.

**Why:** the app had zero real test coverage before this (`core/tests.py`/`simple_landmarks/tests.py` were empty `startapp` stubs, `core/test_model.py` had no assertions) — a pending dependency-bump round was explicitly paused on the reasoning that further bumps shouldn't rely on manual-smoke-test-only verification. This CR unblocks that (see the same-session dependency-bump CR) and gives future changes to `core/utils.py`/`core/models.py`/`core/middleware.py`/views/middleware a real regression net.

**Verify:** `.venv/bin/python -m pytest -q` from the repo root — expect `109 passed, 1 xfailed` (as of `6edb044`; the 1 `xfail(strict=True)` deliberately pins a known, still-unfixed AVIF/`load_image` decode bug, not a broken test). `pytest --co -q` to confirm 0 collection errors and the expected test count if verifying after further changes.

**Rollback:** `git revert 2bde17e` (and `6edb044` if the follow-up bug-fix commit is also being rolled back) restores the pre-coverage state — `core/tests.py`/`simple_landmarks/tests.py`/`core/test_model.py` return, `conftest.py`/`pytest.ini`/`.coveragerc`/`requirements-dev.txt`/`core/tests/`/`simple_landmarks/tests/` are removed.
