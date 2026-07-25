# CR: test directory restructure + pyright/ruff dev tooling

**What changed:**
- Moved `core/tests/*` and `simple_landmarks/tests/*` into a top-level
  `tests/` directory mirroring the source layout: `tests/core/` (8 pure
  `core`-domain test files), `tests/webibex/` (`test_infra.py`,
  `test_settings_security_hardening.py`, `test_manual_login_logout_check.py`
  — these test `webibex/settings.py`, not `core`, so this is their correct
  home, not a mistake), `tests/simple_landmarks/` (empty placeholder,
  unchanged content, kept per explicit user instruction for now).
  `core/tests/conftest.py` moved to top-level `tests/conftest.py` (whole
  file, unchanged content — fixtures now shared across all subpackages).
  All moves via `git mv`; `git diff --stat -M` confirms proper renames for
  every non-empty file (git's rename-detection pairs the zero-byte
  `__init__.py` placeholders onto mismatched arrows since it can't
  distinguish identical empty content — a cosmetic diff artifact, not a
  real delete+add; actual filesystem locations all match).
- `pytest.ini`: `testpaths = core simple_landmarks` → `testpaths = tests`.
  `addopts` (`--cov=core --cov=simple_landmarks --cov=webibex`) left
  unchanged — those are source-coverage targets, independent of where the
  test files themselves live.
- Root `conftest.py`: path-reference comments/docstring updated to point at
  the new `tests/conftest.py` location. No executable-logic change.
- New: `requirements-dev.txt` gained `pyright==1.1.411` and `ruff==0.16.0`.

**Follow-up action:**
- 8 pre-existing `ruff` findings surfaced (first-ever ruff run on this
  codebase, no `ruff.toml` exists) — logged as their own TODO in
  `docs/security-remediation-plan.md` ("ruff baseline findings, first run
  on this codebase"), deliberately left unfixed this session to keep this
  diff's blast radius to the actual requested changes.
- Consider setting up a `ruff.toml` matching the `python.md`-recommended
  baseline config (would enable `E402` and make several pre-existing
  `# noqa: E402` comments meaningful again) as its own dedicated CR.

**Do NOT:**
- Assume `tests/simple_landmarks/__init__.py` being just an empty
  placeholder is a bug — it's deliberate, the user said "we'll add more
  tests maybe we'll rename it or remove: we'll see."
- Re-run `pyright`/`ruff` against the full project expecting a clean scan —
  `pyright` currently reports 66 diagnostics that are `django-stubs`-absence
  noise (`.objects` manager access, Django test `Client` response typing),
  a pre-existing whole-codebase gap unrelated to this change. Only 1 real
  diagnostic exists, already tracked separately
  (`scripts/run_local_e2e_server.py:85`).
- Expect `uv audit` to work in this project — it requires a
  `pyproject.toml`-based uv project; this repo uses plain `requirements.txt`
  and errors with `No pyproject.toml found`.
- Expect a full `pip-audit` scan to succeed from inside this devcontainer —
  the PyPI allowlist here only covers `pyright`/`ruff` specifically (added
  by the user for this CR), not the rest of the dependency tree.

**Trigger:** any future test file addition (use the new `tests/<package>/`
mirrored layout, not co-located `<package>/tests/`); the next dedicated
lint-cleanup pass (see the ruff TODO above); or a `ruff.toml` setup CR.

**Why:** user preference — "I don't like too much to keep the tests within
the same folder with the source code: it would make sense to collect them
under a `./tests/` folder, structured like the source code folders." The
`pyright`/`ruff` addition came from `/post-production`'s tier-4 gate
(triggered by the same session's auth/session-hardening settings change)
needing real lint/type-check tooling rather than the devcontainer's
`uv-guard`-blocked ad-hoc `uv run --with` installs.

**Verify:**
```bash
uv run pytest -q
# → 156 passed, 1 skipped, 1 xfailed, 0 failed (identical to the pre-move
#   baseline, confirmed twice this session)
.venv/bin/ruff check <changed files>
.venv/bin/pyright --pythonpath .venv/bin/python --level warning <changed files>
```

**Rollback:** `git revert` the commit — all test moves are plain `git mv`
renames, `pytest.ini`/`conftest.py` changes are path-string-only, and
`requirements-dev.txt`'s two new lines can be dropped independently of the
test restructure if only one half needs reverting.
