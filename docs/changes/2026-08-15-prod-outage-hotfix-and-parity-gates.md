# CR: production outage hotfix + prod-dependency-parity gates + all-views smoke suite

**What changed:**

## 1. Merge: `db-restore-drill` → `main`

Merge commit `80d7b42`. Brings `scripts/db_restore_drill.py`, its full test
suite (`tests/scripts/`), and related docs/roadmap updates onto `main`.
(`pyrightconfig.json` was already identical on both merge parents before the
merge — it arrived on `main` via a separate path prior to this session, the
merge didn't introduce it.) Most of the branch's commits predate this session
(see `docs/changes/2026-08-09-db-restore-drill.md`) — its final commit,
`6e8b8e0` ("land roadmap/deploy-platform-checklist updates, sync
security-remediation-plan with main, ignore .worktrees"), was made during
this session as part of resolving the dirty working tree described below.

Merge required resolving one real conflict in
`docs/security-remediation-plan.md` (both `main`'s `fix/unauthenticated-views`
branch and `db-restore-drill` had independently extended the file since their
common ancestor) — resolved by taking `main`'s side (confirmed the fuller,
later-dated write-up, a strict superset of the other branch's uncommitted
draft). Dirty working-tree state present at merge time (uncommitted
`docs/roadmap.md`/`docs/security-remediation-plan.md` edits, an untracked
stale `docs/pre-deploy-checklist.md`) was resolved before merging: safe edits
committed, stale/superseded edits discarded (verified byte-for-byte subset of
what `main` already had), colliding untracked file removed.

## 2. Production outage hotfix

**Root cause**: `core/b2_utils.py:8-9` imported `mypy_boto3_s3` (the
`boto3-stubs[s3]` type-stub package, dev-only, listed only in
`requirements-dev.txt`) unconditionally at module level, for two type
annotations only (`S3ServiceResource` return type,
`list[ObjectIdentifierTypeDef]` variable annotation) — never used at runtime.
Landed 2026-07-27 (`863f359`) but never exercised against a
`requirements.txt`-only install before this deploy: local dev, pytest, and
pyright all install `requirements.txt` + `requirements-dev.txt` together (per
`requirements-dev.txt`'s own header comment), so the dev-only stub was always
present locally. Railway's buildpack installs `requirements.txt` alone. No CI
workflow and no Dockerfile exist in this repo to mirror that prod-only
install — nothing else would have caught it either. Every request touches the
import chain `core/views.py` → `core/utils.py` → `core/b2_utils.py`, so every
request 500'd with `ModuleNotFoundError: No module named 'mypy_boto3_s3'`.

**Fix** (`f3abc9f`): moved both imports under `if TYPE_CHECKING:`, quoted the
two referencing annotations as forward references (`"S3ServiceResource"`,
`list["ObjectIdentifierTypeDef"]`). No `requirements.txt` change needed.
Repo-wide sweep found no other production-reachable instance of this pattern
— `scripts/db_restore_drill.py` already does this correctly (`TYPE_CHECKING`
guard + lazy `try/except ImportError` at call time for its own dev-only dep,
`testcontainers`); `training/triplet-reid/tests/*.py` import `pytest` but
that tree isn't wired into `core/`/`webibex/` (separate, undeployed pipeline).

**Verification, in increasing order of fidelity**:
1. Reproduced the exact `ModuleNotFoundError` locally, no network/fresh venv
   needed: `sys.modules['mypy_boto3_s3'] = None` (+ the `.type_defs`
   submodule key) before `import core.b2_utils` — confirmed crash pre-fix,
   clean import post-fix.
2. Physically moved `mypy_boto3_s3` + its dist-info out of
   `.venv/lib/python3.13/site-packages` (true prod-only deps), ran
   `manage.py check` (clean), `manage.py migrate --run-syncdb` (clean),
   started `manage.py runserver`, `GET /` → HTTP 200, real rendered HTML, no
   traceback in the server log. Stub package restored afterward, confirmed
   importable again.
3. Live: `curl -I https://wibex.up.railway.app/` → HTTP 200,
   `Content-Length: 2544` — matches the local prod-parity test byte-for-byte.

**pyright**: zero new errors (the one pre-existing error on this file,
`django-environ`'s missing `default=` stub, is unchanged, just shifted by the
added `TYPE_CHECKING` block).

## 3. Prod-dependency-parity gates + all-views smoke suite (`717bf8b`)

Follow-up hardening so this class of bug can't reach production undetected
again. Full planning-TDD pipeline (code-planner → code-analyst, 42-scenario
spec → code-executioner); plan + spec saved at
`tmp/test-spec-b2-utils-outage.md` (not committed, scratch artifact).

**New files**:
- `tests/core/test_b2_utils_type_checking_guard.py` — regression test for the
  `TYPE_CHECKING` fix itself. Poisons `mypy_boto3_s3` *and its cached
  submodules* (poisoning only the parent doesn't block an already-cached
  `from parent.sub import X` — a real false-negative vector, verified live),
  busts the module cache, proves clean import post-fix, self-verifies by
  proving the same poison DOES raise against an un-guarded control module.
- `tests/webibex/test_prod_dependency_parity.py` — fast, always-on gate
  (default suite, no opt-in marker). Derives the dev-only importable-name set
  as a **complement** (all installed top-level names − requirements.txt
  transitive closure − first-party repo names − non-identifiers), not by
  inverting `packages_distributions()` against literal
  `requirements-dev.txt` names — the naive approach was verified this
  session to miss `mypy_boto3_s3` entirely (`boto3-stubs` has no installed
  dist-info; the real stub package is the separate, undeclared transitive
  dist `mypy-boto3-s3`). Sweeps every module under
  `core`/`webibex`/`simple_landmarks`/`db_management`, poisoned per
  dev-only name, asserts clean import. Excludes
  `db_management.populate_created_at_field` with a documented reason (does
  real DB writes at import time, always raises on live data — confirmed by
  actually importing it). Includes a control/meta test proving the sweep
  harness itself fails on a deliberately un-guarded module; restores
  `sys.modules` in every case including mid-sweep failures.
- `tests/webibex/test_all_views_smoke.py` — all-views crash smoke, default
  suite. Recursive URL-resolver walk excluding third-party `include()`
  subtrees (admin/allauth/filer), per-view `OVERRIDES` for
  method/kwargs/POST-body/auth. Authenticates, asserts only "did not crash"
  (200/302/400/403/404 OK; 500 or an escaping exception is the only
  failure) — deliberately does not re-assert auth-gate semantics, already
  owned by `tests/core/test_views_auth_required.py`'s 46 tests (untouched,
  still 46/46 green). Completeness meta-test asserts exercised ∪ allowlisted
  == discovered exactly, so both a newly-added unrouted view and a stale
  allowlist entry fail loudly. **New latent bug found and pinned** (not
  fixed — no production code touched): `saved_animal_selection_view`'s GET
  branch references an unbound local `oid` → `NameError`, reachable by any
  authenticated GET today; the override dispatches via POST (the working
  path), bug documented inline with a `PIN:`-style comment matching the
  file's existing convention.
- `scripts/verify_prod_parity.py` + `tests/scripts/test_verify_prod_parity_live.py`
  — slow, opt-in `prod_parity`-marked gate mirroring
  `scripts/db_restore_drill.py`'s structure/config-object convention
  (`ParityConfig`, frozen dataclass, every function ≤4 params). Creates an
  isolated venv pinned to `runtime.txt`'s Python version, installs
  `requirements.txt` only (asserted on full argv), copies the repo tree into
  the scratch dir so it never touches the real `db.sqlite3`
  (`settings.py` hardcodes that path, no env override, and this CR is
  test/tooling-only by design), runs `manage.py check` +
  `migrate --run-syncdb`, starts `runserver`, `GET http://127.0.0.1:<port>/`
  (never `localhost` — `ALLOWED_HOSTS` lists `"localhost:8000"`, which never
  matches since Django strips the port before comparing), asserts 200 + no
  traceback in the server log. Child subprocess env is an explicit allowlist
  of the same test values root `conftest.py` already uses, never
  `{**os.environ, **overrides}` — proven not to leak ambient/decoy
  credentials. Refuses to run under `ENVIRONMENT=production` before any side
  effect. Scratch venv lives outside the repo, never under version control.
  Skips cleanly (not fails) when the pinned interpreter or PyPI egress isn't
  available — true in the dev sandbox this was built in (Python 3.13.5 vs.
  `runtime.txt`'s 3.12.5, no guaranteed network); item 2's manual run stands
  as the evidence the underlying mechanism works, this script makes it
  repeatable.
- `tests/conftest.py`: new shared `poison_modules` fixture (used by both the
  b2_utils regression test and the dependency-parity sweep).
- `pytest.ini`: new `prod_parity` marker, doc-comment style matching the
  existing `live_pg_restore`/`live_b2` markers.

**Found and worked around during implementation** (unrelated to the
dev-only-name poisoning itself): reimporting `core.admin`/
`simple_landmarks.admin` during the R2 sweep collided with Django's
`AlreadyRegistered` — fixed with a surgical admin-registry pop/restore keyed
to the exact models each admin module bare-registers, documented inline.

**Tests**: 676 passed, 5 skipped, 1 xfailed, 0 failed (62 new), 92% coverage
— independently re-run and confirmed three times total (right after the
executor's own report, twice more on direct request), identical counts every
time. `tests/core/test_views_auth_required.py` 46/46 and
`tests/core/test_views_smoke.py` 11/11 unaffected. `pytest -m prod_parity`
selects 1, skips cleanly in-sandbox per the constraint above. pyright: only
pre-existing baseline gaps remain; 2 new stub-less-idiom suppressions
justified inline. ruff: clean as of the post-close validation fix below (see
item 5) — was not actually clean when this CR was first implemented, a
session-close validation pass caught and fixed it.

**R8 constraint (no production code changes) verified**: `git diff --stat`
for commit `717bf8b` in isolation touches only `tests/**`, `pytest.ini`, and
`scripts/verify_prod_parity.py` — zero files under `core/`, `webibex/`,
`simple_landmarks/`, or `db_management/`.

## 4. Two ops TODOs logged (`1094b28`, docs-only)

- Postgres collation-version mismatch on the live Railway DB. Checked size
  via Railway's dashboard stats panel: 11.2 MB total, 1.5 MB indexes —
  `REINDEX DATABASE` is a sub-second operation at this size, no
  maintenance-window urgency. Recommended:
  `REINDEX DATABASE CONCURRENTLY railway;` then
  `ALTER DATABASE railway REFRESH COLLATION VERSION;`. **Gated on giving
  professor Alice notice first** (courtesy/awareness, not because of
  expected disruption) — not yet run.
- No B2-vs-DB reconciliation/orphan-detection tool exists. Documented as a
  future TODO (not implemented this session), tied to the existing
  `process_horn_chip` orphaned-file bug (`docs/security-remediation-plan.md`,
  found 2026-07-30) as the known drift source.

## 5. Post-close validation fix: `ruff` UP037 regression in the hotfix itself (uncommitted)

A session-close validation pass (independent Sonnet agent, tasked to check
this CR doc + session notes against live repo state before ingesting them
into memory) found `ruff check .` was **not** actually clean at the time —
`UP037 [*] Remove quotes from type annotation` on
`core/b2_utils.py:76`'s `objects: list["ObjectIdentifierTypeDef"] = [...]`.
Confirmed genuine (not pre-existing): the file was ruff-clean before the
`f3abc9f` hotfix; the quoting the hotfix introduced trips `UP037`, and
nothing in `ruff.toml` excludes it.

Root cause of the false claim: `f3abc9f` quoted *both* stub-referencing
annotations uniformly, but they aren't equivalent — `S3ServiceResource` is a
function return-type annotation (evaluated eagerly at `def` time regardless
of `TYPE_CHECKING`, so it genuinely needs the quotes) while
`ObjectIdentifierTypeDef` is only ever used as a *local variable* annotation
inside a function body, which CPython never evaluates at runtime (confirmed
empirically: `def f(): x: Undefined = 1` does not raise). The redundant
quoting on the local-variable case is exactly what `ruff`'s `UP037` flags.

**Fix**: unquoted `ObjectIdentifierTypeDef` at `core/b2_utils.py:76` (kept
`S3ServiceResource`'s quoting, which is load-bearing). Added an inline
comment at the `TYPE_CHECKING` block explaining the asymmetry so a future
reader doesn't "fix" it back to matching quoting. Corrected
`tests/core/test_b2_utils_type_checking_guard.py`'s T07 source-level
assertion, which had baked in the same wrong assumption ("every reference ...
must be a quoted forward-ref") — it now asserts the two names' *different*
correct shapes explicitly, with the runtime-evaluation reasoning inline.
Re-verified after the fix: `ruff check .` clean, the `sys.modules`-poisoning
regression check for the hotfix still passes, full suite back to 676
passed/5 skipped/1 xfailed/0 failed, pyright still only the same
pre-existing `core/b2_utils.py` error (django-environ's missing `default=`
stub, now at line 30, shifted again by the added comment).

**Not yet committed** — this fix touches `core/b2_utils.py` (production
code) and `tests/core/test_b2_utils_type_checking_guard.py`, outside this
CR's own R8 scope (which was specifically about the test-suite work in items
3-4, not the hotfix in item 2). Belongs as a small follow-up commit on top of
`f3abc9f`/`717bf8b`, not folded into either.

**Follow-up action**: `main` is 2 commits ahead of `origin/main`
(`1094b28`, `717bf8b`, on top of the already-pushed `80d7b42`/`f3abc9f`),
not yet pushed — sandbox has no `ssh` transport, user pushes from host.
Item 5's fix above is uncommitted working-tree state, not yet part of any
commit. Draft email to professor Alice covering all of the above prepared at
`tmp/email-draft-alice-20260815.md` (not committed, scratch artifact), not
yet confirmed sent.
