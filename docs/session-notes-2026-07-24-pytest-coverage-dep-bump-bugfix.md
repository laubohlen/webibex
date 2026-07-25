# Session notes — 2026-07-24 — pytest coverage, dependency bump, first bug-fix round

Continuation session. Built on `9015b46` (TF1→TF2 migration + landmark CSS fix,
prior session). Three commits landed: `2bde17e`, `cfdeddd`, `6edb044`.

## pytest/pytest-django coverage — commit `2bde17e`

First-ever real test suite for the Django app (previously zero coverage —
`core/tests.py`/`simple_landmarks/tests.py` were unmodified `startapp`
boilerplate, `core/test_model.py` had no assertions and a hardcoded path to
another developer's machine — all three deleted).

Pipeline: `code-planner` → `code-analyst` (both run as Opus subagents since
`code-analyst` isn't a registered Agent subagent_type, only a Skill) →
`code-executioner`. Scope decisions made via `AskUserQuestion`: delete
`core/test_model.py` outright, `ENVIRONMENT=test` as the new env-gate value,
`pytest-cov` visibility-only (no `--cov-fail-under` gate, no CI exists), P0
pure-function tests use duck-typed stubs not real Django models.

Infra: root `conftest.py` sets required env vars via `os.environ.setdefault`
at module scope (webibex/settings.py and core/b2_utils.py read several
fail-secure — no default — at *module import time*; core/utils.py's
`endpoint_inference()` reads RunPod env vars as *function default-argument
values*, evaluated at module-def time, even earlier). `pytest.ini`
originally set `DJANGO_SETTINGS_MODULE` via the ini option — this caused a
real collection failure (`pytest-django`'s `pytest_load_initial_conftests`
hookimpl touches `django.conf.settings` eagerly when the ini option is set,
and can run before the root `conftest.py`'s own env-setdefault code has
executed, depending on plugin hook-registration order). Fixed by removing
the ini option and setting `DJANGO_SETTINGS_MODULE` as another
`os.environ.setdefault` call in `conftest.py` instead — pytest-django then
defers its settings check to `pytest_configure`, which always runs after
all conftests are loaded.

First code-executioner run was interrupted (user accidentally stopped the
background agent) — resumed with a fresh agent since a stopped agent can't
be resumed via SendMessage; re-briefed with the current repo state so it
picked up existing partial work rather than restarting.

104 tests landed, 1 intentional `xfail(strict=True)` pinning the AVIF/
`cv2.imread`-returns-`None` gap in `load_image()` (already documented in
`[[project_webibex_local_dev_gaps]]`). 5 other latent production bugs
found and deliberately pinned via `pytest.raises(...)` rather than fixed
(scope was coverage-only this round): `get_task_request_origin`
`UnboundLocalError` on no `HTTP_REFERER`; `scale_coordinate`/
`percentage_coordinate` `ZeroDivisionError` on zero-width/height;
`generate_animal_id_code` `IndexError` on a malformed existing `id_code`;
`Animal.__str__` `TypeError` when `id_code` is `None`; middleware
`except Folder.DoesNotExist` not catching what `get_object_or_404` actually
raises (`Http404`).

`pytest`/`pytest-django`/`pytest-cov` were initially added directly to
`requirements.txt` — user caught this ("what's requirements.txt for? what's
railway use?") since Railway's default Nixpacks builder only installs that
file (no `railway.toml`/`nixpacks.toml` in the repo) and there's no extras
syntax to mark packages dev-only. Split into a new `requirements-dev.txt`;
TODO added to `docs/security-remediation-plan.md`'s existing "Railway
deployment hardening" section to investigate `railway.toml`/`pyproject.toml`
dependency-group support later.

`/post-production` run (tier 3, Opus review + Fable 5 standalone second
opinion — user's established preference over `/judges`, which hardcodes
Opus with no model override). One real finding from Fable 5: T04 (the
`no_network` guard test) was tautological — its function-scoped mock was
created after collection-time imports had already happened, so
`call_count == 0` was trivially always true regardless of whether the
guard actually worked. Rewritten to assert the guard raises on an actual
call attempt instead. Fixed before commit.

## Dependency bump — commit `cfdeddd`

Popped `stash@{0}` from the prior session (Django 5.2.15→16, idna
3.10→15, pillow 12.2.0→12.3.0, pip 24.2→26.1.2, setuptools 78.1.1→83.0.0,
sqlparse 0.5.3→0.5.4 — OSV-verified, deliberately deferred until real test
coverage existed). `.venv` already had the bumped versions installed
(residue from the prior session's stash-verification `uv pip install`
run, before the changes were stashed rather than committed) — no
reinstall needed initially, `pip check` clean, suite green.

Fable 5's second-opinion review caught a real, reproduced CRITICAL issue
the Opus review had missed: `setuptools==83.0.0` no longer ships
`pkg_resources`; `django-polymorphic` (pulled in by `django-filer`, in
`INSTALLED_APPS`) imports it unguarded at module scope →
`ModuleNotFoundError` on every app boot. Reproduced directly (`import
polymorphic`, `import webibex.wsgi` under production-shaped env vars).
Root cause of why the test suite hadn't caught it: `core/tests`/`conftest.py`
had a test-only `pkg_resources` shim (added by `code-executioner` during
the coverage round to work around `.venv` drift) that was masking the
exact same gap the setuptools bump would trigger in production. `setuptools`
pulled from the bump (reverted to `78.1.1`, which already has the
CVE-2025-47273 fix — no urgency to go further right now); the shim removed
now that its own stated removal condition ("once the venv is re-synced to
the pinned setuptools version") was met. Egress needed reopening twice
during this verification (`bin/temp-egress`) — see
`[[project_devcontainer_network_sandbox]]` for the "already open"/stuck-
reload troubleshooting that came up along the way (unrelated
`devcontainer-guard` project, not webibex).

Final commit: 5 of 6 bumps (Django, idna, pillow, pip, sqlparse), verified
via full suite (104 passed) + direct `webibex.wsgi` import under
production-shaped settings (env vars matching Railway's actual deploy
shape) to catch what pytest-django's test-runner settings wouldn't
exercise.

## Bug-fix round 1 — commit `6edb044`

User explicitly scoped to "the easier ones" from the 6 bugs pinned during
the coverage round: B1 (`get_task_request_origin` `UnboundLocalError`), B3
(`generate_animal_id_code` `IndexError`), B4 (`Animal.__str__` `TypeError`),
and the middleware `Http404`/`Folder.DoesNotExist` mismatch. B2
(`ZeroDivisionError`) and B5 (AVIF/`load_image` gap) deferred — both need
more upfront judgment about what correct behavior should even be, not just
"stop it from crashing."

While planning, found a 5th bug with the identical root cause to B4:
`Region.__str__` has the same bare-field `None`-return pattern as
`Animal.__str__` — included in the same round for consistency.

Full planning-TDD pipeline again: `code-planner` → `code-analyst` (test
spec) → `code-executioner`. code-analyst's spec included a genuine gap the
original 3 pinned scenarios for B3 never covered — a batch with BOTH a
malformed and a valid existing `id_code` together, which the consolidated
fix (collapsing the two near-duplicate branches into one linear
filter-then-fallback path) makes newly meaningful to test. Also did a
blast-radius check: `get_task_request_origin` has zero callers anywhere in
the codebase besides its own tests (confirmed dead code, not fixed
uselessly — the fix is still correct and low-risk); `generate_animal_id_code`
has exactly one caller (`core/views.py:505`).

Fallback string decisions (via `AskUserQuestion`): bare bracket sentinel
(`"[No ID Code]"`/`"[No Name]"`) not prefixed like `Location.__str__`'s
style; B3's fix consolidated into one linear path rather than a minimal
patch duplicating the fallback in both branches; B1's two `print()` calls
left untouched (would be inconsistent scope creep — `core/utils.py` has
20+ `print()` call sites with zero logging infra anywhere in `core/`).

Both Opus and Fable 5 reviews came back clean (all INFO-severity, no
blockers) — but surfaced 3 more pre-existing, dormant latent bugs while
tracing the fixes by hand: `get_task_request_origin` confirmed genuinely
dead code; middleware still doesn't catch `Folder.MultipleObjectsReturned`
(a separate exception `get_object_or_404` can also raise, was equally
uncaught before this fix, not a regression); a family of `re.findall`-
related bugs in `generate_animal_id_code` (picks the *first* 3-digit run
not necessarily the sequence number if a prefix ever grows to 3+ digits;
no prefix-scoping on the max; `>999` rollover would exceed the model's
`id_code` `max_length=10`) — all pre-existing, dormant with current data
shapes (`PNGP24`-style 2-digit prefixes), unchanged by this diff. Logged
as tracked TODOs in `docs/security-remediation-plan.md`, plus a new TODO
for adding debug/observability logging at high-risk boundaries (I/O,
user-interaction flow, external API calls, local-dev-specific branches) —
user's explicit request at session close.

Final suite: 109 passed, 1 xfailed (up from 104 passed, 1 xfailed).

## Sandbox / tooling notes (not webibex-specific, but hit repeatedly this session)

- Bash tool has zero filesystem access outside the project directory
  (landlock-sandboxed) — confirmed by repeated `Permission denied` on
  `~/.claude/skills/*`, `~/.claude/feedback/log.jsonl`. The `Read`/`Write`
  tools operate through a different access path and worked fine for
  individual files, but `Write` can't safely append to a 280KB JSONL log
  without reading it whole first (which hits the tool's 256KB read cap) —
  the post-production JSONL feedback-log write was skipped both times this
  session for this reason, findings preserved to a scratchpad file instead.
- `code-analyst` is not a registered `Agent` `subagent_type` in this
  environment, only a `Skill` — its own `--model opus` routing logic
  (default) instructs spawning the full analysis as an isolated Agent
  subagent, which is what was done manually both times this round needed
  it, by re-issuing the skill's own Steps 0/2-6 instructions as the
  subagent's prompt.
- `docker info` confirms the Docker daemon itself is unreachable from
  inside this sandbox (no `/var/run/docker.sock`) — `sonar`, `trufflehog`,
  and any Docker-based `/post-production` check consistently SKIP for this
  reason, not a one-off.

## Remaining from this session's roadmap (not started)

B2 (`ZeroDivisionError` in `scale_coordinate`/`percentage_coordinate`) and
B5 (AVIF/`load_image` None-decode gap) — deferred, need domain judgment on
correct behavior first. Coverage-expansion targets identified but not
started: 25 of 28 views still untested, `core/signals.py` (44%),
`core/b2_utils.py` (36%), `parse_coordinates()`, `process_horn_chip()`.
Session paused here at the user's request (context budget check at 46%).
