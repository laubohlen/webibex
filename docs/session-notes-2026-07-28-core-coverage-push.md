# Session Notes — 2026-07-28 — Pre-Refactor Coverage Push

## Context

User asked "what about the pytest coverage? before any meaningful refactor we should have a good test coverage." No specific refactor was named — general coverage push, prioritized by the user's own proposal: "start maybe from the less dependant modules? this way when we'll refactor these modules we'll both go ahead quickly without risk too much."

## Baseline (start of session)

`uv run pytest --cov-report=term-missing -q` — 184 passed, 1 skipped, 1 xfailed, 57% overall.

| File | Stmts | Miss | Cover |
|---|---|---|---|
| `core/views.py` | 388 | 297 | 23% |
| `core/signals.py` | 180 | 98 | 46% |
| `webibex/urls.py` | 16 | 8 | 50% |
| `core/utils.py` | 275 | 81 | 71% |
| `core/admin.py` | 65 | 18 | 72% |
| `simple_landmarks/views.py` | 1 | 1 | 0% |

These 6 files are exactly the set `ruff.toml` defers from its full ruleset (per-file-ignores) pending 100% measured coverage — see the "ruff-baseline deferred files" TODO in `docs/security-remediation-plan.md`. `manage.py`, `webibex/asgi.py`/`wsgi.py`, `db_management/*`, `scripts/run_local_e2e_server.py` are also in that deferred list but are outside `pytest.ini`'s `--cov` scope entirely (unmeasured, not just low).

## Dependency trace (informed the file order)

Grepped internal imports: `core/admin.py`, `core/signals.py`, `core/utils.py` each import only `.models` (+ `b2_utils` for utils) — no dependency on each other or on `views.py`. `core/views.py` imports `core/utils.py`. `webibex/urls.py` imports `core/views.py` (`from core.views import *`). `simple_landmarks/views.py` has zero internal deps and is 1 statement.

Order chosen: `simple_landmarks/views.py` (dead, excluded) → `core/admin.py` → `core/signals.py` → `core/utils.py` → `core/views.py` → `webibex/urls.py`.

## `simple_landmarks/views.py` — excluded, flagged for deletion

Investigated on request before deciding its fate. `git log --follow -- simple_landmarks/views.py` shows exactly one commit (`83f73dc "started landmarking standalone app"`), never modified since — untouched `django-admin startapp` scaffold. Repo-wide grep: zero imports of `simple_landmarks.views` anywhere. No `urls.py` in that app. `simple_landmarks` itself is a real, used app (models/admin actively imported elsewhere) — only `views.py` is dead.

Decision: exclude from the coverage CR, track for a separate deletion CR. Logged in `docs/security-remediation-plan.md`.

## `core/admin.py` — 72% → 100%, DONE, committed

Two targets: `LocationAdmin.ibeximage_name` (both `hasattr` branches) and `CustomFolderAdmin.tag_left`/`tag_right`/`tag_other` (folded into one test parametrized over `(method, side, word) × file-count(0,1,3)` per user's decision — the critical assertion is that every file in a multi-file queryset gets `.side` persisted, not just the first).

Test invocation approach (user-confirmed): `RequestFactory` + `django.contrib.messages.storage.fallback.FallbackStorage`, not a full admin changelist POST.

No production bug found in `core/admin.py`. `hasattr(obj, "ibeximage")` on line 45 is correct — `IbexImage.location` is a `OneToOneField`, Django's reverse descriptor raises `RelatedObjectDoesNotExist` (subclasses `AttributeError`) when absent, exactly what `hasattr` is designed to catch.

Interesting fixture gotcha found: `ibex_image_factory(name=...)` doesn't stick — `core/signals.py:127`'s `process_uploaded_image` post_save signal deterministically renames uploaded files based on animal/exif/season, overriding the factory's `name=` kwarg. Test adjusted to assert against the actual persisted name.

New fixture: `location_factory` in `tests/conftest.py` (`Location` model has all fields nullable).

pyright: 3 new findings, all accepted as tracked `django-stubs`/testing-pattern debt (`Location.objects` unknown-attribute gap matching 4 pre-existing instances; `WSGIRequest.session`/`._messages` unknown-attribute, standard Django testing pattern stubs don't model).

Result: 195 passed / 1 skipped / 1 xfailed, 0 regressions. Committed as 2 commits (`d036be6` docs, `4bdba31` tests).

## `core/signals.py` — 46% → 98%, DONE, committed (`7e0363f`)

9 functions/receivers needed coverage: `user_signed_up_callback`, `get_decimal_from_dms`, `extract_gps_coords`, `process_uploaded_image` (EXIF/GPS/folder-side branches), `initialise_landmark_items`, `delete_landmark_items`, `delete_associated_location`, `delete_ibexchip_file`, `create_folder_for_animal_on_change` (full branch matrix). `create_user_folders` and `check_animal_id_change` were already incidentally covered — confirmed, not re-tested.

`IbexImage.exif` confirmed to be `filer.models.abstract.BaseImage.exif`, a `cached_property` (non-data descriptor) — direct instance assignment (`image.exif = {...}` before `.save()`) cleanly overrides it, no `unittest.mock.patch` needed. `ibex_image_factory` extended with an optional `exif: dict | None = None` kwarg, byte-identical path preserved when omitted (verified via an immediate full-suite run right after the fixture edit, before any new test existed).

New fixtures: `landmark_factory`, `ibex_chip_factory` in `tests/conftest.py`.

**4 findings surfaced, all documented via tests (not fixed), all logged in `docs/security-remediation-plan.md` with full detail:**
1. `create_folder_for_animal_on_change` (line ~301-305): `UnboundLocalError` when `instance.side` is outside `{"L","R","O"}` — `target_folder` referenced before assignment. Test: `pytest.raises(UnboundLocalError)`.
2. `get_decimal_from_dms` (lines 63-89): uncaught `TypeError` (not `None`) on malformed-but-indexable DMS input — the arithmetic consuming `to_float()`'s results sits outside the `try/except` guarding the `to_float` calls. Swallowed downstream by `extract_gps_coords`'s own outer `except`, so not currently crash-reachable from production, but the pure function's own contract is broken. Tests: `pytest.raises(TypeError)`.
3. `signals.py:192-193` — dead/unreachable `else` branch (`dt_object` from `strptime` is always a `datetime` or already raised).
4. `signals.py:271-274` — dead/latent `except User.DoesNotExist` (a `None` owner raises `AttributeError` on `.username` one line earlier, never reaches this handler).

Findings 3+4 cap `core/signals.py`'s achievable coverage at 98% (confirmed exactly: 180 stmts, 3 missing — lines 193, 272, 274). Same situation as `core/models.py` (98%) and `custom_template_tags.py` (94%), which already have a documented case-by-case `ruff.toml` exception.

One mid-implementation deviation (test-only, not a `core/signals.py` bug): the `delete_ibexchip_file` test's `FieldFile.delete` monkeypatch initially crashed `chip.delete()` because a non-delegating stub broke `easy_thumbnails`'/filer's internal double-delete guard (which relies on the real `delete()` clearing `self.name`). Fixed by making the spy delegate to the original `delete()` while still recording the call.

pyright: 13 findings, all the same already-accepted `django-stubs` `.objects` gap (no new category).

Result: 224 passed / 1 skipped / 1 xfailed, 0 regressions. `git diff -- core/signals.py` confirmed empty (zero production changes).

**Commit signing hiccup, resolved within the same session**: `commit.gpgsign=true` configured with `gpg.ssh.program=/Users/trincuz/.ssh/git-ssh-sign-secretive.sh`, a macOS host path absent in this Linux sandbox. 3 earlier commits this session went through silently unsigned (no `gpgsig` header, confirmed via `git cat-file`); this file's first commit attempt hard-failed on the same config. Rather than bypass signing, the commit message was written to `tmp/commit-msg-2026-07-28-core-signals-coverage.txt` for the user to commit manually. The user committed it themselves shortly after (landed as `7e0363f`, 20:44:25, WITH a valid `gpgsig` SSH signature block — confirmed via `git cat-file -p`), using a copy of the message filed under `tmp/commit-msg-20260728/` (this project's dated-subdirectory convention for commit-msg drafts, per the pre-existing `tmp/commit-msg-20260723/` precedent). Exact mechanism that let the manual commit succeed where the automated one failed is not established — possibly ssh-agent/keychain state differs between the assistant's shell and the user's interactive shell. Docs findings for this file were committed separately (`9fc7fa1`) before the signing failure surfaced.

## Mutation testing — tracked as TODO, sequenced as a gate before refactor

No `mutmut`/`cosmic-ray` installed. Only precedent in repo: a one-off manual "empirical mutation probes" technique (Fable5 adversarial trace) used in `docs/changes/2026-07-26-auth-hardening-test-coverage-gaps.md`, not repeatable infra.

Initial framing ("revisit after the coverage initiative") was corrected same-day by the user: mutation testing is a hard gate before the refactor starts, not an indefinite someday-item. Final sequencing logged in `docs/security-remediation-plan.md`: coverage initiative (in progress) → install `mutmut` → run against every touched file → triage survivors → then the refactor.

## Remaining queue

`core/utils.py` (71%) next, then `core/views.py` (23%, 388 stmts — the biggest/riskiest, depends on `utils.py`), then `webibex/urls.py` (50%, depends on `views.py`).
