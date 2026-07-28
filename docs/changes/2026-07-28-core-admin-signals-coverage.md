# CR: raise test coverage on `core/admin.py` and `core/signals.py` (pre-refactor push, files 1-2)

**What changed:**
- New: `tests/core/test_admin.py` — `LocationAdmin.ibeximage_name` (both branches),
  `CustomFolderAdmin.tag_left`/`tag_right`/`tag_other` (single test parametrized over
  `(method, side, word) × file-count(0,1,3)`, asserting every file in a multi-file
  queryset gets `.side` persisted, not just the first). `RequestFactory` +
  `FallbackStorage` invocation, no full admin changelist POST.
- New: `tests/core/test_signals.py` — 23 scenarios covering `user_signed_up_callback`,
  `get_decimal_from_dms`, `extract_gps_coords` (pure functions, full counter-input
  matrix), `process_uploaded_image`'s EXIF/GPS/folder-side branches, landmark-item
  lifecycle, `delete_ibexchip_file`, and the full `create_folder_for_animal_on_change`
  branch matrix (L/R/O + two filename-parts branches + the bug-triggering else).
- Modified: `tests/conftest.py` — new `location_factory`, `landmark_factory`,
  `ibex_chip_factory` fixtures; `ibex_image_factory` gets an optional
  `exif: dict | None = None` kwarg (byte-identical `.objects.create(**defaults)` path
  preserved when omitted — verified via an immediate full-suite run right after the
  edit, before any new test existed).
- Modified: `docs/security-remediation-plan.md` — 6 new TODO entries: dead-code
  flag for `simple_landmarks/views.py`, `UnboundLocalError` bug in
  `create_folder_for_animal_on_change`, uncaught `TypeError` bug in
  `get_decimal_from_dms`, two dead/unreachable branches in `core/signals.py`, and a
  mutation-testing TODO sequenced as a gate before the eventual refactor.
- Zero changes to `core/admin.py` or `core/signals.py` themselves — test-only CR by
  design; both files' bugs are documented via `pytest.raises`, not fixed.

**Follow-up action:** none required to land this CR — all of it is committed:
`d036be6` (dead-code doc), `4bdba31` (`core/admin.py` tests), `9fc7fa1`
(`core/signals.py` findings doc), `7e0363f` (`core/signals.py` tests). Note:
`7e0363f`'s first commit attempt hard-failed on a broken git-signing config
(`commit.gpgsign=true` pointing at a macOS-only script path absent in this Linux
sandbox) — the assistant wrote the commit message to a `tmp/` draft rather than
bypass signing, and the user committed it manually shortly after (landed WITH a
valid signature). See `docs/session-notes-2026-07-28-core-coverage-push.md` for the
full account; the signing-failure root cause itself is not resolved/understood, just
worked around this once.

`core/admin.py` (100%) and `core/signals.py` (98%) are now both case-by-case
candidates for `ruff.toml`'s per-file-ignores removal, matching the precedent set for
`core/models.py` (98%) and `custom_template_tags.py` (94%) in
`docs/changes/2026-07-27-ruff-coverage-gate-expansion.md` — not done as part of this
CR, tracked as its own follow-up.

**Do NOT:**
- Treat this CR as having fixed any bugs. `UnboundLocalError` in
  `create_folder_for_animal_on_change` and the `TypeError` in `get_decimal_from_dms`
  are both real, both left in place, both tracked as separate bug-fix TODOs in
  `docs/security-remediation-plan.md` — the new tests document current behavior via
  `pytest.raises`, they do not assert the "fixed" behavior.
- Chase `core/signals.py`'s remaining 2% (lines 193, 272, 274) with contrived tests —
  both are genuinely unreachable dead code (documented in the same TODO doc), not a
  test gap.
- Assume `simple_landmarks/views.py`'s 0% coverage is part of this initiative's scope
  — it's explicitly excluded, flagged for a separate deletion CR (confirmed dead
  `startapp` scaffold, zero call sites).
- Skip the full-existing-suite run immediately after any future `ibex_image_factory`
  edit — that fixture is used by every image-creating test in the suite; this CR's
  own step-4 regression gate is what caught (or would have caught) any breakage from
  the `exif` kwarg addition before it could hide inside a larger diff.

**Trigger:** continues for the next files in the same initiative — `core/utils.py`
(71%) next, then `core/views.py` (23%, depends on `utils.py`), then `webibex/urls.py`
(50%, depends on `views.py`). Same procedure each time: code-planner → code-analyst →
code-executioner → `/post-production` → commit. Full sequencing (including the
mutation-testing gate before the eventual refactor) is in
`docs/security-remediation-plan.md`.

**Why:** user asked for coverage before a planned refactor, then confirmed a
leaf-first dependency ordering so each file's new tests become a safety net for the
next file up the import chain. Two genuine bugs and two dead branches surfaced along
the way — documenting rather than fixing keeps this CR test-only and low-risk, per
the same discipline the `ruff-coverage-gate-expansion` CR used for its own two files.

**Verify:** `UV_CACHE_DIR=<writable-dir> uv run pytest --cov-report=term-missing -q`
→ 224 passed, 1 skipped, 1 xfailed (from a 184-passed baseline); `core/admin.py` 100%
(was 72%), `core/signals.py` 98% (was 46%), project total 67% (was 57%). `ruff check
tests/conftest.py tests/core/test_admin.py tests/core/test_signals.py` → all clean.
`git diff -- core/admin.py core/signals.py` → empty on both, confirming test-only
scope. `pyright` on the changed test files → 13 findings on `core/signals.py`'s test
work + 3 on `core/admin.py`'s, all the same pre-existing `django-stubs` `.objects`
gap already accepted as tracked debt (no new category introduced).

**Rollback:** `git revert 7e0363f 9fc7fa1 4bdba31 d036be6` (newest-first, or use
`git revert d036be6^..7e0363f` for the whole range in one step) — all 4 commits are
test-and-doc-only, no schema/migration/external-service changes, safe to revert
cleanly.
