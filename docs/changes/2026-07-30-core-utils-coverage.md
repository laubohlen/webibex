# CR: raise test coverage on `core/utils.py` to 100% (pre-refactor push, file 3)

**What changed:**
- New: `tests/core/test_utils_process_horn_chip.py` (292 lines) — 13 scenarios covering
  `process_horn_chip`'s local-vs-cloud storage branch, pre-existing-chip handling (file
  and row present, file without row, row without file), the side="L"/"R"/"O" flip
  branch (spied via `mirror_coordinate`, wraps= the real implementation), and the cloud
  decode-failure paths (`None` download result, undecodable non-empty bytes, and a
  distinct empty-bytes `cv2.error` counter-input discovered while writing the test,
  not in the original spec).
- Modified: `tests/core/test_utils_pure.py` — 6 `parse_coordinates` scenarios (T01/T06
  happy-path parametrize including unvalidated negative coordinates; T02-T04 one
  parametrized `AssertionError` test covering zero-keys/two-keys/malformed-split;
  T05 parametrized `ValueError` test for non-integer values). Module docstring updated
  to note `django.test.RequestFactory` (no DB) is now used alongside the existing
  duck-typed-stub pure-function tests.
- Modified: `tests/core/test_utils_db.py` — 5 bonus `generate_animal_id_code`
  scenarios: the `>999` rollover-collision test (the most important addition — proves
  a real duplicate `id_code` via `Animal.objects.filter(id_code=result).exists()`,
  not just a string match) plus its just-under-rollover boundary control, and a
  parametrized test covering the rollover simple form, the first-3-digit-run
  misparse, and the no-prefix-scoping bug. `Animal` added to the existing
  `core.models` import.
- Modified: `docs/security-remediation-plan.md` — 3 new `## TODO —` sections
  (`generate_animal_id_code`'s three bugs; `process_horn_chip`'s dead code and
  missing guards; `parse_coordinates`'s bare-`assert` validation) plus one addendum
  sentence under the existing mutation-testing TODO's "Prime candidates" paragraph,
  flagging the deliberately-untested cloud-branch side="R" flip variant as a future
  mutation-testing target.
- Zero changes to `core/utils.py` itself — test-only CR by design, confirmed via
  `git diff -- core/utils.py` (empty). No new `conftest.py` fixtures needed — all 7
  fixtures used (`user_factory`, `ibex_image_factory`, `ibex_chip_factory`,
  `animal_factory`, `tiny_png_bytes`, `mock_b2`'s underlying pattern via direct
  `unittest.mock.patch("core.b2_utils....")`, `no_network`'s autouse guard) already
  existed.

**Follow-up action:** none required to land this CR conceptually, but it has **not
been committed** — per the launching agent's explicit instruction for this session,
work stops after verification and the user reviews before committing (same pattern as
the 2026-07-28 CR's manual-commit workaround for a git-signing issue in this sandbox).
No commit SHAs to report yet; the three touched/new files listed above are ready for
`git add` + commit once reviewed.

**Do NOT:**
- Treat this CR as having fixed any bugs. The `>999` rollover collision, no-prefix-
  scoping, and first-3-digit-run misparse in `generate_animal_id_code`; the dead
  `chip_url` assignments, the commented-out `b2_utils.delete_files` call (with its
  now-false log message), the missing `None`-guard vs. `embed_new_chip`'s guard, and
  the file/row desync in `process_horn_chip`; and the bare `assert`s in
  `parse_coordinates` are all real, all left in place, all tracked as separate
  bug-fix TODOs in `docs/security-remediation-plan.md`.
- Uncomment `b2_utils.delete_files` in `process_horn_chip` without first updating
  `test_process_horn_chip_cloud_replaces_existing_chip_no_b2_delete`'s
  `delete_mock.assert_not_called()` to the opposite assertion — that line is the pin.
- Change `@pytest.mark.django_db(transaction=True)` back to the plain marker on
  `test_process_horn_chip_local_row_without_file_raises_integrity_error` or
  `test_process_horn_chip_cloud_check_file_exists_false_raises_integrity_error` — the
  plain marker leaves the atomic block broken after the `IntegrityError`, and the
  follow-up `IbexChip.objects.filter(...).exists()` assertion raises
  `TransactionManagementError` instead of actually checking anything.
- Create an `IbexImage` outside the `override_settings(MEDIA_ROOT=str(tmp_path))`
  block — the upload lands in the real `media/` directory instead of the test
  sandbox.
- Call `user_factory()` more than once per test without passing a shared `owner=` to
  both `ibex_image_factory`/`ibex_chip_factory` — a second default `user_factory()`
  call collides on the unique `username="testuser"` default
  (`IntegrityError: UNIQUE constraint failed: core_user.username`).
- Add a cloud-branch side="R" flip test as a 4th `test_process_horn_chip_side_flip`
  parametrize case — this was a deliberate, locked scope decision (kept the CR
  bounded to the local-branch flip matrix); it's tracked as a mutation-testing
  candidate in `docs/security-remediation-plan.md` instead.
- Chase `chip_url` (lines 359, 479) as a coverage gap — both assignments execute and
  already count as covered; the finding is that their computed value is never used,
  a code-quality issue, not a test gap.

**Trigger:** continues for the next files in the same initiative —
`core/views.py` (23%) next (depends on `core/utils.py`, now fully tested), then
`webibex/urls.py` (50%, depends on `core/views.py`). Same procedure each time:
code-planner → code-analyst → code-executioner → `/post-production` → commit. Full
sequencing (including the mutation-testing gate before the eventual refactor) is in
`docs/security-remediation-plan.md`.

**Why:** continuing the leaf-first coverage initiative — `core/utils.py` sits between
`core/admin.py`/`core/signals.py` (done) and `core/views.py`/`webibex/urls.py` (next),
so its own test suite becomes the safety net for the next file up the import chain.
Three genuine bugs (`generate_animal_id_code`'s rollover/scoping/misparse trio) and
four findings in `process_horn_chip` (two dead-code, one missing guard, one
desync) surfaced along the way — documenting rather than fixing keeps this CR
test-only and low-risk, per the same discipline used for the two prior files in this
series.

**Verify:** `UV_CACHE_DIR=<writable-dir> uv run pytest --cov-report=term-missing -q`
→ **251 passed, 1 skipped, 1 xfailed** (from the 224-passed baseline — exactly +27:
9 in `test_utils_pure.py`, 5 in `test_utils_db.py`, 13 in the new
`test_utils_process_horn_chip.py`). `core/utils.py` → **100%** (was 71%; 275
statements, 0 missing — the prior 81-missing-statement gap at lines 37-45 and
346-483 is now fully closed, no dead-code ceiling as initially confirmed by
code-analyst). Project total → **74%** (was 67%). `ruff check
tests/core/test_utils_process_horn_chip.py tests/core/test_utils_pure.py
tests/core/test_utils_db.py` → all clean (fixed 4 `SIM117` nested-`with` findings and
1 `E501` line-length finding during this CR, before declaring done). `git diff --
core/utils.py` → empty, confirming test-only scope. `pyright` on the three
changed/new test files → `test_utils_pure.py` and `test_utils_process_horn_chip.py`:
0 findings each; `test_utils_db.py`: 21 findings (20 pre-existing + exactly 1 new,
same `.objects`-on-`type[Animal]` `reportAttributeAccessIssue` category as all 20
pre-existing ones — confirmed via `git stash`-diffing pyright output before/after
this CR's edit — no new finding category introduced).

**Rollback:** revert the relevant commit(s) once landed — all test-and-doc-only, no
schema/migration/external-service changes, safe to revert cleanly. (No commits exist
yet for this CR at time of writing; this section will be updated with the actual
commit hash(es) once the user commits.)
