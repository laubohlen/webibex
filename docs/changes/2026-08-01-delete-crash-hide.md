# CR: guard Tools-menu Delete crash, hide until real semantics decided

**What changed:**
- Modified: `core/views.py`, `multi_task_view` — added `elif task == "delete":
  return redirect("unidentified-images")` before the final `else:` block (same
  shape as the existing `tag_left`/`tag_right`/`tag_other` sibling branches). This
  means `task == "delete"` never reaches `utils.multi_task_url(...)` at all, so
  the tuple-unpack that used to crash on `None` is now unreachable for this input.
- Modified: `templates/core/unidentified_images.html` — removed
  `<option value="delete">Delete</option>` from the Tools `<select>`, so the crash
  path can't be selected from the UI either (belt-and-suspenders alongside the
  view guard).
- New: `tests/core/test_views_multi_task.py` (28 tests) — 7 P0 red→green tests
  (redirect behavior on single/multi selection, the "doesn't actually delete
  anything" invariant, both real URL entry points — `unidentified-images` via
  delegation and direct `multi-task/` — the nonexistent-id case proving the guard
  sits before `get_object_or_404`, and a login-required check), 2 template tests
  (no `delete` option present, exact remaining option set), 5 no-regression tests
  on the other Tools actions (tag branches, view, landmark, locate, pagination),
  7 counter-input pins on pre-existing malformed-POST crashes (explicitly out of
  scope — see Do NOT below), 1 verification-only check that the existing
  `test_multi_task_url_delete_branch_returns_none` stays green and unmodified.
- Zero changes to `core/utils.py` — `multi_task_url`'s own `delete`/`else` branches
  (which still return `None`) are byte-identical to before, confirmed via
  `git diff --stat core/utils.py` (empty).

**Follow-up action:** none required to land — already committed
(`3d5168d`, this session).

**Do NOT:**
- Treat this CR as having implemented real deletion. What "Delete" should actually
  do — hard delete vs. cascade to `IbexChip`/`Embedding`/the B2 file, soft-delete/
  archive, confirmation UX — is a separate, still-open design decision pending the
  professor's input, tracked in `docs/security-remediation-plan.md`. This CR only
  makes the crash unreachable; it does not decide or implement the real behavior.
- Assume `parse_coordinates`-style malformed-input hardening was also fixed here.
  A raw POST to `multi-task/` with missing/empty/non-numeric `selected-files` still
  crashes at `views.py:832-833` (unrelated pre-existing parsing that runs *before*
  the new branch) — this is pinned by 5 counter-input tests as pre-existing,
  explicitly out of scope. Fixing it would need a guard above every branch, which
  would violate "no other branch changes."
- Touch the single-image Delete button (`templates/core/image_read_new.html:13`,
  wired to the `delete-image` URL / `image_delete` view) — that one already works
  correctly and was never part of this bug. Only the multi-select Tools-menu
  "Delete" (dashboard bulk action) was broken.
- Re-add `<option value="delete">` to the template without also reverting the
  view guard — the two changes are a matched pair (belt-and-suspenders), not
  independently reversible.

**Trigger:** next time the professor answers the open design question on what
Delete should actually do (hard/cascade/soft-delete, confirmation UX) — that
becomes a new CR building on this one, likely also needing the B2
versioning/Object Lock decision from the same session's ransomware-hardening TODO
so real deletion and recoverability policy land together.

**Why:** confirmed reachable crash bug: dashboard → select row(s) → Tools →
Delete → `TypeError: cannot unpack non-iterable NoneType object` (500), traced to
`utils.multi_task_url`'s `delete` branch having no deletion logic and no `return`
statement. "Loud crash, no data touched" was already judged the safer failure mode
than a rushed implementation of the wrong kind of delete (prior session's own
framing, `docs/security-remediation-plan.md`) — this CR converts that from "crashes
loudly" to "silently redirects, still does nothing," closing the crash without
guessing at the real semantics.

**Verify:** `pytest tests/core/ -q` → 263 passed, 1 skipped, 1 xfailed (pre-fix
baseline + 28 new). `ruff check core/views.py tests/core/test_views_multi_task.py`
→ clean. `git diff --stat core/utils.py tests/core/test_utils_db.py` → empty.
Manual check against a live local dev server (no browser-automation tool available
in this sandbox, used direct HTTP requests instead): Tools dropdown HTML confirmed
to render exactly `["", "tag_left", "tag_right", "tag_other", "locate", "landmark",
"view"]`, no `delete`; POST `task=delete` on real unidentified images → 302 to
`/unidentified/`, not a 500; POST `task=view` on the same images still works
(200, correct template). `/post-production` ran clean (Tier 3 via a benign
`login` keyword false-positive from `client.force_login()` test calls; zero
findings from checks [1]-[7], ruff, insecure-defaults, request-adherence PASS on
all 4 requirements).

**Rollback:** revert commit `3d5168d`. Test-and-view-and-template-only, no
migration, no schema change, no external-service dependency — safe to revert
cleanly if needed.
