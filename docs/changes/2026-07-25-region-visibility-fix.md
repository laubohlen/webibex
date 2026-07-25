# CR: make Region visibility shared, not per-owner-filtered

**What changed:**
- Modified: `core/views.py:665` (`create_loaction`) — `region_qs = Region.objects.filter(owner=request.user)` → `region_qs = Region.objects.all()`.
- Modified: `core/utils.py:382` (`multi_task_url`'s `"locate"` branch) — same change.
- Modified: `core/tests/test_utils_db.py`, `core/tests/test_views_smoke.py` — 9 new tests: cross-owner region membership (discriminating, not just non-empty), an `owner=None` orphaned-region edge case, and a new regression guard (`test_region_edit_permission_unchanged_for_non_owner`) proving edit permission was NOT accidentally widened.
- Modified: `docs/security-remediation-plan.md` — the pre-existing "region dropdown empty" TODO marked resolved/decided.
- Status: committed as `f5f24cf`.

**Follow-up action:** none required for this fix itself. Two related, explicitly deferred follow-ups logged as separate TODOs in `docs/security-remediation-plan.md`: (1) whether region *coordinates* (not just names) should be reduced to name-only + on-demand detail in the location-picker UI, and (2) a pre-existing, unrelated IDOR on `location-id`/`oid` in `save_image_location`/`create_loaction` found by the adversarial review below.

**Do NOT:**
- Touch `save_region`'s edit-permission check (`core/views.py:534`), `delete_region` (`:593`), or `update_region` (`:604-606`) as part of any future work on this same area — those are deliberately unchanged; only *visibility* for selection/display widened, not *edit rights*. Any change there needs its own dedicated review.
- Assume this fix is a regression — confirmed via `git blame` that both `owner=` filters date to `46a66a8f`/`a6724250` (2025-02, original developer), over a year before this session.
- Conflate this decision with `region_overview`'s pre-existing (untouched, already-unfiltered) behavior — if a future direction reverses this to "private by design," `region_overview` (`core/views.py:612-614`) would *also* need to become owner-scoped for consistency; reverting just these two lines would leave a half-private state.

**Trigger:** if the professor/domain-owner confirms shared-by-design is *not* correct after all — re-scope to include `region_overview` (see above), not just these two call sites.

**Why:** `region_overview` (region list/management page) and `save_image_location` (the actual assignment-persisting step) already treated regions as shared/unfiltered — only these two dropdown-building call sites were outliers hiding data from users who didn't create the region. Confirmed via reading the actual code, not assumed.

**Verify:** `.venv/bin/python -m pytest core/tests/test_utils_db.py core/tests/test_views_smoke.py -q` — 149 passed overall (full suite), 9 of which are new for this CR. Security verification: an Opus `/post-production` review (tier 4, elevated for the authorization-boundary nature), a full 11-commit `/security-review`, and an Opus-authors-prompt/Fable5-executes adversarial pass (4 specific bypass candidates: region-assignment IDOR, spoofed UI affordances, queryset-membership-as-authorization, and independent re-verification of the new edit-permission test) — all clean, zero bypasses confirmed.

**Rollback:** `git revert f5f24cf` — restores the per-owner filters and removes the 9 new tests. Note this would NOT make regions fully private end-to-end (see "Do NOT" above re: `region_overview`).
