# CR: gate 6 unauthenticated views behind @login_required

**What changed:**
- Modified: `core/views.py` — added `@login_required` to 6 views that had no
  auth gate at all (the `login_required` import already existed at the top
  of the file, used by 20+ other views): `save_landmarks_view` (was
  `views.py:101`), `results_over_view` (`:164`), `default_chip_compare_view`
  (`:174`), `project_chip_compare_view` (`:267`),
  `geographic_chip_compare_view` (`:350`), `rerun_view` (`:440`). Decorators
  added bottom-up by line number so earlier line numbers stayed valid across
  edits. Zero other lines touched — `git diff --stat core/views.py` shows
  exactly 6 one-line insertions.
- New: `tests/core/test_views_auth_required.py` (46 tests, T01-T25 per the
  planning-TDD test spec matrix) —
  - **P0 (anonymous access)**: parametrized GET and POST sweeps over all 5
    URL-routed gated views (never 200, always 302-to-login), a direct
    `RequestFactory` + `AnonymousUser` call for `geographic_chip_compare_view`
    (no URL route exists for it), and counter-inputs proving the decorator
    precedes every pre-existing bug/crash a bad anonymous request would
    otherwise hit (nonexistent `oid` would 404 post-`get_object_or_404`;
    missing/non-int `region` would `TypeError`/`ValueError`; GET on
    `save-landmarks` would `ValueError`; GET on `result-refined` would
    `TypeError`) — all now a clean 302 instead.
  - **R2 proof**: anonymous POST to `save_landmarks_view` asserted to never
    call `utils.process_horn_chip`, never touch the `no_network`-guarded
    `requests.post` boundary (the real RunPod HTTP call), never create an
    `IbexChip`, never mutate the `LandmarkItem` rows it targeted.
  - **P1 (authenticated behavior unchanged)**: happy paths for all 6 views,
    the `project_chip_compare_view` → `geographic_chip_compare_view`
    delegation branch proven via a real spy (`wraps=`, not a stub — actually
    calls the real function), owner/year-range gallery scoping preserved
    (counter-inputs: cross-owner chip excluded, year+5 boundary excluded),
    coordinate-scaling math for `save_landmarks_view` pinned to the correct
    scale direction (counter-input: unscaled values must NOT appear).
  - **Pinned pre-existing bugs, deliberately NOT fixed**: `rerun_view`
    authenticated still raises `TemplateDoesNotExist` (renders
    `core/result.html`, which doesn't exist as a file);
    `save_landmarks_view` authenticated GET still raises `ValueError` (the
    view's `else: pass` branch implicitly returns `None`). Both documented
    with a pin docstring pointing at this doc, matching the existing pin
    convention (`tests/core/test_views_multi_task.py:335-348`).
  - New file-local fixtures (not added to `tests/conftest.py` — narrow
    enough to this file's needs): `embedding_factory`, `landmark_setup`,
    `chip_with_embedding` (composite), `gate_scenario` (shared precondition
    for the T07/T08 URL sweeps).

**Follow-up action:** push to `origin` (`github.com/laubohlen/webibex.git`)
alongside the rest of the `docs/pre-deploy-checklist.md` release blockers —
this fix alone does not deploy itself; see that checklist's §0.

**Do NOT:**
- Treat this CR as closing the underlying IDOR class. `save_image_location`,
  `create_loaction`, and `save_landmarks_view`'s missing owner check on
  `image-id` still let any *authenticated* user act on another user's data —
  separate, already-tracked, still open (`docs/security-remediation-plan.md`).
  This fix only closes the *unauthenticated* population down to the
  already-trusted login-gated one.
- Assume the two pinned bugs (`rerun_view`'s `TemplateDoesNotExist`,
  `save_landmarks_view` GET's `ValueError`) were fixed here. Both are
  pre-existing, predate this session, and are explicitly out of scope —
  pinned by `test_rerun_view_authenticated_still_raises_template_does_not_exist`
  and `test_save_landmarks_view_authenticated_get_still_raises_value_error`.
- Treat the project-version-bump mentioned in `docs/pre-deploy-checklist.md`
  as part of this fix. Explicitly deferred by the user — this repo has no
  `VERSION` file or `pyproject.toml` yet (only `requirements.txt`), so there
  is nothing to bump; a `pyproject.toml` (which would carry
  `[project.version]`) is planned as separate future work.
- Commit this from inside the sandboxed devcontainer. Commit signing is
  broken here (host-only key, known issue) — left as uncommitted
  working-tree changes for the user to commit from the host.

**Trigger:** none pending on this fix itself. The IDOR follow-up above is
tracked separately in `docs/security-remediation-plan.md` and
`docs/pre-deploy-checklist.md`'s "Explicitly deferred" section.

**Why:** `docs/security-remediation-plan.md`'s 2026-08-14 "New, more severe
finding" — `save_landmarks_view` had **no** `@login_required` at all (unlike
the rest of the IDOR-class entries in that doc, which at least require
login), and additionally triggers a real, billed RunPod inference HTTP call
with no auth and no ownership check on the `image-id` it's given. The same
audit found the other 5 views missing the decorator too. No global auth
middleware covers the gap (`MIDDLEWARE` has no `LoginRequiredMiddleware`);
re-verified 2026-08-14 that no template links to any of the 6 without
already being login-gated itself, and `rerun_view` has no entry point at
all — confirming a missed-decorator gap, not an intentional public surface.

**Verify:** `pytest -q` from worktree root → 325 passed (279 pre-fix baseline
+ 46 new), 1 skipped, 1 xfailed — zero regressions. `ruff check core/views.py
tests/core/test_views_auth_required.py` → clean (`core/views.py` is fully
ruff-exempted per `ruff.toml`; the new test file needed several
line-length wraps to satisfy E501, not in the `tests/**` ignore list).
`pyright` → not clean project-wide (`pyrightconfig.json` was only added
2026-08-09 as a first-ever baseline, "matches this repo's de facto current
state" — never clean before this CR either), but this diff introduces zero
new errors: no pyright error references `login_required`, and the new test
file's errors are the same pre-existing `Model.objects`/`WSGIRequest.user`
stub-gap pattern (no django-stubs plugin) already present in `conftest.py`,
`core/signals.py`, and multiple existing test files
(`test_models.py`/`test_signals.py`/`test_middleware.py`/`test_utils_db.py`/
`test_views_smoke.py`). `manage.py check --deploy` → 7 warnings, all expected
under the ambient `ENVIRONMENT=test` used to run the check (the
`SESSION_COOKIE_SECURE`/`CSRF_COOKIE_SECURE`/`SECURE_SSL_REDIRECT`/HSTS
settings are gated to `ENVIRONMENT == "production"` by design, per the
2026-07-25 auth-hardening fix); informational only, not a regression.

Manual mutant-matrix (each decorator commented out one at a time, full new
test file re-run, decorator restored before the next) — every removal caught
by at least one test:

| View | Decorator removed → tests caught it |
|---|---|
| `save_landmarks_view` | 8 failed: `test_save_landmarks_view_anon_post_redirects_to_login_not_success`, `test_all_gated_views_anon_get_never_200[save-landmarks]`, `test_all_gated_views_anon_post_never_200[save-landmarks]`, `test_save_landmarks_view_anon_get_counter_input_redirects_not_valueerror`, `test_save_landmarks_view_anon_post_never_reaches_runpod_or_saves`, `test_save_landmarks_view_anon_post_with_next_id_index_never_delegates`, `test_save_landmarks_view_anon_get_contrast_redirects_no_exception`, `test_gated_views_are_login_required_wrapped[save_landmarks_view]` |
| `results_over_view` | 5 failed: `test_results_over_view_anon_redirects_to_login`, `test_all_gated_views_anon_get_never_200[results-overview]`, `test_all_gated_views_anon_post_never_200[results-overview]`, `test_login_redirect_exact_contract`, `test_gated_views_are_login_required_wrapped[results_over_view]` |
| `default_chip_compare_view` | 5 failed: `test_default_chip_compare_view_anon_redirects_to_login`, `test_default_chip_compare_view_anon_nonexistent_oid_still_redirects_not_404`, `test_all_gated_views_anon_get_never_200[result-default]`, `test_all_gated_views_anon_post_never_200[result-default]`, `test_gated_views_are_login_required_wrapped[default_chip_compare_view]` |
| `project_chip_compare_view` | 7 failed: `test_project_chip_compare_view_anon_post_toggle_false_redirects`, both `test_project_chip_compare_view_anon_post_bad_region_still_redirects` params, `test_project_chip_compare_view_anon_post_without_toggle_does_not_delegate`, `test_project_chip_compare_view_anon_post_toggle_capital_false_does_not_delegate`, `test_all_gated_views_anon_post_never_200[result-refined]`, `test_gated_views_are_login_required_wrapped[project_chip_compare_view]`. Notable: the GET-sweep `[result-refined]` case did NOT fail — the "not toggle=false" branch delegates into `geographic_chip_compare_view`, whose own (still-present) decorator masks the removal for that one path. Killed instead by the toggle=false-branch and delegation-mock tests, which exercise the code path that actually needs this view's own gate — confirms defense-in-depth between the two views, not a test gap. |
| `geographic_chip_compare_view` | 2 failed: `test_geographic_chip_compare_view_anon_direct_call_redirects`, `test_gated_views_are_login_required_wrapped[geographic_chip_compare_view]` |
| `rerun_view` | 4 failed: `test_rerun_view_anon_redirects_cleanly_not_template_does_not_exist`, `test_all_gated_views_anon_get_never_200[run-again]`, `test_all_gated_views_anon_post_never_200[run-again]`, `test_gated_views_are_login_required_wrapped[rerun_view]` |

**Rollback:** revert the 6 `@login_required` additions in `core/views.py` and
delete `tests/core/test_views_auth_required.py`. Test-and-view-only, no
migration, no schema change — safe to revert cleanly if needed (though doing
so re-opens the vulnerability this CR closes).
