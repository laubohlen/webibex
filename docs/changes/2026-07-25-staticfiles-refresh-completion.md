# CR: complete the staticfiles/admin+filer refresh (planned in a prior session)

**What changed:**
- Modified: `staticfiles/admin/*` (21 files) — Django 5.2.16's own admin CSS/JS/img, regenerated via `collectstatic --clear`.
- Deleted: `staticfiles/admin/js/collapse.js` — orphaned, no longer shipped by the pinned Django version.
- Added: `staticfiles/admin/css/unusable_password_field.css`, `staticfiles/admin/js/unusable_password_field.js` — genuine new Django 5.2.16 admin files (confirmed via direct `.venv` package inspection — referenced by `django/contrib/admin/templates/admin/auth/user/{add_form,change_password}.html`), not leftovers.
- Modified: `staticfiles/filer/*` (4 files) — django-filer 3.3.0's own CSS/JS.
- Added: `core/tests/test_static_assets_collectstatic.py` (byte-exact RED→GREEN oracle + meta-oracle layer proving the comparator isn't vacuous), `core/tests/test_admin_filer_smoke.py` (routing/middleware smoke tests, explicitly documented as non-staleness guards).
- Status: committed as `54c35d6`.

**Follow-up action:** none — self-contained, already collected and committed.

**Do NOT:**
- Re-run `manage.py collectstatic` broadly without scoping the diff first — a prior session's attempt at this touched unrelated `staticfiles/admin|filer` drift while trying to fix a CSS bug elsewhere; keep static-asset refreshes as their own dedicated commit.
- Assume the admin dashboard or filer folder page needs separate testing beyond what was done here — `core/admin.py` registers `Image`/`Folder` directly on the Django admin site (`admin.site.register(Image, CustomImageAdmin)`, `admin.site.register(Folder, CustomFolderAdmin)`), so `/webibex/filer/folder/` **is** the admin's filer interface, not a separate standalone widget page. Testing upload there already exercises both admin and filer static assets together.

**Trigger:** any future Django or django-filer version bump — re-run `collectstatic --clear`, review the diff, verify via `test_static_assets_collectstatic.py`'s byte-exact oracle before committing.

**Why:** `staticfiles/` is a committed build artifact in this repo (not gitignored, since Railway's Nixpacks builder doesn't run `collectstatic` at deploy time) — it silently drifted out of sync with the pinned Django/django-filer versions after `480607b` bumped the requirement without re-collecting. Confirmed via `.venv` package inspection, not guessed.

**Verify:** `.venv/bin/python -m pytest core/tests/test_static_assets_collectstatic.py core/tests/test_admin_filer_smoke.py -q` — all green. Manual visual check: `/webibex/` (admin dashboard, light + dark mode) and `/webibex/filer/folder/` (upload via the widget) both confirmed working this session.

**Rollback:** `git revert 54c35d6` restores the stale `staticfiles/` tree and removes the two new test files.
