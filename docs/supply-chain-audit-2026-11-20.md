# Supply-chain audit 2026-11-20 (dependency list — private remote only)

Follow-up to `supply-chain-audit-2026-10-12.md`. The findings tables below describe the state **at audit time** (before any dependency change). See "Status after fixes" for what was done the same day.

## Status after fixes (2026-11-20, same day)
Each bump was tested in `.venv` with the full suite (706 passed, 5 skipped, 1 xfailed) and committed on its own.

| Package | Was → now | Result |
|---|---|---|
| certifi | 2025.1.31 → 2026.10.18 | committed (`feaff06`) |
| Django | 5.2.16 → 5.2.17 | committed (`a720d5e`) |
| pillow | 12.3.0 → 13.0.0 | committed (`7ad63f8`) |
| sqlparse | 0.5.4 → 0.6.0 | committed (its own commit after `7ad63f8`) |
| geopy | 2.4.1 → 2.5.0 | committed (GHSA-mhvh-fq92-pfmr, ReDoS in `Point.from_string`; app only calls `distance()` on float tuples, `core/utils.py`) |
| django-filer | 3.3.0 (unchanged) | **deferred**, see "django-filer decision" |

Re-scan of the committed `requirements.txt` (OSV, after the certifi/Django/pillow commits, before sqlparse): certifi, Django and pillow showed 0 advisories. Still open then, counted with PYSEC duplicates: django-filer 1, geopy 2, pip 2, setuptools 3, sqlparse 10 (closed since), urllib3 14.

**Still open:** urllib3 (6 HIGH, blocked by botocore cap), django-filer (MODERATE, deferred), setuptools, pip (MODERATE, rated not reachable on 10-12), JS dev tooling `source-map-js` (HIGH) and `postcss-selector-parser` (MODERATE).

## Scope and method
- `/supply-chain` over `requirements.txt` (35 pins) and `node/` (pnpm). `requirements-dev.txt` not CVE-scanned.
- Python CVEs from the OSV API (no `pip-audit`/`uv audit` path for plain `requirements.txt`). Severities are GitHub-advisory labels, not CVSS.
- No reachability review this run; the 2026-10-12 review covers only Django, sqlparse, urllib3, geopy, setuptools, pip.

## New since 2026-10-12
| Package | Installed | Advisories | Fixed in |
|---|---|---|---|
| certifi | 2025.1.31 | CRITICAL GHSA-6q48-6c7w-9gw2 (emSign roots, compromised intermediate keys) | 2026.10.18 |
| pillow | 12.3.0 | OSV range match: 5 CRITICAL, 24 HIGH, 14 MODERATE (+ unrated PYSEC duplicates). Individual advisories not inspected. | 13.0.0 |
| django-filer | 3.3.0 | MODERATE GHSA-vf85-8ff2-3r9w (missing per-object authorization; see "django-filer decision" — the advisory title and body disagree) | 3.5.0 |

## Unchanged from 2026-10-12
urllib3 1.26.20 (6 HIGH, 1 MODERATE; 2.x only, blocked by botocore cap), sqlparse 0.5.4, Django 5.2.16 (fix 5.2.17), geopy 2.4.1, setuptools 78.1.1, pip 26.1.2. Nothing upgraded since.

## JS (`node/`, dev-only tooling)
- `pnpm audit`: source-map-js 1.2.1 HIGH (fix 1.2.2), postcss-selector-parser 6.1.4 MODERATE (fix 6.1.5). `braces` finding from 10-12 no longer reported.
- Lifecycle scan: `prepare` in postcss-selector-parser (normal); `postinstall` in `ljharb-monorepo-symlink-test` inside `resolve`'s test dir (likely fixture, not manually verified).
- Lockfile integrity inconclusive (registry 403); lock has 89 `integrity:` entries.

## Other checks
- Typosquatting (manual), dependency confusion, yanked: clean. `django-polymorphic` 3.1.0 last released 2021-11-18.
- No hashes in `requirements*.txt`; no CI config.
- socket.dev: npm 0 findings; PyPI flags certifi, pillow (CRITICAL), django, urllib3, sqlparse.

## django-filer decision: deferred
**Advisory.** GHSA-vf85-8ff2-3r9w, CVSS:3.1 `AV:N/AC:L/PR:L/UI:N/S:U/C:N/I:L/A:N` (4.3 by my calculation; GitHub rates MODERATE), CWE-285/639/862. The title says "`discard` and `edit_redirect` version transitions", but the body describes the "Delete files and/or folders" admin action skipping per-folder edit-permission checks. Affects `<= 3.4.4`, only with `FILER_ENABLE_PERMISSIONS = True`. Exploit needs a staff user who is read-only on a folder via `FolderPermission` (`can_edit = DENY`) but holds the global `delete_*` permission. Integrity only; no confidentiality claim. OSV lists 3.5.0, 3.5.1 and 3.6.0 as unaffected. django-filer's own changelog was not read (GitHub unreachable from the sandbox).

**Applies to this project.** `FILER_ENABLE_PERMISSIONS = True` (`webibex/settings.py:228`) and `CustomFolderAdmin` lists `delete_files_or_folders` (`core/admin.py:61`). An earlier same-day guess that it was a django-cms-versioning issue and likely not reachable was wrong.

**Why exposure is probably low (not confirmed).** Signup does not set `is_staff` (commented out, `core/signals.py:29`), so ordinary users cannot reach the filer admin. No first-party code creates `FolderPermission` rows. Superusers bypass the checks. The prod database was not inspected. To check, in a Django shell on prod: `FolderPermission.objects.count()` and `get_user_model().objects.filter(is_staff=True, is_superuser=False).count()`. If both are 0 (or no such staff user holds delete permission), nobody can exploit it; if not, upgrade soon.

**Why deferred.** Upgrading is not a drop-in change. Tried in `.venv` with 3.6.0:
- Install clean, `pip check` clean, `makemigrations --check`: no changes. New required dependency `py-svg-hush` 0.3.0 (no deps; cp312 manylinux wheel not checked) would need pinning in `requirements.txt`.
- `staticfiles/` is a committed build artifact. The byte-exact oracle tests (`tests/core/test_static_assets_collectstatic.py`) failed until `collectstatic --clear` was re-run, which changed only `staticfiles/filer/` (18 modified, 8 deleted, 3 added). Per `docs/changes/2026-07-25-staticfiles-refresh-completion.md` this must be its own commit. The test docstring still names Django 5.2.16 / django-filer 3.3.0.
- **Breaking:** 3.6.0 no longer ships `static/filer/js/libs/` (`class`, `dropzone`, `fileuploader`, `jquery-ui`, `jquery.cookie`, `mediator`). The project overrides `templates/admin/filer/base_site.html`, `file/change_form.html` and `image/change_form.html`, which still load them; they would 404. Tests do not catch this.
- The 28 overrides in `templates/admin/filer/` compared against the 3.3.0 wheel and the installed 3.6.0: 11 are identical to 3.6.0; 9 are unmodified 3.3.0 copies that 3.6.0 changed upstream (`actions`, `breadcrumbs`, `delete_selected_files_confirmation`, `file/change_form`, `folder/change_form`, `choose_copy_destination`, `choose_move_destination`, `image/change_form`, `clipboard`); 8 differ from 3.3.0 (`base_site`, `directory_listing`, `directory_table_list`, `directory_thumbnail_list`, `file_icon`, `detail_info`, `admin_file`, `admin_folder`). The differences look like leftovers from an older filer version rather than deliberate customizations: `base_site` is only reformatted; several files contain a stray `}` (`{% if max_filesize %}}data-max-filesize=...`, a pre-existing rendering bug); some use old-API calls (`file_icon size="160x160"`, `cms-icon-select`, `file.name`). Only commit `3c324dc add filer staticfiles` touches them; no recorded reason for the overrides.

**To do when picked up** (one commit each, in this order): (1) re-sync or delete the `templates/admin/filer/` overrides and click through the filer admin (upload, folder and thumbnail views, image change form) — deleting them is the smallest change but alters the admin UI; (2) bump `django-filer==3.6.0` and add `py-svg-hush==0.3.0`; (3) `collectstatic --clear` refresh of `staticfiles/filer/`, update the test docstring versions. Run the suite plus the `webibex.wsgi` prod-parity check from `CLAUDE.md`.

## Upgrade notes (checked 2026-11-20, PyPI metadata only)
- certifi 2026.10.18, pillow 13.0.0 (requires Python ≥3.11; prod is 3.12.5), Django 5.2.17 all exist and are not yanked.
- django-filer 3.3.0 and easy-thumbnails 2.10 do not cap pillow.
- Not tested: pillow 13 against image handling; setuptools 83 vs `pkg_resources` in django-polymorphic.
