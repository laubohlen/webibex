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
| pip | 26.1.2 → 26.3 | committed (GHSA-qwm4-qh6w-59xr / CVE-2026-13346, fixed in 26.2.0) |
| django-polymorphic | 3.1.0 → 4.11.7 | committed (`bb7635f`); prerequisite for setuptools, see "setuptools decision" |
| setuptools | 78.1.1 → 84.2.1 | committed (`7a0ca88`); fixes GHSA-h35f-9h28-mq5c and GHSA-xmj3-9gf7-h52w |
| source-map-js (node/, transitive of tailwindcss) | 1.2.1 → 1.2.4 | committed (`3366ce0`); GHSA-68fv-2mgg-jv7q (HIGH) |
| postcss-selector-parser (node/, transitive) | 6.1.4 → 6.1.6 | committed (`3366ce0`); GHSA-rj75-hqrm-r3gf (MODERATE). `pnpm audit` now clean; Tailwind output byte-identical before/after |
| django-filer | 3.3.0 (unchanged) | **deferred ~mid-December**, parked on branch `wip/django-filer-3.6.0` (`751ec2e`), see "django-filer decision" |

Re-scan of the committed `requirements.txt` (OSV, after the certifi/Django/pillow commits, before sqlparse): certifi, Django and pillow showed 0 advisories. Still open then, counted with PYSEC duplicates: django-filer 1, geopy 2, pip 2, setuptools 3, sqlparse 10 (closed since), urllib3 14.

**Still open:** urllib3 (6 HIGH, blocked by botocore cap), django-filer (MODERATE, deferred). `requirements-dev.txt` was scanned later the same day, see "requirements-dev.txt scan".

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

**Parked (2026-11-20, deferred ~mid-December by choice).** The 3.6.0 work is on branch `wip/django-filer-3.6.0` (`751ec2e`): `py-svg-hush==0.3.0` pin plus the `staticfiles/filer/**` refresh. It does **not** contain the `django-filer==3.6.0` pin itself (one line in `requirements.txt`) and does **not** contain `staticfiles/filer/js/dist/` (3 files: `filer-base.bundle.js`, `admin-file-widget.bundle.js`, `LICENSES.txt`) because `.gitignore:39` ignores `dist/`. Railway does not run `collectstatic`, so on resume add an exception (`!staticfiles/filer/js/dist/` or `git add -f`) and re-run `collectstatic --clear`. I did not confirm that the 3.6.0 templates load those bundles. `main` is on 3.3.0 and `.venv` matches it. A stray `dist/` left in the working tree makes the two `filer` oracle tests in `tests/core/test_static_assets_collectstatic.py` fail.

**py-svg-hush 0.3.0 (vetted, not yet adopted).** Required by django-filer 3.6.0. `cp312` `manylinux_2_17_x86_64` wheel exists (prod is Python 3.12.5), so no source build; not yanked; no runtime dependencies; OSV lists no advisories. Four releases (0.1.0 to 0.3.0), 0.3.0 published 2026-02-13, single maintainer, native Rust extension that sanitizes untrusted SVG. Repository `github.com/jams2/py-svg-hush`; source and maintainer not inspected.

**To do when picked up** (one commit each, in this order): (1) re-sync or delete the `templates/admin/filer/` overrides and click through the filer admin (upload, folder and thumbnail views, image change form) — deleting them is the smallest change but alters the admin UI; (2) bump `django-filer==3.6.0` and add `py-svg-hush==0.3.0`; (3) `collectstatic --clear` refresh of `staticfiles/filer/`, update the test docstring versions. Run the suite plus the `webibex.wsgi` prod-parity check from `CLAUDE.md`.

## setuptools decision: done after django-polymorphic bump
Initially deferred (see "Why deferred" below); resolved the same day by bumping `django-polymorphic` first. django-polymorphic 4.11.7 supports Django 4.2-6.1, Python >=3.10, has no `pkg_resources` use, and the app has no first-party polymorphic code (used only through django-filer, which caps it at `<5.0` from 3.6.0; 3.3.0 has no cap). Then setuptools 84.2.1 (not yanked, no OSV advisories). Suite 706 passed, 5 skipped, 1 xfailed after each bump; `makemigrations --check` clean. The only remaining `pkg_resources` importers in `.venv` are `_pytest/monkeypatch.py` and `werkzeug/testapp.py` (dev/test only). The `webibex.wsgi` prod-parity check was not run (needs host).

**Advisories** (OSV, on 78.1.1): GHSA-h35f-9h28-mq5c (MODERATE; `MANIFEST.in` exclusion bypass in sdist via NFC/NFD collision on macOS APFS/HFS+; fixed 83.0.0) and GHSA-xmj3-9gf7-h52w (LOW; Windows path escape via `namespace_packages.txt` in legacy wheel install; fixed 84.1.0). Both are platform-specific (macOS/Windows); prod is Linux (Railway) and does not build sdists. Not verified against the deployed environment.

**Why deferred.** The 83.0.0 and 84.1.0 wheels contain no `pkg_resources` (checked by listing the wheels). `django-polymorphic` 3.1.0 does `import pkg_resources` unconditionally (`polymorphic/__init__.py:9`), so `import polymorphic` would fail on prod; the test venv would show the same failure. Latest setuptools is 84.2.1.

**pip** 26.1.2 → 26.3 was a drop-in bump (suite 706 passed, 5 skipped, 1 xfailed).

## Prod-parity check (2026-11-20)
`scripts/verify_prod_parity.py` run by the user on the host after the pip, django-polymorphic 4.11.7 and setuptools 84.2.1 bumps (`main` at that point, django-filer 3.3.0): PASS ("requirements.txt alone is sufficient"; `manage.py check`, migrate on sqlite, one `GET /` returning 200). Limits: the host was macOS with Python 3.12.16 (prod pins 3.12.5 on Linux), so it did not check Linux wheel availability for every pin; it exercises startup and `/` only, not the filer admin. Not re-run for the JS lockfile commit (not Python).

## requirements-dev.txt scan (2026-11-20, not shipped to Railway)
OSV query on the installed `.venv` versions of the 71 packages not in `requirements.txt` (the 10 direct dev pins plus transitive and stray packages). The 10 direct pins have 0 advisories. Four transitive/stray packages have advisories:

| Package | Version | Advisories | Pulled in by | Assessment |
|---|---|---|---|---|
| cryptography | 49.0.0 | GHSA-g6cj-pr64-35w5 HIGH (PKCS#7 EnvelopedData Bleichenbacher oracle), fixed 50.0.0 | `moto` (test mocks) | dev/test only; not reachable from app code (not reviewed in detail) |
| werkzeug | 3.1.8 | GHSA-g6x2-hccm-hh4m MODERATE (`safe_join` on Windows device names), fixed 3.1.9 | `moto` | dev/test only; Windows-specific |
| msgpack | 1.2.1 | GHSA-j586-36cw-2gc2 HIGH (memory leak in C extension), fixed 1.2.3 | `CacheControl`, which is in neither requirements file | env extra, noise |
| filelock | 3.32.0 | 12 advisories (2 HIGH, rest MODERATE/LOW; lock-file permission/symlink issues), fixed 4.2.1 | nothing requires it; in neither requirements file | env extra, noise |

Not fixed: all four are outside `requirements.txt` and not installed on prod. Checked afterwards: `moto` 4.2.14 requires `cryptography >=3.3.1` and `werkzeug !=2.2.0,!=2.2.1,>=0.5` (no upper caps), so the installed versions are just what resolved when `.venv` was built, and upgrading them is not blocked by moto (whether 50.0.0 / 3.1.9 exist and pass the suite was not checked: PyPI returned 403). `msgpack` comes from `CacheControl`, which comes from `pip_audit` (installed in `.venv`, in neither requirements file); the origin of `filelock` is unknown (nothing requires it). A text search of the installed `moto` found no `pkcs7`/`EnvelopedData`/`safe_join` use, and the project's own code, tests and scripts import none of the four packages; this is not a full reachability analysis. Dev tools' transitive versions are not pinned, so this scan describes this `.venv` only.

**Resolved in `.venv` (nothing committed, nothing pins these):** upgraded `cryptography` 49.0.0 → 50.0.2, `werkzeug` 3.1.8 → 3.1.9, `msgpack` 1.2.1 → 1.2.4, `filelock` 3.32.0 → 4.2.12 (all exist on PyPI, not yanked). `pip check` clean, `moto` still 4.2.14, suite 706 passed, 5 skipped, 1 xfailed, and a re-scan of the same 71 packages shows 0 OSV advisories. A fresh `pip install -r requirements-dev.txt` resolves whatever is newest at that time; pinning the dev tooling is a separate, undecided choice.

## Upgrade notes (checked 2026-11-20, PyPI metadata only)
- certifi 2026.10.18, pillow 13.0.0 (requires Python ≥3.11; prod is 3.12.5), Django 5.2.17 all exist and are not yanked.
- django-filer 3.3.0 and easy-thumbnails 2.10 do not cap pillow.
- Not tested: pillow 13 against image handling; setuptools 83 vs `pkg_resources` in django-polymorphic.
