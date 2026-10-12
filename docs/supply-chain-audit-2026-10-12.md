# Supply-chain audit 2026-10-12 (dependency list — private remote only)

## Scope and method
- `/supply-chain` pass over `requirements.txt` (35 pinned packages), `requirements-dev.txt` (not CVE-scanned), `node/package.json` + `node/pnpm-lock.yaml` (pnpm 11.27.0).
- Python CVEs from the OSV API (`api.osv.dev`), not `pip-audit`: `uvx pip-audit` failed because PyPI returned 403 from the sandbox.
- Severities are GitHub-advisory labels (HIGH → MAJOR, MODERATE → MINOR), not CVSS scores.
- Reachability review done by an Opus subagent (read-only). Its file:line claims were not re-verified in the main session.

## Audit findings (CRITICAL 0, MAJOR 5, MINOR 4, INFO 4)
| Package | Installed | Advisories | Fixed in |
|---|---|---|---|
| urllib3 | 1.26.20 | 6 HIGH + 1 MODERATE (GHSA-2xpw-w6gg-jr37, -38jv-5279-wg99, -8988-9cw3-xx77, -gm62-xv2j-4w53, -qccp-gfcp-xxvc, -vxq7-64xx-v4gw; -pq67-6m6q-mj2v moderate) | 2.5.0–2.8.0 (2.x only) |
| sqlparse | 0.5.4 | 3 HIGH (GHSA-f2ff-p2ww-7p4p, -prg7-hcfm-mfcr, -pwgv-4x5q-6m9f) + 2 MODERATE | 0.6.0 |
| Django | 5.2.16 | HIGH GHSA-wvqv-fj8w-qmhm, MODERATE GHSA-q238-5cxm-5c9h (both GeoDjango) | 5.2.17 |
| setuptools | 78.1.1 | MODERATE GHSA-h35f-9h28-mq5c | 83.0.0 |
| pip | 26.1.2 | MODERATE GHSA-qwm4-qh6w-59xr | 26.2.0 |
| geopy | 2.4.1 | MODERATE GHSA-mhvh-fq92-pfmr | 2.5.0 |
| braces (dev, JS) | 3.0.3 | HIGH GHSA-vfj7-8cjw-p6xm, via `clean-css-cli>chokidar` | >=3.0.4 |
| source-map-js (dev, JS) | 1.2.1 | HIGH GHSA-68fv-2mgg-jv7q, via `tailwindcss>postcss` | >=1.2.2 |
| postcss-selector-parser (dev, JS) | 6.1.4 | MODERATE GHSA-rj75-hqrm-r3gf | >=7.1.6 |

Other checks:
- socket.dev: npm 2 checked / 0 findings; PyPI 35 checked, flagged urllib3, sqlparse, django (cve) — consistent with OSV.
- Typosquatting (manual), dependency confusion (no extra index, no private packages): clean.
- `requirements.txt`: full tree pinned with `==`, no `--hash=` lines; no `pyproject.toml`/`uv.lock`.
- No CI config in repo (no `.github/workflows`, `.gitlab-ci.yml`, Jenkinsfile, Dockerfile).
- `node/.npmrc`: `ignore-scripts=true`.
- JS lockfile integrity check inconclusive: pnpm-guard policy verification got 403 from `registry.npmjs.org`. `git status node` clean.
- Not checked: yanked/abandoned (PyPI/npm metadata 403), lifecycle-script scan (`lifecycle-scripts-check.py` unreadable), `requirements-dev.txt` CVEs.

## Reachability review (Opus subagent) — all NOT-REACHABLE in production
- **Django/GeoDjango**: `django.contrib.gis` not in `INSTALLED_APPS` (`webibex/settings.py:76-90`; `debug_toolbar` appended at :107); lat/lon are `FloatField` (`core/models.py:30-31,67-70`); grep for gis/GEOS/GDAL/postgis in first-party code and in filer/polymorphic/easy_thumbnails/allauth/storages found nothing.
- **sqlparse**: no first-party import. Django 5.2.16 calls it in `base/operations.py:365-366` (`prepare_sql_script`; Postgres overrides at `postgresql/operations.py:193-194`), `sqlite3/introspection.py:189-237` and :281,:415,:435 (dev only), `test/runner.py:101` (`--debug-sql`). No `RunSQL` in repo migrations or in filer/allauth/easy_thumbnails/polymorphic.
- **urllib3**: outbound HTTP only to `core/utils.py:247` (RunPod, env-set URL), `core/b2_utils.py` + django-storages (env-set `AWS_S3_ENDPOINT_URL`, Backblaze). `requests` handles redirects itself and strips `Authorization` cross-host. Conditional on no `HTTPS_PROXY` in the prod env (not visible).
- **geopy**: only `core/utils.py:551` `distance()` on float tuples (`Point.from_sequence`, not `from_string`); user lat/lon strings (`core/views.py:550-551,573-574,593-594`) hit a `FloatField` before any geopy call.
- **setuptools**: build-time advisory; project has no `setup.py`/`pyproject.toml`/`MANIFEST.in`, builds no sdists.
- **pip**: Procfile is `gunicorn webibex.wsgi --timeout 120`; no custom index config in repo.

Reviewer could not see: Railway env vars (`HTTPS_PROXY`, `AWS_S3_ENDPOINT_URL`, `INFERENCE_ENDPOINT_URL_OVERRIDE`), whether `DATABASE_URL` uses a `postgis://` scheme, Railway builder config. No tests or dynamic checks were run.

## Upgrade constraints discovered
- `django-polymorphic` 3.1.0 runs `import pkg_resources` on import (reported by the subagent). Whether setuptools 83 still ships `pkg_resources` is unverified; if not, bumping setuptools breaks `polymorphic` → `filer` → app. Same class as the 2026-08-15 prod-only import outage (see `CLAUDE.md`).
- urllib3 2.x is blocked: botocore 1.29.165 caps urllib3 below 2 unconditionally (`Requires-Dist: urllib3 (<1.27,>=1.25.4)` in installed METADATA, no python-version marker; verified by fact-check, so the cap applies on prod's 3.12.5 too). `boto3==1.26.0` is pinned deliberately because Backblaze rejects `x-amz-checksum-algorithm` (per subagent).
- Prior JS triage: commit `45f3746` (2026-09-19) resolved 31 JS transitive CVEs; the three JS findings above remain/are new in build-time tooling only.

## Decisions
- No dependency changes made this session; waiting on contract signature before further work.
- Upgrade order suggested by reachability (all low urgency): Django 5.2.17 → sqlparse 0.6.0 → geopy 2.5.0 → setuptools 83 (after the `pkg_resources` check) → pip → urllib3/boto3 as a separate B2-compatibility project.

## Validation
Sonnet fact-check 2026-10-12: pins, JS versions, file:line claims, Django internals and git facts confirmed; one hedge corrected (botocore cap). Not verified: advisory IDs/severities (OSV, offline), socket.dev results, setuptools 83 `pkg_resources` question, severity totals recount, `requests` cross-host `Authorization` stripping.
