# Supply-chain audit 2026-11-20 (dependency list — private remote only)

Follow-up to `supply-chain-audit-2026-10-12.md`. State below is **before** any dependency change (certifi still 2025.1.31 at audit time).

## Scope and method
- `/supply-chain` over `requirements.txt` (35 pins) and `node/` (pnpm). `requirements-dev.txt` not CVE-scanned.
- Python CVEs from the OSV API (no `pip-audit`/`uv audit` path for plain `requirements.txt`). Severities are GitHub-advisory labels, not CVSS.
- No reachability review this run; the 2026-10-12 review covers only Django, sqlparse, urllib3, geopy, setuptools, pip.

## New since 2026-10-12
| Package | Installed | Advisories | Fixed in |
|---|---|---|---|
| certifi | 2025.1.31 | CRITICAL GHSA-6q48-6c7w-9gw2 (emSign roots, compromised intermediate keys) | 2026.10.18 |
| pillow | 12.3.0 | OSV range match: 5 CRITICAL, 24 HIGH, 14 MODERATE (+ unrated PYSEC duplicates). Individual advisories not inspected. | 13.0.0 |
| django-filer | 3.3.0 | MODERATE GHSA-vf85-8ff2-3r9w (missing per-object authorization on `discard`/`edit_*`) | 3.5.0 |

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

## Upgrade notes (checked 2026-11-20, PyPI metadata only)
- certifi 2026.10.18, pillow 13.0.0 (requires Python ≥3.11; prod is 3.12.5), Django 5.2.17 all exist and are not yanked.
- django-filer 3.3.0 and easy-thumbnails 2.10 do not cap pillow.
- Not tested: pillow 13 against image handling; setuptools 83 vs `pkg_resources` in django-polymorphic.
