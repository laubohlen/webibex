# webibex — Project Instructions

Local to this project, loaded alongside `~/.claude/CLAUDE.md`.

## Environment gotchas

- **Module-level/def-time-eager imports break prod but not test venv.** Code that imports or reads config at module scope (not lazily, e.g. Django settings, `mypy_boto3_s3`, `setuptools`/`pkg_resources`) can behave correctly in the dev/test venv while breaking in a prod-shaped environment with only `requirements.txt` installed — caused a real production outage (2026-08-15, `mypy_boto3_s3` unconditional module-level import). A `prod_parity` pytest marker was added afterward; as a standing pre-deploy check, also verify `webibex.wsgi` imports cleanly with only `requirements.txt` installed and prod-shaped env vars, not just via pytest-django.
- **Secrets fallback**: `~/.config/secrets/<NAME>` works as a token source in this sandbox when macOS Keychain doesn't — try both before declaring a secret unreachable (e.g. `SONAR_TOKEN`, `BRAIN_TOKEN`).

## Supply Chain Exceptions

- **`node/` uses `pnpm-lock.yaml`, not `package-lock.json`/`npm ci`** (deviation from the global convention). `npm` is permanently disabled in this project's devcontainer image (`npm-guard` policy, no runtime override) — `pnpm` is the only workable package manager here. `package.json` still pins exact versions (no `^`/`~`); `pnpm audit` is the CVE-scan equivalent of `npm audit`. See `docs/security-remediation-plan.md` (2026-09-19 JS transitive CVE fix) for the migration history.

## Headless Chromium (`dcg-chromium`) against this app

webibex is server-rendered Django (+ `django-allauth`), not an SPA — there's no client-side auth
SDK to mock, so the technique here is server-side, not the "alias the SDK module" trick that
applies to client-rendered apps.

```bash
dcg-chromium --dump-dom http://127.0.0.1:8000/<path>                       # plain HTTP, no extra flags
dcg-chromium --window-size=1280,900 --screenshot=<path>.png http://127.0.0.1:8000/<path>
```

- **Use `http://127.0.0.1:8000`, never `http://localhost:8000`.** Confirmed against
  `webibex/settings.py:44`: `ALLOWED_HOSTS = ["wibex.up.railway.app", "127.0.0.1",
  "localhost:8000"]`. Django's host validation strips the port before comparing (`Host:
  localhost:8000` → compared value is just `"localhost"`), so the `"localhost:8000"` entry can
  **never** actually match anything — hitting `localhost:8000` returns `400 DisallowedHost`.
  `127.0.0.1` is a real, correctly-matching entry. (The dead `ALLOWED_HOSTS` entry itself is a
  latent settings bug, separate from this note — not fixed here.)
- **Secure-cookie settings do NOT apply in ordinary local dev** — checked `settings.py:54-57`:
  `SESSION_COOKIE_SECURE`/`CSRF_COOKIE_SECURE`/`SECURE_SSL_REDIRECT` only activate when
  `ENVIRONMENT == "production"` or `POSTGRES_LOCALLY` (hardcoded `False`), both false in a normal
  `ENVIRONMENT=development` run. No `--ignore-certificate-errors`-driven cookie confusion here
  unless you're deliberately running with one of those flags set.
- **`--ignore-certificate-errors` is still needed for any Squid-bumped external HTTPS domain**
  (Chromium doesn't trust this container's Squid MITM CA the way `curl`/`pip`/`uv` do) — not
  relevant for the plain-HTTP local dev server itself, but relevant if a page makes external
  HTTPS requests (e.g. this app's OSM/map tile calls) during the capture.
- **Auth bypass, if ever needed, must be server-side** — three options, most-to-least invasive:
  1. A `settings.DEBUG`-gated + explicit-env-var-gated middleware/decorator calling
     `django.contrib.auth.login(request, user)` before the view runs (works with `allauth`'s
     session model unchanged — `allauth` still uses `django.contrib.sessions` underneath).
  2. Cookie injection via `playwright-core` driving this same `headless_shell` binary
     (`executablePath`): get a real `sessionid` via `manage.py shell` (`SessionStore`, set
     `_auth_user_id`/`_auth_user_backend`, `.save()`, read `.session_key`), then
     `browserContext.addCookies(...)` before navigating. No app-code changes.
  3. Django's test `Client` — backend-only, never touches a real browser.
     Whichever of (1)/(2) is used: unique marker string, gitignored new-file pattern, a
     pre-commit check on staged **file content** (not diff) for the marker — not added
     proactively here since no such patch exists in this repo yet.
- **DOM noise / false negatives to know about**: a capture without `--virtual-time-budget=6000`
  can land before an async route finishes mounting (misleadingly blank page — retry with a
  larger budget before treating a blank capture as a bug); a fontless render can "succeed" under
  a naive non-zero-byte check while genuinely missing text (own spike: 2727 bytes fontless vs
  12819 bytes with fonts on an identical page — don't use `> 0` as a render-succeeded oracle);
  with `DEBUG=False`, static files aren't served by `runserver` by default, so a `--screenshot`
  can come back with correct HTML but zero CSS applied — expected, not a rendering bug.
- Page `console.log`/`console.error` surface automatically on stderr as `INFO:CONSOLE:` lines.
