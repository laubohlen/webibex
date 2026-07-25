# CR: tracked local e2e dev-server runner (`scripts/run_local_e2e_server.py`)

**What changed:**
- New: `scripts/run_local_e2e_server.py` — a tracked, committed promotion of
  the prior gitignored scratch script `tmp/e2e_runserver_with_media.py`
  (see `docs/changes/2026-07-23-e2e-local-runpod-test-tooling.md`), extended
  with three new opt-in flags:
  - `--use-dummy-creds`: fills `SECRET_KEY`/`AWS_*`/`RUNPOD_*` with the exact
    placeholder values `conftest.py` already uses for pytest, but only for
    vars not already set in the environment (never clobbers a real `.env`).
    Safe because those vars are read unconditionally at `webibex/settings.py`
    import time (`env(X)`, no default) but never actually called out to
    unless a view touches S3/B2 storage or RunPod inference.
  - `--inference-override URL`: sets `INFERENCE_ENDPOINT_URL_OVERRIDE` (the
    hook added in `docs/changes/2026-07-25-runpod-endpoint-override-and-script-hardening.md`)
    so `core/utils.py:endpoint_inference()` calls a local RunPod inference
    container instead of the real `api.runpod.ai`, which is unreachable from
    this devcontainer's sandboxed network regardless of credentials.
  - `--create-test-user NAME [--test-password PW]`: creates/updates a
    throwaway Django user, since the existing `db.sqlite3`'s known usernames
    (`smoketestuser`, `chiptestuser`, `e2e_admin`) have no recorded passwords.
  - Hard guard: refuses to run at all if the resolved `ENVIRONMENT` is
    `"production"` — dev-only tooling, never meant to touch a real deployment.
  - All informational `print()` calls use `flush=True` — stdout is
    block-buffered when redirected to a file under `nohup ... &`, so without
    an explicit flush, the credential/override/test-user summary lines never
    reach the log while the (long-running) server process is still alive.
    Found and fixed during this session's own testing of the script.

**Follow-up action:** none required. No project skill was created for this
(explicit user preference — standalone, deterministic scripts living
entirely within the repo, not a `.claude/skills/` wrapper). This doc plus
the script's own module docstring and `--help` output are the reference.

**Do NOT:**
- Assume the two prints-per-line in the log (once "created"/"filled", once
  "updated") are a bug — Django's autoreloader re-execs this entire script
  once as a subprocess (`RUN_MAIN` env var mechanism) before actually
  serving, so the whole `main()` runs twice. Harmless: `get_or_create` +
  `set_password` is idempotent, and the second pass's env-var checks find
  everything already set (no re-print of "dummy credentials filled" on
  the second pass specifically, since that one IS conditional on "not
  already set" — only the override/test-user lines repeat).
- Expect a live server started this way to be reachable from a host
  browser without a host-side tunnel (e.g. `bin/dev-tunnel`) — that tooling
  lives on the host, not in this devcontainer, and isn't something a Claude
  session running inside the devcontainer can drive itself. Verify via
  `curl` from inside the devcontainer; hand off actual browser click-through
  to the user or a host-side Playwright session.
- Forget that changing any of the three flags requires a full process
  restart (`kill` the PID, re-run) — this is env-var configuration, not
  code, so Django's autoreload (which only watches file changes) won't
  pick it up.

**Trigger:** any future manual verification of a webibex change that needs
a real running server (not just `pytest`) — auth/session/settings changes,
UI/template changes, or as the base for recording Playwright tests against
a locally running instance.

**Why:** this session needed to manually verify login/logout still worked
after the `SESSION_COOKIE_SECURE`/`CSRF_COOKIE_SECURE`/`SECURE_SSL_REDIRECT`/
etc. hardening change (see
`docs/security-remediation-plan.md`, "TODO — auth/session hardening
settings missing ... RESOLVED 2026-07-25"). A live server was needed because
unit tests only assert settings *values*, not actual request/response
behavior. The existing `tmp/e2e_runserver_with_media.py` from a prior
session already solved the DEBUG/static-files/RunPod-override problem, but
being gitignored scratch meant it would be lost and need re-deriving next
time — not worth doing twice, especially since Playwright test recording
will need this same setup again.

**Verify:**
```bash
.venv/bin/python scripts/run_local_e2e_server.py 0.0.0.0:8000 \
  --use-dummy-creds --create-test-user verify_temp
# then, from another shell:
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8000/accounts/login/
# → 200
```
Confirmed working this session: server boots with zero real secrets, serves
`/accounts/login/` and `/` at 200, a throwaway user can log in (note:
allauth's `login` field is the *email* address here —
`ACCOUNT_AUTHENTICATION_METHOD = "email"` in `webibex/settings.py:259` — not
the username), reach a login-required page, and log out correctly. Also
confirmed the `--inference-override` flag correctly routes
`endpoint_inference()` to a locally running RunPod container instead of the
unreachable real endpoint.

**Rollback:** delete `scripts/run_local_e2e_server.py`. No app code is
touched by this CR — it's a standalone dev-tooling script with no imports
from the rest of the codebase beyond Django/manage.py itself.
