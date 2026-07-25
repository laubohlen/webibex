#!/usr/bin/env python3
"""Tracked, hardened local e2e dev-server runner for manual/Playwright testing.

Promotes the gitignored scratch script `tmp/e2e_runserver_with_media.py`
(see `docs/changes/2026-07-25-local-e2e-dev-server-runner.md`) into a
committed, reusable tool so this setup survives `tmp/` cleanup and doesn't
need re-deriving every session.

What it does, and why:

1. **Serves MEDIA_URL/STATIC_URL like DEBUG=True, without django-debug-toolbar**
   (not installed in this environment; real `ENVIRONMENT=development` crashes
   on startup because of it). `INSTALLED_APPS`/`MIDDLEWARE`'s debug_toolbar
   decision is baked in at `webibex/settings.py` *import* time, based on
   `ENVIRONMENT` at process start -- flipping `settings.DEBUG` afterwards
   can't undo or redo that. But `webibex/urls.py`'s
   `if settings.DEBUG: urlpatterns += static(...)` block evaluates at
   *its own* module-import time, and `webibex.urls` isn't imported until
   Django first resolves a URL (lazily, on first request). So: keep
   `ENVIRONMENT=e2e-test` (skips debug_toolbar at settings-load time), then
   flip `settings.DEBUG = True` here, before `urls.py` has ever been
   imported -- it sees `DEBUG=True` when it finally loads and adds the
   `static()` patterns itself.

2. **Optionally fills in dummy credentials** (`--use-dummy-creds`) so the
   server can boot without real AWS/B2/RunPod secrets -- matches the exact
   placeholder values `conftest.py` already uses for the pytest suite. Only
   fills a var if it isn't already set (never clobbers a real `.env`).
   Safe because these vars are read unconditionally at import time
   (`env(X)`, no default -- fail-secure) but never actually *called out to*
   unless a view touches S3/B2 storage or the RunPod inference endpoint.

3. **Optionally points `embed_new_chip()`/`endpoint_inference()` at a local
   inference container** (`--inference-override URL`) instead of the real
   `api.runpod.ai` (unreachable from this devcontainer's sandboxed network
   regardless of credentials -- see
   `docs/changes/2026-07-25-runpod-endpoint-override-and-script-hardening.md`).
   Sets `INFERENCE_ENDPOINT_URL_OVERRIDE`, read at call time by
   `core/utils.py:endpoint_inference()`.

4. **Optionally creates a throwaway test user** (`--create-test-user NAME`)
   for login flows, rather than needing to know an existing account's
   password.

Hard guard: refuses to run at all if the resolved `ENVIRONMENT` is
`"production"` -- this is dev-only tooling, never meant to touch a real
deployment.

Usage:
    scripts/run_local_e2e_server.py [addr] \\
        [--environment e2e-test] \\
        [--use-dummy-creds] \\
        [--inference-override http://localhost:8001/runsync] \\
        [--create-test-user NAME] [--test-password PASSWORD]

Example (matches the manual verification done in the
2026-07-25 auth/session-hardening CR):
    scripts/run_local_e2e_server.py 0.0.0.0:8000 \\
        --use-dummy-creds \\
        --inference-override http://localhost:8001/runsync \\
        --create-test-user settings_verify_temp
"""

from __future__ import annotations

import argparse
import os
import sys

# conftest.py's exact dummy values -- kept in sync deliberately, both boot
# the same settings.py import-time env() reads with the same placeholders.
DUMMY_CREDENTIALS: dict[str, str] = {
    "SECRET_KEY": "test-secret-key-not-for-production",
    "AWS_ACCESS_KEY_ID": "test-aws-access-key-id",
    "AWS_SECRET_ACCESS_KEY": "test-aws-secret-access-key",
    "AWS_S3_ENDPOINT_URL": "https://example-b2-endpoint.invalid",
    "AWS_STORAGE_BUCKET_NAME": "test-bucket",
    "AWS_S3_REGION_NAME": "us-west-000",
    "RUNPOD_ENDPOINT_ID": "test-runpod-endpoint-id",
    "RUNPOD_API_KEY": "test-runpod-api-key",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("addr", nargs="?", default="0.0.0.0:8000")
    parser.add_argument("--environment", default="e2e-test")
    parser.add_argument("--use-dummy-creds", action="store_true")
    parser.add_argument("--inference-override", default=None)
    parser.add_argument("--create-test-user", default=None, metavar="USERNAME")
    parser.add_argument("--test-password", default="LocalE2E-Test-12345")
    return parser.parse_args()


def apply_dummy_credentials() -> list[str]:
    filled = []
    for key, value in DUMMY_CREDENTIALS.items():
        if key not in os.environ:
            os.environ[key] = value
            filled.append(key)
    return filled


def create_test_user(username: str, password: str) -> None:
    from django.contrib.auth import get_user_model

    user_model = get_user_model()
    user, created = user_model.objects.get_or_create(
        username=username, defaults={"email": f"{username}@example.invalid"}
    )
    user.set_password(password)
    user.is_active = True
    user.save()
    action = "created" if created else "updated"
    print(f"test user {action}: username={username!r} email={user.email!r}", flush=True)


def main() -> None:
    args = parse_args()

    os.environ.setdefault("ENVIRONMENT", args.environment)
    if os.environ["ENVIRONMENT"] == "production":
        print(
            "refusing to run: ENVIRONMENT=production -- this is dev-only "
            "tooling, never meant to touch a real deployment.",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(1)

    if args.use_dummy_creds:
        filled = apply_dummy_credentials()
        if filled:
            print(f"dummy credentials filled (not already set): {', '.join(filled)}", flush=True)

    if args.inference_override:
        os.environ["INFERENCE_ENDPOINT_URL_OVERRIDE"] = args.inference_override
        print(f"INFERENCE_ENDPOINT_URL_OVERRIDE={args.inference_override}", flush=True)

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "webibex.settings")

    import django

    django.setup()

    from django.conf import settings

    settings.DEBUG = True  # see module docstring point 1

    if args.create_test_user:
        create_test_user(args.create_test_user, args.test_password)
        print(
            f"login with: {args.create_test_user}@example.invalid / {args.test_password}",
            flush=True,
        )

    from django.core.management import execute_from_command_line

    execute_from_command_line([sys.argv[0], "runserver", args.addr])


if __name__ == "__main__":
    main()
