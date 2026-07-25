"""Regression tests for the production-only security hardening block added
to `webibex/settings.py` (SESSION_COOKIE_SECURE, CSRF_COOKIE_SECURE,
SECURE_SSL_REDIRECT, SECURE_PROXY_SSL_HEADER, SECURE_HSTS_SECONDS) -- see
docs/security-remediation-plan.md, "TODO -- auth/session hardening settings
missing (found 2026-07-24, RESOLVED 2026-07-25)".

T01-T04 from the approved code-analyst test spec matrix. These assert the
REAL import-time settings values via `importlib.reload`, matching the
established project pattern in `tests/webibex/test_infra.py` (no
`@override_settings` -- that patches already-evaluated settings and never
re-runs the module-level gate logic these tests exist to guard).

T05 (POSTGRES_LOCALLY == True alternate path) and T06 (`manage.py
check --deploy` subprocess test) are intentionally not implemented -- see
docs/security-remediation-plan.md for rationale.
"""

import importlib

import pytest

import webibex.settings

# Names introduced by the new `if ENVIRONMENT == "production" or
# POSTGRES_LOCALLY == True:` block in webibex/settings.py. There is no
# `else` branch, so under any other ENVIRONMENT these names are simply never
# assigned -- `hasattr` is the correct presence check, not a comparison
# against a "disabled" sentinel value.
SECURITY_HARDENING_SETTINGS = (
    "SESSION_COOKIE_SECURE",
    "CSRF_COOKIE_SECURE",
    "SECURE_SSL_REDIRECT",
    "SECURE_PROXY_SSL_HEADER",
    "SECURE_HSTS_SECONDS",
)


@pytest.fixture
def reload_settings_env(monkeypatch):
    """Reload `webibex.settings` under a patched environment.

    Returns a `reload_with(**env_vars)` callable: sets each env var via
    `monkeypatch.setenv`, then `importlib.reload`s `webibex.settings` and
    returns the reloaded module object (never `django.conf.settings` --
    that's a cached `LazyObject`, stale after a raw module reload).

    Teardown restores ENVIRONMENT=test (the ambient pytest state set by the
    root conftest.py) and reloads the module again. That reload alone is
    NOT sufficient to fully reset state: `importlib.reload` re-executes the
    module body in the module's *existing* `__dict__` -- it does not clear
    attributes that the re-executed code path no longer assigns. Under
    ENVIRONMENT=test the hardening `if` block is skipped entirely, so any
    of the 5 names set by a prior production/dev-local reload in this test
    would otherwise silently leak into every test that runs afterward in
    this same pytest process (the module object is process-global). The
    explicit `delattr` loop below closes that gap.
    """

    def _reload_with(**env_vars):
        for key, value in env_vars.items():
            monkeypatch.setenv(key, value)
        return importlib.reload(webibex.settings)

    yield _reload_with

    monkeypatch.setenv("ENVIRONMENT", "test")
    reloaded = importlib.reload(webibex.settings)
    for name in SECURITY_HARDENING_SETTINGS:
        if hasattr(reloaded, name):
            delattr(reloaded, name)


# T01 (P0, happy path) ------------------------------------------------------
def test_hardening_settings_present_with_correct_values_under_environment_production(
    reload_settings_env,
):
    """Under ENVIRONMENT=production, all 5 hardening settings are set correctly.

    Also covers T04: SECURE_PROXY_SSL_HEADER is exactly the tuple Django's
    SecurityMiddleware expects, and is present whenever SECURE_SSL_REDIRECT
    is True -- the coupling that prevents a Railway redirect loop (Railway
    terminates TLS at its edge and forwards plain HTTP).
    """
    reloaded = reload_settings_env(
        ENVIRONMENT="production",
        DATABASE_URL="sqlite://:memory:",
        EMAIL_ADRESS="test@example.invalid",
        EMAIL_HOST_PASSWORD="test-email-password",
    )

    assert reloaded.SESSION_COOKIE_SECURE is True
    assert reloaded.CSRF_COOKIE_SECURE is True
    assert reloaded.SECURE_SSL_REDIRECT is True
    assert reloaded.SECURE_PROXY_SSL_HEADER == ("HTTP_X_FORWARDED_PROTO", "https")
    assert reloaded.SECURE_HSTS_SECONDS == 3600
    if reloaded.SECURE_SSL_REDIRECT is True:
        assert hasattr(reloaded, "SECURE_PROXY_SSL_HEADER")


# T02 (P0, no-regression keystone) ------------------------------------------
def test_hardening_settings_absent_under_ambient_environment_test():
    """Under the ambient ENVIRONMENT=test (no reload), none of the 5 new
    hardening settings exist on the settings module.

    This is the cheapest and most important regression guard -- it is also
    what would have caught the POSTGRES_LOCALLY ordering bug directly (a
    `NameError` at import time under ENVIRONMENT=test would have failed
    collection entirely, well before this specific assertion).
    """
    for name in SECURITY_HARDENING_SETTINGS:
        assert not hasattr(webibex.settings, name)


# T03 (P1, dev regression) ---------------------------------------------------
def test_hardening_settings_absent_under_environment_development(reload_settings_env):
    """Under ENVIRONMENT=development, none of the 5 hardening settings exist.

    Proves local dev over plain http://127.0.0.1:8000 is unaffected. No
    extra no-default env vars are needed: the dev branch never satisfies
    `ENVIRONMENT == "production" or POSTGRES_LOCALLY == True` (POSTGRES_LOCALLY
    is hardcoded False), so it never hits the DB/STORAGES/EMAIL/hardening
    gates that require DATABASE_URL/EMAIL_ADRESS/EMAIL_HOST_PASSWORD.
    """
    reloaded = reload_settings_env(ENVIRONMENT="development")

    for name in SECURITY_HARDENING_SETTINGS:
        assert not hasattr(reloaded, name)
