"""T01-T05: pytest/Django test infrastructure itself.

These are the R1 keystone tests -- they assert the REAL import-time settings
value (no override_settings), confirm collection is clean, and confirm the
no_network guard is active for the whole suite.
"""

import importlib
import os
from pathlib import Path

import pytest
from django.conf import settings


# T01 --------------------------------------------------------------------
def test_settings_import_under_environment_test_is_safe():
    assert settings.DEBUG is False
    assert "debug_toolbar" not in settings.INSTALLED_APPS
    assert settings.DATABASES["default"]["ENGINE"].endswith("sqlite3")
    assert hasattr(settings, "MEDIA_ROOT")
    # webibex/settings.py sets EMAIL_BACKEND to the console backend for any
    # non-production ENVIRONMENT (verified by import succeeding without the
    # production branch's env(...) no-default reads for EMAIL_ADRESS /
    # EMAIL_HOST_PASSWORD crashing). But pytest-django calls Django's own
    # django.test.utils.setup_test_environment(), which unconditionally
    # overrides EMAIL_BACKEND to the locmem backend for the whole test run
    # (so tests can assert against django.core.mail.outbox) -- so by the
    # time this assertion runs, the observable value is always locmem, not
    # console. The actual safety property this test guards is "no real SMTP
    # backend with real credentials/network," so assert that directly
    # instead of a backend string this test can never observe under
    # pytest-django.
    assert settings.EMAIL_BACKEND != "django.core.mail.backends.smtp.EmailBackend"


# T02 --------------------------------------------------------------------
def test_endpoint_inference_import_time_default_args_resolve_without_crash():
    utils = importlib.import_module("core.utils")

    endpoint_id_default, endpoint_api_key_default = utils.endpoint_inference.__defaults__

    assert isinstance(endpoint_id_default, str)
    assert endpoint_id_default != ""
    assert isinstance(endpoint_api_key_default, str)
    assert endpoint_api_key_default != ""


# T03 --------------------------------------------------------------------
def test_db_management_test_script_is_excluded_from_collection():
    assert Path("db_management/test.py").exists()
    # The file's name does not match pytest's default python_files glob
    # ("test_*.py" / "*_test.py") *and* it is additionally defensively
    # listed in the root conftest.py collect_ignore, so it is never
    # imported by pytest either way.
    import conftest as root_conftest

    assert "db_management/test.py" in root_conftest.collect_ignore


def test_core_test_model_deleted():
    assert not Path("core/test_model.py").exists(), (
        "core/test_model.py must be deleted (no assertions, hardcoded path "
        "to another dev's machine) -- see finalized decision #1"
    )


# T04 --------------------------------------------------------------------
def test_no_network_guard_blocks_real_egress_attempts(no_network):
    """Confirms the no_network guard is live and actually blocks egress.

    A function-scoped mock can't observe calls that happened before it
    existed (e.g. at import time, during collection) -- asserting
    call_count == 0 right after the fixture is created would be trivially
    true regardless of whether the guard works at all. Instead, prove the
    guard is effective by making a call through the patched entry points
    and confirming it raises -- that's the property every other test in
    the suite relies on to keep real credentials/network out of the run.
    """
    import boto3
    import requests

    with pytest.raises(AssertionError, match="no_network guard tripped"):
        requests.post("https://example.invalid/")

    with pytest.raises(AssertionError, match="no_network guard tripped"):
        boto3.resource("s3")


# T05 --------------------------------------------------------------------
def test_pytest_ini_django_settings_module_resolves():
    assert os.environ["DJANGO_SETTINGS_MODULE"] == "webibex.settings"
    assert settings.configured is True
