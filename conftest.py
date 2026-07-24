"""Root pytest configuration.

CRITICAL ORDERING: webibex/settings.py and core/b2_utils.py do fail-secure
env(X) (no default) at MODULE IMPORT TIME for SECRET_KEY / AWS_*.
core/utils.py:endpoint_inference() reads env("RUNPOD_ENDPOINT_ID") /
env("RUNPOD_API_KEY") as FUNCTION DEFAULT-ARGUMENT VALUES, evaluated once at
core.utils module-def time -- even earlier than a call-time read.

All required env vars are therefore set here, at MODULE SCOPE, before
Django settings get force-loaded and before django.setup() runs (this
conftest.py is always imported before any nested-directory conftest.py,
e.g. core/tests/conftest.py, regardless of pytest-django's own internal
hook-ordering -- see the django.setup() call below for why that matters).
"""

import os
import sys

# ENVIRONMENT DRIFT WORKAROUND: the installed setuptools (83.0.0) no longer
# ships pkg_resources, but requirements.txt pins setuptools==78.1.1 (which
# does). django-filer -> django-polymorphic imports pkg_resources at module
# scope. Re-pinning the venv to the requirements.txt version requires network
# access this sandbox doesn't have, and requirements.txt's setuptools line is
# explicitly out of scope (conflicts with a pending git stash). pip vendors a
# fully standalone, functional copy of pkg_resources -- reuse it here so
# third-party imports resolve without touching requirements.txt or production
# source. Remove once the venv is re-synced to the pinned setuptools version.
if "pkg_resources" not in sys.modules:
    try:
        import pkg_resources  # noqa: F401
    except ModuleNotFoundError:
        import pip._vendor.pkg_resources as _pkg_resources

        sys.modules["pkg_resources"] = _pkg_resources

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "webibex.settings")
os.environ.setdefault("ENVIRONMENT", "test")
os.environ.setdefault("SECRET_KEY", "test-secret-key-not-for-production")
os.environ.setdefault("AWS_ACCESS_KEY_ID", "test-aws-access-key-id")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "test-aws-secret-access-key")
os.environ.setdefault("AWS_S3_ENDPOINT_URL", "https://example-b2-endpoint.invalid")
os.environ.setdefault("AWS_STORAGE_BUCKET_NAME", "test-bucket")
os.environ.setdefault("AWS_S3_REGION_NAME", "us-west-000")
os.environ.setdefault("RUNPOD_ENDPOINT_ID", "test-runpod-endpoint-id")
os.environ.setdefault("RUNPOD_API_KEY", "test-runpod-api-key")

# db_management/test.py is a one-off data-migration script (not a real test) --
# it imports Django models at module scope and mutates a live DB, it must
# never be picked up by pytest's collection glob.
collect_ignore = ["db_management/test.py"]

# EXPLICIT django.setup(): once `core/tests` (or any nested test dir) is part
# of the collection args, pytest treats `core/tests/conftest.py` as an
# "initial" conftest too and loads it in the same early phase as this root
# conftest.py -- but ahead of pytest-django's own `_setup_django()` hookimpl,
# which normally populates the app registry. `core/tests/conftest.py` imports
# `filer.models` at module scope (needed for Folder-backed fixtures), which
# needs the app registry populated *now*, not whenever pytest-django gets to
# it. Calling django.setup() ourselves, right after env vars are set and
# before any nested conftest.py can load, removes the dependency on
# pytest-django's internal hook-ordering timing entirely. Safe to call even
# when pytest-django also calls it later -- django.setup() is a no-op once
# apps.ready is True.
import django  # noqa: E402

django.setup()

import boto3  # noqa: E402  (import after env setdefault, before fixtures)
import pytest  # noqa: E402
import requests  # noqa: E402


def _blocked_requests_post(*_args, **_kwargs):
    raise AssertionError(
        "Real network call via requests.post() attempted during tests -- "
        "no_network guard tripped. Use the `mock_runpod` fixture instead."
    )


def _blocked_boto3_resource(*_args, **_kwargs):
    raise AssertionError(
        "Real network call via boto3.resource() attempted during tests -- "
        "no_network guard tripped. Use the `mock_b2` fixture instead."
    )


@pytest.fixture(autouse=True)
def no_network():
    """Autouse guard: block real network egress for the whole suite.

    Patches boto3.resource and requests.post globally so accidental real
    calls raise instead of silently reaching AWS/B2/RunPod. Individual tests
    that need a mocked response should reconfigure the returned mocks (see
    `mock_runpod` / `mock_b2`) rather than removing this guard.
    """
    from unittest import mock

    with (
        mock.patch.object(requests, "post", side_effect=_blocked_requests_post) as post_patch,
        mock.patch.object(boto3, "resource", side_effect=_blocked_boto3_resource) as resource_patch,
    ):
        yield post_patch, resource_patch
