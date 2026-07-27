"""Root pytest configuration.

CRITICAL ORDERING: webibex/settings.py and core/b2_utils.py do fail-secure
env(X) (no default) at MODULE IMPORT TIME for SECRET_KEY / AWS_*.
core/utils.py:endpoint_inference() reads env("RUNPOD_ENDPOINT_ID") /
env("RUNPOD_API_KEY") as FUNCTION DEFAULT-ARGUMENT VALUES, evaluated once at
core.utils module-def time -- even earlier than a call-time read.

All required env vars are therefore set here, at MODULE SCOPE, before
Django settings get force-loaded and before django.setup() runs (this
conftest.py is always imported before any nested-directory conftest.py,
e.g. tests/conftest.py, regardless of pytest-django's own internal
hook-ordering -- see the django.setup() call below for why that matters).
"""

import os
from collections.abc import Generator
from typing import Any

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

# moto reads this to know which custom (non-AWS) endpoints it should
# intercept -- must be set before any `moto.mock_s3()` context starts, so
# module scope (here) is early enough. Matches AWS_S3_ENDPOINT_URL above.
os.environ.setdefault("MOTO_S3_CUSTOM_ENDPOINTS", "https://example-b2-endpoint.invalid")

# get_b2_resource() passes endpoint_url + signature_version='s3v4' but no
# region_name -- botocore can raise NoRegionError while resolving the
# signer without a region in scope. Harmless for the existing non-moto
# tests either way; defensive default for the moto_s3 tier.
os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")

# EXPLICIT django.setup(): once `tests` (or any nested test dir) is part
# of the collection args, pytest treats `tests/conftest.py` as an
# "initial" conftest too and loads it in the same early phase as this root
# conftest.py -- but ahead of pytest-django's own `_setup_django()` hookimpl,
# which normally populates the app registry. `tests/conftest.py` imports
# `filer.models` at module scope (needed for Folder-backed fixtures), which
# needs the app registry populated *now*, not whenever pytest-django gets to
# it. Calling django.setup() ourselves, right after env vars are set and
# before any nested conftest.py can load, removes the dependency on
# pytest-django's internal hook-ordering timing entirely. Safe to call even
# when pytest-django also calls it later -- django.setup() is a no-op once
# apps.ready is True.
import django

django.setup()

import boto3  # noqa: E402 -- must follow django.setup() above
import pytest  # noqa: E402 -- must follow django.setup() above
import requests  # noqa: E402 -- must follow django.setup() above


def _blocked_requests_post(*_args: object, **_kwargs: object) -> None:
    raise AssertionError(
        "Real network call via requests.post() attempted during tests -- "
        "no_network guard tripped. Use the `mock_runpod` fixture instead."
    )


def _blocked_boto3_resource(*_args: object, **_kwargs: object) -> None:
    raise AssertionError(
        "Real network call via boto3.resource() attempted during tests -- "
        "no_network guard tripped. Use the `mock_b2` fixture instead."
    )


@pytest.fixture(autouse=True)
def no_network(request: pytest.FixtureRequest) -> Generator[tuple[Any, Any | None]]:
    """Autouse guard: block real network egress for the whole suite.

    Patches boto3.resource and requests.post globally so accidental real
    calls raise instead of silently reaching AWS/B2/RunPod. Individual tests
    that need a mocked response should reconfigure the returned mocks (see
    `mock_runpod` / `mock_b2`) rather than removing this guard.

    Tests marked `@pytest.mark.moto_s3` (see tests/conftest.py::moto_b2) skip
    the boto3.resource patch: moto's `mock_s3()` intercepts at the
    botocore/urllib3 HTTP layer, not by replacing boto3.resource, so leaving
    this guard active under moto_s3 would incorrectly block every
    legitimate boto3.resource() call the B2 code makes even while moto is
    running. The requests.post patch stays unconditional -- moto_s3 only
    concerns S3, not other outbound HTTP.

    Returns a (post_patch, resource_patch) tuple; `resource_patch` is `None`
    when the boto3.resource patch was skipped (moto_s3-marked tests). See
    `mock_runpod` for the existing consumer of this contract.
    """
    from contextlib import ExitStack
    from unittest import mock

    with ExitStack() as stack:
        post_patch = stack.enter_context(
            mock.patch.object(requests, "post", side_effect=_blocked_requests_post)
        )
        resource_patch = None
        if request.node.get_closest_marker("moto_s3") is None:
            resource_patch = stack.enter_context(
                mock.patch.object(
                    boto3, "resource", side_effect=_blocked_boto3_resource
                )
            )
        yield post_patch, resource_patch


# Hardening for the moto_s3 bypass above: nothing else stops a future test
# from adding @pytest.mark.moto_s3 just to silence the boto3.resource guard
# without actually mocking S3, letting a real network call through
# unguarded. Fail collection instead. The one exception is a guard test that
# deliberately asserts the *requests.post* patch still applies under the
# marker -- it doesn't touch S3 by design, so it has no moto_b2 fixture.
_MOTO_S3_MISUSE_ALLOWLIST = {"test_moto_s3_marker_does_not_bypass_requests_post_guard"}


def pytest_collection_modifyitems(items: list[pytest.Function]) -> None:
    for item in items:
        if (
            item.get_closest_marker("moto_s3") is not None
            and "moto_b2" not in item.fixturenames
            and item.originalname not in _MOTO_S3_MISUSE_ALLOWLIST
        ):
            raise pytest.UsageError(
                f"{item.nodeid}: @pytest.mark.moto_s3 requires the `moto_b2` fixture "
                "(this marker only exists to let moto intercept boto3.resource -- "
                "without moto_b2 active, boto3.resource() would reach the real "
                "network)."
            )
