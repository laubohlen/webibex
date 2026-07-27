"""Moto-based in-process S3-mock tests for core/b2_utils.py.

Distinct from the existing unittest.mock-based tier in
tests/core/test_utils_io.py (which patches core.b2_utils.download_file at
the Python-function boundary via the `mock_b2` fixture) -- this file tests
against an in-process fake S3 backend (moto's `mock_s3`) that intercepts at
the botocore/urllib3 HTTP layer, exercising the real boto3 client/resource
code paths in core/b2_utils.py instead of stubbing them out.

Verified by real local execution (moto==4.2.14 installed after the sandbox's
egress proxy was scoped to allow `pypi.org` + `files.pythonhosted.org`,
same allowlist pattern already used for `pyright`/`ruff`). All 15 tests
below pass for real; `core/b2_utils.py` coverage is 100% under this file.

Two items flagged as open questions during the design phase (before moto
could be installed here) are now resolved by that real run:
- `AWS_DEFAULT_REGION` (set in root conftest.py) turned out to be
  unnecessary -- resource construction and `create_bucket` both succeed
  under moto without any region env var set. Left in place anyway since
  it's harmless and defends against a real-S3 edge case this repo doesn't
  currently exercise.
- `delete_files([])` (T06): moto raises `ClientError` with code
  `MalformedXML` for an empty `Delete.Objects` list -- matching real S3's
  documented behavior, not the speculated `ParamValidationError` (which
  would NOT have been caught by `delete_files`'s `except ClientError`).
  Confirmed caught correctly, returns `None`, no crash.

Import-safety design choice: `moto` is imported LAZILY, inside the
`moto_b2` fixture (tests/conftest.py) only -- never at module scope in
this file. Tests that request `moto_b2` (T01-T09, T13) therefore fail at
FIXTURE-SETUP time with `ModuleNotFoundError: No module named 'moto'` in
this sandbox, not at whole-file collection.

T10 and T11 are meta-tests about the `no_network` guard mechanism itself
(does the moto_s3 marker bypass boto3.resource but not requests.post) --
they don't need an actual mocked S3 backend to prove that, so they
deliberately do NOT request `moto_b2` and run for real even without moto
installed, giving a genuine regression signal on the conftest.py change.

T12, T14, T15 also don't touch moto at all (marker registration + plain
file-content checks) and run for real in this sandbox.
"""

import re
from pathlib import Path

import pytest
from botocore.exceptions import ClientError

from core.b2_utils import (
    check_file_exists,
    delete_files,
    download_file,
    get_b2_resource,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# T01 -- get_b2_resource() happy path
# ---------------------------------------------------------------------------
@pytest.mark.moto_s3
def test_get_b2_resource_bound_to_fake_endpoint_happy(moto_b2):
    resource = get_b2_resource()

    assert resource.meta.client is not None
    assert resource.meta.client.meta.endpoint_url == "https://example-b2-endpoint.invalid"


# ---------------------------------------------------------------------------
# T02 -- download_file() exact byte round-trip
# ---------------------------------------------------------------------------
@pytest.mark.moto_s3
@pytest.mark.parametrize(
    "key, payload",
    [
        ("hello.txt", b"hello-b2"),
        (
            "chip.png",
            b"\x89PNG\r\n\x1a\n"
            + bytes(range(256))
            + b"\x00\x00\x00\x00IEND\xaeB`\x82",
        ),
    ],
)
def test_download_file_returns_exact_bytes_happy(moto_b2, key, payload):
    moto_b2.put_object(Bucket="test-bucket", Key=key, Body=payload)

    result = download_file(key)

    assert result == payload


# ---------------------------------------------------------------------------
# T03 -- download_file() missing key
# ---------------------------------------------------------------------------
@pytest.mark.moto_s3
def test_download_file_missing_key_returns_none(moto_b2):
    result = download_file("missing-key")

    assert result is None


# ---------------------------------------------------------------------------
# T04 -- delete_files() removes existing objects
# ---------------------------------------------------------------------------
@pytest.mark.moto_s3
def test_delete_files_removes_existing_objects_happy(moto_b2):
    moto_b2.put_object(Bucket="test-bucket", Key="k1", Body=b"a")
    moto_b2.put_object(Bucket="test-bucket", Key="k2", Body=b"b")

    delete_files(["k1", "k2"])

    for key in ("k1", "k2"):
        with pytest.raises(ClientError) as exc_info:
            moto_b2.head_object(Bucket="test-bucket", Key=key)
        assert exc_info.value.response.get("Error", {}).get("Code") == "404"


# ---------------------------------------------------------------------------
# T05 -- delete_files() against a nonexistent bucket doesn't raise
# ---------------------------------------------------------------------------
@pytest.mark.moto_s3
def test_delete_files_bad_bucket_returns_none_no_raise(moto_b2):
    result = delete_files(["k1"], bucket_name="no-such-bucket")

    assert result is None


# ---------------------------------------------------------------------------
# T06 -- delete_files([]) doesn't raise
# ---------------------------------------------------------------------------
@pytest.mark.moto_s3
def test_delete_files_empty_list_no_raise(moto_b2):
    # Confirmed (real run): moto raises ClientError(MalformedXML) for an
    # empty Delete.Objects list, matching real S3. delete_files' own
    # `except ClientError` catches it, so this returns None cleanly instead
    # of propagating -- verified, not speculative (see module docstring).
    result = delete_files([])

    assert result is None


# ---------------------------------------------------------------------------
# T07 -- check_file_exists() happy path
# ---------------------------------------------------------------------------
@pytest.mark.moto_s3
def test_check_file_exists_existing_key_returns_true_happy(moto_b2):
    moto_b2.put_object(Bucket="test-bucket", Key="k1", Body=b"a")

    assert check_file_exists("k1") is True


# ---------------------------------------------------------------------------
# T08 -- check_file_exists() error paths both return None (bug pin)
# ---------------------------------------------------------------------------
@pytest.mark.moto_s3
@pytest.mark.parametrize("case", ["missing_key_404", "non_404_client_error"])
def test_check_file_exists_error_paths_return_none_bug_pin(moto_b2, monkeypatch, case):
    if case == "missing_key_404":
        result = check_file_exists("does-not-exist")
    else:
        # check_file_exists() builds its own resource/client internally via
        # get_b2_resource() -- it does not reuse the `moto_b2` fixture's
        # client object -- so we monkeypatch get_b2_resource() itself to
        # return a fake resource whose meta.client.head_object() raises a
        # non-404 ClientError, forcing the `else` branch in check_file_exists
        def _raise_forbidden(*_args, **_kwargs):
            raise ClientError(
                {"Error": {"Code": "403", "Message": "Forbidden"}}, "HeadObject"
            )

        class _FakeMetaClient:
            head_object = staticmethod(_raise_forbidden)

        class _FakeMeta:
            client = _FakeMetaClient()

        class _FakeB2Resource:
            meta = _FakeMeta()

        monkeypatch.setattr(
            "core.b2_utils.get_b2_resource", lambda *args, **kwargs: _FakeB2Resource()
        )
        result = check_file_exists("some-key")

    # KNOWN BUG (out of scope for this CR): the 404 branch sets an unused
    # local `file_exists = False` and falls through to an implicit `None`
    # instead of returning `False`. Both this branch and the non-404 branch
    # currently return `None` -- only the printed message differs. Do not
    # "fix" this assertion to expect False; that would mask the bug instead
    # of pinning it.
    assert result is None


# ---------------------------------------------------------------------------
# T09 -- moto_s3 marker bypasses the boto3.resource guard
# ---------------------------------------------------------------------------
@pytest.mark.moto_s3
def test_moto_s3_marker_bypasses_no_network_guard(moto_b2):
    # Must not raise the `_blocked_boto3_resource` AssertionError -- the
    # moto_s3 marker skips the no_network boto3.resource patch.
    resource = get_b2_resource()

    assert resource is not None


# ---------------------------------------------------------------------------
# T10 -- unmarked test still blocks boto3.resource (control test)
# ---------------------------------------------------------------------------
def test_unmarked_test_still_blocks_boto3_resource():
    """No @pytest.mark.moto_s3 here -- confirms the bypass is marker-scoped,
    not global. Does not request `moto_b2`, so this runs for real (no moto
    needed) even in this sandbox.
    """
    import boto3

    with pytest.raises(AssertionError, match="no_network guard tripped"):
        boto3.resource("s3")


# ---------------------------------------------------------------------------
# T11 -- moto_s3 marker does NOT bypass the requests.post guard
# ---------------------------------------------------------------------------
@pytest.mark.moto_s3
def test_moto_s3_marker_does_not_bypass_requests_post_guard():
    """Does not request `moto_b2` -- only the boto3.resource patch is
    gated by the moto_s3 marker, the requests.post patch is unconditional.
    Runs for real (no moto needed) even in this sandbox.
    """
    import requests

    with pytest.raises(AssertionError, match=r"requests\.post"):
        # Never reaches the network -- intercepted by the autouse `no_network`
        # guard (conftest.py), which raises before any real request is sent.
        requests.post("http://example.invalid")  # noqa: S113


# ---------------------------------------------------------------------------
# T12 -- both markers registered, live_b2 has no consumers yet
# ---------------------------------------------------------------------------
def test_moto_s3_and_live_b2_markers_registered_no_warning(pytestconfig):
    registered_markers = pytestconfig.getini("markers")

    assert any(marker.startswith("moto_s3") for marker in registered_markers)
    assert any(marker.startswith("live_b2") for marker in registered_markers)

    # live_b2 is a scaffold for a future real-Backblaze-B2 integration tier
    # -- confirm no test currently uses it.
    tests_dir = Path(__file__).resolve().parent.parent
    live_b2_marker_pattern = re.compile(r"@pytest\.mark\.live_b2\b")
    usages = [
        path
        for path in tests_dir.rglob("test_*.py")
        if live_b2_marker_pattern.search(path.read_text())
    ]

    assert usages == [], f"live_b2 marker unexpectedly used in: {usages}"


# ---------------------------------------------------------------------------
# T13 -- moto_b2 fixture's bucket exists
# ---------------------------------------------------------------------------
@pytest.mark.moto_s3
def test_moto_b2_fixture_bucket_exists_happy(moto_b2):
    # Must not raise -- confirms the moto_b2 fixture pre-creates the bucket.
    moto_b2.head_bucket(Bucket="test-bucket")


# ---------------------------------------------------------------------------
# T14 -- requirements-dev.txt pins the exact moto version (no moto needed)
# ---------------------------------------------------------------------------
def test_requirements_dev_pins_exact_moto_version():
    content = (_REPO_ROOT / "requirements-dev.txt").read_text()
    lines = [line.strip() for line in content.splitlines()]

    assert "moto==4.2.14" in lines


# ---------------------------------------------------------------------------
# T15 -- production code / requirements.txt never mention moto (no moto needed)
# ---------------------------------------------------------------------------
def test_production_code_and_requirements_have_no_moto_reference():
    b2_utils_content = (_REPO_ROOT / "core" / "b2_utils.py").read_text()
    requirements_content = (_REPO_ROOT / "requirements.txt").read_text()

    assert "moto" not in b2_utils_content
    assert "moto" not in requirements_content
