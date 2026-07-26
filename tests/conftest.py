"""Shared fixtures for the `core` app test package."""

from io import BytesIO
from types import SimpleNamespace
from unittest import mock

import boto3
import pytest
from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import SimpleUploadedFile
from filer.models import Folder
from PIL import Image

from core.models import Animal, IbexImage, Region


# ---------------------------------------------------------------------------
# Duck-typed stubs for P0 (no-DB) pure-function tests.
# ---------------------------------------------------------------------------
class RegionStub:
    """Duck-typed stand-in for core.models.Region -- no DB required."""

    def __init__(self, origin_latitude=None, origin_longitude=None, radius=None):
        self.origin_latitude = origin_latitude
        self.origin_longitude = origin_longitude
        self.radius = radius


class _IbexImageStub:
    def __init__(self, animal_id):
        self.animal_id = animal_id


class ChipStub:
    """Duck-typed stand-in for an IbexChip + nested ibex_image.animal_id."""

    def __init__(self, animal_id):
        self.ibex_image = _IbexImageStub(animal_id)


@pytest.fixture
def region_stub_cls():
    return RegionStub


@pytest.fixture
def chip_stub_cls():
    return ChipStub


@pytest.fixture
def chip_stub():
    """A single ready-to-use ChipStub instance (animal_id=1)."""
    return ChipStub(animal_id=1)


@pytest.fixture
def ibex_chip_stub_factory():
    """Duck-typed stand-in for an IbexChip: only needs `.file.name`."""

    def _make(name="x.png"):
        return SimpleNamespace(file=SimpleNamespace(name=name))

    return _make


# ---------------------------------------------------------------------------
# Django model factories (plain functions, no factory_boy per KISS).
# ---------------------------------------------------------------------------
@pytest.fixture
def user_factory(db):
    def _make(username="testuser", password="test-pass-12345", **overrides):
        user_model = get_user_model()
        defaults = {"username": username, "email": f"{username}@example.invalid"}
        defaults.update(overrides)
        return user_model.objects.create_user(password=password, **defaults)

    return _make


@pytest.fixture
def animal_factory(db):
    def _make(**overrides):
        defaults = {"id_code": "PNGP24_001"}
        defaults.update(overrides)
        return Animal.objects.create(**defaults)

    return _make


@pytest.fixture
def region_factory(db):
    def _make(owner=None, **overrides):
        defaults = {
            "name": "Test Region",
            "origin_latitude": 46.0,
            "origin_longitude": 8.0,
            "radius": 2000,
            "owner": owner,
        }
        defaults.update(overrides)
        return Region.objects.create(**defaults)

    return _make


@pytest.fixture
def folder_fixture(db):
    """django-filer Folder factory. Note: `user_factory` already
    auto-creates a `<username>_files` folder via the `create_user_folders`
    post_save signal -- use this fixture for extra/nested folders or when a
    test needs to delete the auto-created folder to simulate its absence.
    """

    def _make(name, owner=None, parent=None):
        return Folder.objects.create(name=name, owner=owner, parent=parent)

    return _make


# ---------------------------------------------------------------------------
# Image bytes / uploaded files.
# ---------------------------------------------------------------------------
@pytest.fixture
def tiny_png_bytes():
    """4x3 pure-red PNG, generated in-memory (no committed fixture file)."""
    img = Image.new("RGB", (4, 3), color=(255, 0, 0))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def tiny_jpeg_bytes():
    """4x3 pure-red JPEG, generated in-memory (no committed fixture file)."""
    img = Image.new("RGB", (4, 3), color=(255, 0, 0))
    buf = BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def tiny_png(tiny_png_bytes):
    return SimpleUploadedFile("tiny.png", tiny_png_bytes, content_type="image/png")


@pytest.fixture
def tiny_jpeg(tiny_jpeg_bytes):
    return SimpleUploadedFile("tiny.jpeg", tiny_jpeg_bytes, content_type="image/jpeg")


@pytest.fixture
def corrupted_image():
    """Empty/truncated bytes -- cv2.imread cannot decode this (T40)."""
    return SimpleUploadedFile("corrupted.png", b"not a real image", content_type="image/png")


@pytest.fixture
def ibex_image_factory(db, user_factory, tiny_png_bytes):
    def _make(owner=None, side=None, name="upload", **overrides):
        if owner is None:
            owner = user_factory()
        folder = Folder.objects.filter(name=f"{owner.username}_files", owner=owner).first()
        upload = SimpleUploadedFile(f"{name}.png", tiny_png_bytes, content_type="image/png")
        defaults = {
            "original_filename": upload.name,
            "file": upload,
            "folder": folder,
            "owner": owner,
            "side": side,
        }
        defaults.update(overrides)
        return IbexImage.objects.create(**defaults)

    return _make


# ---------------------------------------------------------------------------
# I/O boundary mocks.
# ---------------------------------------------------------------------------
@pytest.fixture
def mock_runpod(no_network):
    """Reconfigure the (already-patched, autouse) requests.post mock.

    Reuses the same patch object installed by the root `no_network` fixture
    instead of double-patching `requests.post`.
    """
    post_patch, _resource_patch = no_network
    post_patch.side_effect = None
    post_patch.return_value = mock.Mock()
    return post_patch


@pytest.fixture
def mock_b2():
    """Patch core.b2_utils.download_file (the function embed_new_chip calls),
    independent of the low-level boto3.resource guard.
    """
    with mock.patch("core.b2_utils.download_file") as download_mock:
        yield download_mock


@pytest.fixture
def moto_b2():
    """In-process S3 mock for core.b2_utils, distinct from `mock_b2` above.

    `mock_b2` stubs out core.b2_utils.download_file at the Python-function
    boundary. This fixture instead starts moto's `mock_s3()` (moto 4.2.14 --
    NOT `mock_aws`, which is 5.x-only and unavailable at this pin), which
    intercepts at the botocore/urllib3 HTTP layer, so the real
    get_b2_resource/download_file/delete_files/check_file_exists code paths
    in core.b2_utils.py run against a real (fake) S3 backend.

    Consumers MUST also carry `@pytest.mark.moto_s3` -- see
    conftest.py::no_network, which skips its boto3.resource-blocking patch
    only for tests bearing that marker.

    Bucket creation is pinned to `us-east-1` specifically (independent of
    whatever AWS_DEFAULT_REGION the app's own resource calls use) to avoid
    needing a CreateBucketConfiguration/LocationConstraint on create_bucket.

    Yields a boto3 S3 *client* (not a resource) -- the most directly useful
    contract for tests that call put_object/head_bucket/head_object/etc.
    against the mock directly.
    """
    from moto import (
        mock_s3,  # lazy import -- see test_b2_utils_moto.py module docstring
    )

    with mock_s3():
        setup_resource = boto3.resource("s3", region_name="us-east-1")
        setup_resource.create_bucket(Bucket="test-bucket")
        yield boto3.client("s3", region_name="us-east-1")
