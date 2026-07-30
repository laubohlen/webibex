"""T07-T16: `core.utils.process_horn_chip` scenarios.

Covers the local-vs-cloud storage branch, pre-existing-chip handling
(local file-and-row present, file-without-row, row-without-file), the
side="L"/"R"/"O" flip branch, and the cloud decode-failure paths
(undecodable bytes, `None` download result).

All bugs surfaced while writing this suite are pinned via
`pytest.raises(...)`/negative mock assertions, not fixed -- see
`docs/security-remediation-plan.md` for the corresponding TODO entries:
- the `>=1000` orphaned-B2-file risk (T13's `delete_files` never called),
- the file/row desync producing either `Http404` or `IntegrityError`
  depending on which side is missing (T10/T11/T14),
- the missing `None`-guard on `b2_utils.download_file` vs. `embed_new_chip`'s
  guard for the same call (T16).

Critical gotchas (see the coverage push's test spec for full derivation):
- `IbexImage` MUST be created INSIDE the `override_settings(MEDIA_ROOT=...)`
  block, or the upload lands in the real `media/` directory.
- `user_factory()` + `ibex_image_factory()` + `ibex_chip_factory()` in the
  same test MUST share one `owner=` -- otherwise a second `user_factory()`
  call collides on the unique `username` default.
- The cloud branch (`POSTGRES_LOCALLY=True` override) still writes the
  *image* to local test media (STORAGES backend is frozen at Django
  settings-import time) -- same pattern as
  `tests/core/test_utils_io.py::test_embed_new_chip_cloud_storage_and_cloud_endpoint_branch`,
  used here as the template for the cloud-branch override pattern.
- Tests that assert DB state AFTER an `IntegrityError` use
  `@pytest.mark.django_db(transaction=True)` (not the plain marker) --
  otherwise the broken atomic block raises `TransactionManagementError` on
  the follow-up query instead of the state actually being checked.
"""

from pathlib import Path
from unittest import mock

import cv2
import pytest
from django.db.utils import IntegrityError
from django.http import Http404
from django.test import override_settings

from core.models import IbexChip
from core.utils import get_chip_filename, mirror_coordinate, process_horn_chip


def _local_chip_path(img):
    """Reproduce core/utils.py:358-360's local chip-path computation without
    hardcoding today's date (the filer upload path is date-partitioned)."""
    return Path(img.file.path).parent / get_chip_filename(img.file.name, "png")


# T07 -----------------------------------------------------------------------
@pytest.mark.django_db
def test_process_horn_chip_local_happy_path(tmp_path, user_factory, ibex_image_factory):
    with override_settings(MEDIA_ROOT=str(tmp_path)):
        owner = user_factory()
        img = ibex_image_factory(owner=owner, side="L")
        with mock.patch("core.utils.embed_new_chip") as embed_mock:
            process_horn_chip(img, x_horn=2, y_horn=1, x_eye=1, y_eye=2)
        chip = IbexChip.objects.get(ibex_image=img)
        assert chip.file.name.endswith("_chip.png")
        assert Path(chip.file.path).is_file()
        assert embed_mock.call_count == 1
        assert embed_mock.call_args.args[0].pk == chip.pk


# T08 -------------------------------------------------------------------
@pytest.mark.django_db
def test_process_horn_chip_local_replaces_existing_chip(
    tmp_path, user_factory, ibex_image_factory, ibex_chip_factory, tiny_png_bytes
):
    with override_settings(MEDIA_ROOT=str(tmp_path)):
        owner = user_factory()
        img = ibex_image_factory(owner=owner, side="L")
        old = ibex_chip_factory(owner=owner, ibex_image=img)
        # write a real file at the computed chip_path so the is_file() guard is True
        chip_path = _local_chip_path(img)
        chip_path.write_bytes(tiny_png_bytes)
        with mock.patch("core.utils.embed_new_chip") as embed_mock:
            process_horn_chip(img, x_horn=2, y_horn=1, x_eye=1, y_eye=2)
        assert not IbexChip.objects.filter(pk=old.pk).exists()
        assert IbexChip.objects.count() == 1
        new_chip = IbexChip.objects.get(ibex_image=img)
        assert new_chip.pk != old.pk
        assert embed_mock.call_args.args[0].pk == new_chip.pk


# T09 -------------------------------------------------------------------
@pytest.mark.django_db
@pytest.mark.parametrize("side,expected_mirror_calls", [("L", 0), ("R", 2), ("O", 0)])
def test_process_horn_chip_side_flip(
    tmp_path, user_factory, ibex_image_factory, side, expected_mirror_calls
):
    with override_settings(MEDIA_ROOT=str(tmp_path / side)):
        owner = user_factory(username=f"owner_{side}")
        img = ibex_image_factory(owner=owner, side=side, name=f"img_{side}")
        with (
            mock.patch("core.utils.embed_new_chip"),
            mock.patch(
                "core.utils.mirror_coordinate", wraps=mirror_coordinate
            ) as mirror_spy,
        ):
            process_horn_chip(img, x_horn=3, y_horn=0, x_eye=1, y_eye=2)
        assert mirror_spy.call_count == expected_mirror_calls
        if side == "R":
            # image.width == 4 per tiny_png_bytes fixture (4x3)
            call_args_list = [c.args for c in mirror_spy.call_args_list]
            assert (1, 4) in call_args_list
            assert (3, 4) in call_args_list


# T10 ---------------------------------------------------------------------
@pytest.mark.django_db
def test_process_horn_chip_local_file_without_row_raises_404(
    tmp_path, user_factory, ibex_image_factory, tiny_png_bytes
):
    with override_settings(MEDIA_ROOT=str(tmp_path)):
        owner = user_factory()
        img = ibex_image_factory(owner=owner, side="L")
        chip_path = _local_chip_path(img)
        chip_path.write_bytes(tiny_png_bytes)
        with pytest.raises(Http404):
            process_horn_chip(img, x_horn=2, y_horn=1, x_eye=1, y_eye=2)
        assert IbexChip.objects.count() == 0


# T11 -------------------------------------------------------------------
@pytest.mark.django_db(transaction=True)
def test_process_horn_chip_local_row_without_file_raises_integrity_error(
    tmp_path, user_factory, ibex_image_factory, ibex_chip_factory
):
    with override_settings(MEDIA_ROOT=str(tmp_path)):
        owner = user_factory()
        img = ibex_image_factory(owner=owner, side="L")
        old = ibex_chip_factory(owner=owner, ibex_image=img)
        # do NOT write a file at chip_path -- the is_file() guard stays False,
        # so process_horn_chip skips straight to shutil.copy2() on a chip
        # path whose parent directory exists but sibling file doesn't; the
        # OneToOne `ibex_image` constraint is what actually fails, below.
        with pytest.raises(IntegrityError, match="ibex_image_id"):
            process_horn_chip(img, x_horn=2, y_horn=1, x_eye=1, y_eye=2)
        assert IbexChip.objects.filter(pk=old.pk).exists()  # old row was never deleted


# T12 -------------------------------------------------------------------
@pytest.mark.django_db
def test_process_horn_chip_cloud_no_existing_chip(
    tmp_path, user_factory, ibex_image_factory, tiny_png_bytes
):
    with override_settings(MEDIA_ROOT=str(tmp_path)):
        owner = user_factory()
        img = ibex_image_factory(owner=owner, side="L")
        with (
            mock.patch("core.utils.embed_new_chip") as embed_mock,
            mock.patch(
                "core.b2_utils.download_file", return_value=tiny_png_bytes
            ) as download_mock,
            mock.patch("core.b2_utils.check_file_exists") as exists_mock,
            mock.patch("core.b2_utils.delete_files") as delete_mock,
            override_settings(POSTGRES_LOCALLY=True, AWS_LOCATION="media"),
        ):
            process_horn_chip(img, x_horn=2, y_horn=1, x_eye=1, y_eye=2)
        assert download_mock.call_count == 1
        assert exists_mock.call_count == 0
        assert delete_mock.call_count == 0
        chip = IbexChip.objects.get(ibex_image=img)
        assert chip.file.name.endswith("_chip.png")
        assert embed_mock.call_args.args[0].pk == chip.pk


# T13 -------------------------------------------------------------------
@pytest.mark.django_db
def test_process_horn_chip_cloud_replaces_existing_chip_no_b2_delete(
    tmp_path, user_factory, ibex_image_factory, ibex_chip_factory, tiny_png_bytes
):
    """Pins the orphaned-B2-file bug: the old `IbexChip` DB row is deleted,
    but `b2_utils.delete_files` is never called (its call site is commented
    out in core/utils.py) -- the old chip file is left behind on Backblaze."""
    with override_settings(MEDIA_ROOT=str(tmp_path)):
        owner = user_factory()
        img = ibex_image_factory(owner=owner, side="L")
        old = ibex_chip_factory(owner=owner, ibex_image=img)
        with (
            mock.patch("core.utils.embed_new_chip") as embed_mock,
            mock.patch("core.b2_utils.download_file", return_value=tiny_png_bytes),
            mock.patch(
                "core.b2_utils.check_file_exists", return_value=True
            ) as exists_mock,
            mock.patch("core.b2_utils.delete_files") as delete_mock,
            override_settings(POSTGRES_LOCALLY=True, AWS_LOCATION="media"),
        ):
            process_horn_chip(img, x_horn=2, y_horn=1, x_eye=1, y_eye=2)
        assert exists_mock.call_count == 1
        assert not IbexChip.objects.filter(pk=old.pk).exists()
        assert IbexChip.objects.count() == 1
        delete_mock.assert_not_called()  # THE PIN -- do not remove
        assert embed_mock.call_count == 1


# T14 -------------------------------------------------------------------
@pytest.mark.django_db(transaction=True)
def test_process_horn_chip_cloud_check_file_exists_false_raises_integrity_error(
    tmp_path, user_factory, ibex_image_factory, ibex_chip_factory, tiny_png_bytes
):
    """When check_file_exists() returns False, the old DB row is left in
    place (not "not deleted because bad" -- the code path never reaches
    the delete call), and the unconditional `IbexChip.objects.create(...)`
    a few lines later collides with it on the OneToOne `ibex_image`
    constraint -- the call does not complete successfully."""
    with override_settings(MEDIA_ROOT=str(tmp_path)):
        owner = user_factory()
        img = ibex_image_factory(owner=owner, side="L")
        old = ibex_chip_factory(owner=owner, ibex_image=img)
        with (
            mock.patch("core.utils.embed_new_chip") as embed_mock,
            mock.patch("core.b2_utils.download_file", return_value=tiny_png_bytes),
            mock.patch("core.b2_utils.check_file_exists", return_value=False),
            mock.patch("core.b2_utils.delete_files") as delete_mock,
            override_settings(POSTGRES_LOCALLY=True, AWS_LOCATION="media"),
            pytest.raises(IntegrityError, match="ibex_image_id"),
        ):
            process_horn_chip(img, x_horn=2, y_horn=1, x_eye=1, y_eye=2)
        assert IbexChip.objects.filter(pk=old.pk).exists()  # row survives
        delete_mock.assert_not_called()
        embed_mock.assert_not_called()


# T15 -------------------------------------------------------------------
@pytest.mark.django_db
def test_process_horn_chip_cloud_undecodable_bytes_raises_value_error(
    tmp_path, user_factory, ibex_image_factory
):
    with override_settings(MEDIA_ROOT=str(tmp_path)):
        owner = user_factory()
        img = ibex_image_factory(owner=owner, side="L")
        with (
            mock.patch("core.utils.embed_new_chip") as embed_mock,
            mock.patch("core.b2_utils.download_file", return_value=b"not an image"),
            override_settings(POSTGRES_LOCALLY=True, AWS_LOCATION="media"),
            pytest.raises(ValueError, match="Failed to load image from cloud"),
        ):
            process_horn_chip(img, x_horn=2, y_horn=1, x_eye=1, y_eye=2)
        assert IbexChip.objects.count() == 0
        embed_mock.assert_not_called()


# T15 counter-input -------------------------------------------------------
@pytest.mark.django_db
def test_process_horn_chip_cloud_empty_bytes_raises_cv2_error(
    tmp_path, user_factory, ibex_image_factory
):
    """Counter-input to T15: `b""` does NOT hit the same `ValueError` branch.
    `np.frombuffer(b"", np.uint8)` yields an empty buffer, and
    `cv2.imdecode` raises a raw, uncaught `cv2.error` (OpenCV assertion
    `!buf.empty()`) before the `img is None` check is ever reached --
    pinned as-is, not fixed, not the same failure mode as non-empty
    undecodable bytes."""
    with override_settings(MEDIA_ROOT=str(tmp_path)):
        owner = user_factory()
        img = ibex_image_factory(owner=owner, side="L")
        with (
            mock.patch("core.utils.embed_new_chip") as embed_mock,
            mock.patch("core.b2_utils.download_file", return_value=b""),
            override_settings(POSTGRES_LOCALLY=True, AWS_LOCATION="media"),
            pytest.raises(cv2.error),
        ):
            process_horn_chip(img, x_horn=2, y_horn=1, x_eye=1, y_eye=2)
        assert IbexChip.objects.count() == 0
        embed_mock.assert_not_called()


# T16 -------------------------------------------------------------------
@pytest.mark.django_db
def test_process_horn_chip_cloud_download_returns_none_raises_type_error(
    tmp_path, user_factory, ibex_image_factory
):
    """Empirically found (not in the original bugs list): unlike
    `embed_new_chip`, which explicitly guards `img_object is None` before
    calling `np.frombuffer`, `process_horn_chip`'s cloud branch has no such
    guard -- `np.frombuffer(None, np.uint8)` raises a raw `TypeError`."""
    with override_settings(MEDIA_ROOT=str(tmp_path)):
        owner = user_factory()
        img = ibex_image_factory(owner=owner, side="L")
        with (
            mock.patch("core.utils.embed_new_chip") as embed_mock,
            mock.patch("core.b2_utils.download_file", return_value=None),
            override_settings(POSTGRES_LOCALLY=True, AWS_LOCATION="media"),
            pytest.raises(TypeError, match="bytes-like object"),
        ):
            process_horn_chip(img, x_horn=2, y_horn=1, x_eye=1, y_eye=2)
        embed_mock.assert_not_called()
