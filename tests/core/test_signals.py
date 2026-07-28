"""Tests for core/signals.py -- pure-function DMS/GPS conversion, the
allauth user_signed_up callback, IbexImage post_save processing (filename,
side tagging, location/GPS, created_at), landmark-item lifecycle,
IbexChip file cleanup, and the animal-change folder/rename signal.

Scope: TEST-ONLY. core/signals.py is not modified.

Four pre-existing findings (documented in docs/security-remediation-plan.md,
2026-07-28) are asserted verbatim here rather than fixed:

1. `create_folder_for_animal_on_change` raises `UnboundLocalError` when
   `instance.side` is outside {"L", "R", "O"} (`target_folder` referenced
   before assignment) -- see T23.
2. `get_decimal_from_dms` raises an uncaught `TypeError` (not `None`) on
   malformed-but-indexable DMS input -- the arithmetic that consumes
   `to_float()`'s results sits outside the try/except that wraps the
   `to_float` calls -- see T05/T06.
3. `signals.py:192-193` -- dead/unreachable `else` branch inside
   `process_uploaded_image` (`dt_object` is always a `datetime`, or
   `strptime` already raised). Left uncovered intentionally.
4. `signals.py:271-273` -- dead/latent `except User.DoesNotExist` inside
   `create_folder_for_animal_on_change` (cannot fire for a `None` owner --
   `user.username` on the next line raises `AttributeError` instead). Left
   uncovered intentionally.
"""

from datetime import datetime
from types import SimpleNamespace

import pytest
from allauth.account.signals import user_signed_up
from django.contrib.auth.models import Group
from django.contrib.contenttypes.models import ContentType
from django.db.models.fields.files import FieldFile
from django.utils import timezone
from filer.models import Folder

from core.models import IbexImage, Location
from core.signals import extract_gps_coords, get_decimal_from_dms
from simple_landmarks.models import LandmarkItem

# ---------------------------------------------------------------------------
# T01-T06 -- get_decimal_from_dms (pure function, no DB)
# ---------------------------------------------------------------------------


def test_get_decimal_from_dms_with_float_tuple_and_north_ref_returns_positive():
    result = get_decimal_from_dms((46.0, 30.0, 0.0), "N")

    assert result == pytest.approx(46.5)


def test_get_decimal_from_dms_with_rational_tuples_returns_decimal():
    result = get_decimal_from_dms(((46, 1), (30, 1), (0, 1)), "N")

    assert result == pytest.approx(46.5)


@pytest.mark.parametrize(
    "ref, expected_sign",
    [
        ("N", 1),
        ("S", -1),
        ("E", 1),
        ("W", -1),
    ],
)
def test_get_decimal_from_dms_ref_determines_sign(ref, expected_sign):
    result = get_decimal_from_dms((46, 30, 0), ref)

    assert result == pytest.approx(expected_sign * 46.5)


def test_get_decimal_from_dms_with_short_tuple_returns_none():
    result = get_decimal_from_dms((46.0, 30.0), "N")

    assert result is None


def test_get_decimal_from_dms_with_malformed_seconds_raises_type_error():
    # Finding 2: to_float("abc") swallows the ValueError and returns None,
    # but the downstream `seconds / 3600.0` arithmetic sits outside the
    # try/except that guards to_float -- documents the current TypeError,
    # not a fixed None-safe behavior.
    with pytest.raises(TypeError):
        get_decimal_from_dms((46.0, 30.0, "abc"), "N")


def test_get_decimal_from_dms_with_malformed_rational_degrees_raises_type_error():
    # Finding 2 variant: the tuple-division except swallows the error for
    # `degrees`, but the downstream `degrees + ...` arithmetic is unguarded.
    with pytest.raises(TypeError):
        get_decimal_from_dms(((1, "x"), 30.0, 0.0), "N")


# ---------------------------------------------------------------------------
# T07-T10 -- extract_gps_coords (duck-typed stub, no DB)
# ---------------------------------------------------------------------------


def test_extract_gps_coords_with_well_formed_gpsinfo_returns_coords():
    filer_image = SimpleNamespace(
        exif={
            "GPSInfo": {
                1: "N",  # GPSLatitudeRef
                2: (46, 30, 0),  # GPSLatitude
                3: "E",  # GPSLongitudeRef
                4: (8, 0, 0),  # GPSLongitude
            }
        }
    )

    lat, lng = extract_gps_coords(filer_image)

    assert lat == pytest.approx(46.5)
    assert lng == pytest.approx(8.0)


def test_extract_gps_coords_with_no_gpsinfo_returns_none_none():
    filer_image = SimpleNamespace(exif={})

    lat, lng = extract_gps_coords(filer_image)

    assert (lat, lng) == (None, None)


def test_extract_gps_coords_with_missing_required_key_returns_none_none():
    filer_image = SimpleNamespace(
        exif={
            "GPSInfo": {
                1: "N",  # GPSLatitudeRef
                2: (46, 30, 0),  # GPSLatitude
                # 3 (GPSLongitudeRef) intentionally omitted
                4: (8, 0, 0),  # GPSLongitude
            }
        }
    )

    lat, lng = extract_gps_coords(filer_image)

    assert (lat, lng) == (None, None)


def test_extract_gps_coords_with_malformed_dms_swallows_error_returns_none_none():
    filer_image = SimpleNamespace(
        exif={
            "GPSInfo": {
                1: "N",  # GPSLatitudeRef
                2: (1, 2, "x"),  # GPSLatitude -- malformed seconds
                3: "E",  # GPSLongitudeRef
                4: (8, 0, 0),  # GPSLongitude
            }
        }
    )

    lat, lng = extract_gps_coords(filer_image)

    assert (lat, lng) == (None, None)


# ---------------------------------------------------------------------------
# T11 -- user_signed_up_callback (DB, real allauth signal)
# ---------------------------------------------------------------------------


@pytest.mark.django_db
def test_user_signed_up_callback_adds_user_to_public_users_group(user_factory):
    user = user_factory()

    user_signed_up.send(sender=user.__class__, request=None, user=user)

    assert Group.objects.filter(name="public_users").exists()
    assert user.groups.filter(name="public_users").exists()


# ---------------------------------------------------------------------------
# T12-T15 -- process_uploaded_image (DB, via extended ibex_image_factory)
# ---------------------------------------------------------------------------


@pytest.mark.django_db
def test_process_uploaded_image_with_exif_datetime_sets_name_and_created_at(
    ibex_image_factory,
):
    image = ibex_image_factory(exif={"DateTime": "2024:01:15 10:30:00"})

    persisted = IbexImage.objects.get(pk=image.pk)
    assert "24_01_15_103000" in persisted.name
    assert persisted.created_at == timezone.make_aware(datetime(2024, 1, 15, 10, 30, 0))


@pytest.mark.django_db
def test_process_uploaded_image_with_truthy_exif_no_datetime_uses_noexifdata(
    ibex_image_factory,
):
    image = ibex_image_factory(exif={"Orientation": 1})

    persisted = IbexImage.objects.get(pk=image.pk)
    assert "noexifdata" in persisted.name


@pytest.mark.django_db
@pytest.mark.parametrize(
    "folder_name, expected_side",
    [
        ("_left_upload", "L"),
        ("_right_upload", "R"),
        ("_other_upload", "O"),
    ],
)
def test_process_uploaded_image_tags_side_from_parent_folder(
    folder_name, expected_side, ibex_image_factory, user_factory
):
    owner = user_factory()
    # Auto-created by the existing create_user_folders signal when
    # user_factory() runs -- fail loudly if the lookup misses.
    folder = Folder.objects.get(name=folder_name, owner=owner)

    image = ibex_image_factory(owner=owner, folder=folder)

    persisted = IbexImage.objects.get(pk=image.pk)
    assert persisted.side == expected_side


@pytest.mark.django_db
def test_process_uploaded_image_with_gps_exif_populates_location(ibex_image_factory):
    exif = {
        "GPSInfo": {
            1: "N",  # GPSLatitudeRef
            2: (46, 30, 0),  # GPSLatitude
            3: "E",  # GPSLongitudeRef
            4: (8, 0, 0),  # GPSLongitude
        }
    }

    image = ibex_image_factory(exif=exif)

    persisted = IbexImage.objects.get(pk=image.pk)
    assert persisted.location is not None
    assert persisted.location.latitude == pytest.approx(46.5)
    assert persisted.location.longitude == pytest.approx(8.0)


# ---------------------------------------------------------------------------
# T16-T18 -- landmark items and location cleanup on IbexImage lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.django_db
def test_initialise_landmark_items_creates_one_item_per_landmark(
    landmark_factory, ibex_image_factory
):
    landmark_factory()
    landmark_factory()

    image = ibex_image_factory()

    items = LandmarkItem.objects.filter(
        object_id=image.id, content_type=ContentType.objects.get_for_model(IbexImage)
    )
    assert items.count() == 2
    for item in items:
        assert item.x_coordinate == 0
        assert item.y_coordinate == 0


@pytest.mark.django_db
def test_delete_landmark_items_removes_items_on_image_delete(
    landmark_factory, ibex_image_factory
):
    landmark_factory()
    landmark_factory()
    image = ibex_image_factory()
    image_id = image.id
    content_type = ContentType.objects.get_for_model(IbexImage)
    items = LandmarkItem.objects.filter(object_id=image_id, content_type=content_type)
    assert items.count() == 2

    image.delete()

    assert items.count() == 0


@pytest.mark.django_db
def test_delete_associated_location_removes_location_on_image_delete(
    ibex_image_factory,
):
    image = ibex_image_factory()
    loc_pk = image.location_id
    assert loc_pk is not None

    image.delete()

    assert Location.objects.filter(pk=loc_pk).exists() is False


# ---------------------------------------------------------------------------
# T19 -- delete_ibexchip_file (DB + monkeypatch spy)
# ---------------------------------------------------------------------------


@pytest.mark.django_db
def test_delete_ibexchip_file_calls_field_file_delete_with_save_false(
    ibex_chip_factory, monkeypatch
):
    # Spy delegates to the original FieldFile.delete rather than replacing
    # it outright: easy_thumbnails' ThumbnailerFieldFile.delete() relies on
    # the real base-class delete() clearing `self.name` to short-circuit a
    # second, filer-internal cleanup call (filer/models/filemodels.py:306).
    # A non-delegating stub leaves `self.name` set, which crashes that
    # second call on an already-deleted Source cache row -- an artifact of
    # the spy, not a core/signals.py bug.
    chip = ibex_chip_factory()
    calls = []
    original_delete = FieldFile.delete

    def spy_delete(self, *args, **kwargs):
        calls.append({"args": args, "kwargs": kwargs})
        return original_delete(self, *args, **kwargs)

    monkeypatch.setattr(FieldFile, "delete", spy_delete)

    chip.delete()

    # core/signals.py:247 calls `instance.file.delete(save=False)` as a
    # keyword argument -- exactly one such call should occur. (filer's own
    # internal follow-up call at filemodels.py:306 uses a positional
    # argument and is not part of the signal under test.)
    save_false_kwarg_calls = [c for c in calls if c["kwargs"] == {"save": False}]
    assert len(save_false_kwarg_calls) == 1


# ---------------------------------------------------------------------------
# T20-T23 -- create_folder_for_animal_on_change (DB, full branch matrix)
# ---------------------------------------------------------------------------


@pytest.mark.django_db
def test_create_folder_for_animal_on_change_with_left_side_creates_folders(
    ibex_image_factory, animal_factory
):
    image = ibex_image_factory(side="L")
    animal = animal_factory(id_code="A1")

    image.animal = animal
    image.save()

    assert Folder.objects.filter(name="A1", owner=image.owner).exists()
    assert Folder.objects.filter(name="left_A1", owner=image.owner).exists()
    persisted = IbexImage.objects.get(pk=image.pk)
    assert persisted.folder.name == "left_A1"
    assert persisted.name.startswith("A1_")


@pytest.mark.django_db
@pytest.mark.parametrize(
    "side, expected_folder_name",
    [
        ("R", "right_A1"),
        ("O", "other_A1"),
    ],
)
def test_create_folder_for_animal_on_change_with_right_or_other_side(
    side, expected_folder_name, ibex_image_factory, animal_factory
):
    image = ibex_image_factory(side=side)
    animal = animal_factory(id_code="A1")

    image.animal = animal
    image.save()

    persisted = IbexImage.objects.get(pk=image.pk)
    assert persisted.folder.name == expected_folder_name


@pytest.mark.django_db
def test_create_folder_for_animal_on_change_with_short_name_uses_second_part(
    ibex_image_factory, animal_factory
):
    image = ibex_image_factory(side="L")
    animal = animal_factory(id_code="A1")
    image.name = "ab_cd"
    image.save()

    image.animal = animal
    image.save()

    persisted = IbexImage.objects.get(pk=image.pk)
    assert persisted.name == "A1_cd"


@pytest.mark.django_db
def test_create_folder_for_animal_on_change_invalid_side_raises_unbound_local_error(
    ibex_image_factory, animal_factory
):
    # Finding 1: target_folder is only assigned for side in {"L", "R", "O"};
    # any other value (including None) falls through to the bare `else:
    # pass`, and the later `instance.folder = target_folder` raises
    # UnboundLocalError. Documents the current crash, does not fix it.
    image = ibex_image_factory(side=None)
    animal = animal_factory(id_code="A1")

    with pytest.raises(UnboundLocalError):
        image.animal = animal
        image.save()
