"""T26-T33: core/utils.py functions that touch the database or a request."""

import pytest
from django.test import RequestFactory, override_settings

from core.models import Location, Region
from core.utils import generate_animal_id_code, get_task_request_origin, multi_task_url


# T26 ------------------------------------------------------------------
@pytest.mark.django_db
def test_generate_animal_id_code_first_new_animal():
    result = generate_animal_id_code("PNGP24_---_24_06_15_174811.jpg")
    assert result == "PNGP24_001"


# T27 --------------------------------------------------------------------
@pytest.mark.django_db
def test_generate_animal_id_code_increments_from_max(animal_factory):
    animal_factory(id_code="PNGP24_003")
    animal_factory(id_code="PNGP24_012")

    result = generate_animal_id_code("PNGP24_---_24_06_15_174811.jpg")

    assert result == "PNGP24_013"


# T28 --------------------------------------------------------------------
@pytest.mark.django_db
@pytest.mark.parametrize(
    "malformed_id_code",
    [
        "PNGP24_ab",  # letters only, no digit run
        "PNGP24_",  # nothing after underscore
        "PNGP24_12",  # 2 digits, below the 3-digit pattern
    ],
)
def test_generate_animal_id_code_malformed_existing_code_falls_back_to_001(
    animal_factory, malformed_id_code
):
    """Bug B3 (fixed): a malformed existing id_code (contains "_" but no
    3-digit run for re.findall to match) is filtered out instead of
    crashing, and generation falls back to "{prefix}_001"."""
    animal_factory(id_code=malformed_id_code)

    result = generate_animal_id_code("PNGP24_---_24_06_15_174811.jpg")

    assert result == "PNGP24_001"


# S6 ------------------------------------------------------------------------
@pytest.mark.django_db
def test_generate_animal_id_code_mixed_valid_and_malformed_uses_max_of_valid(
    animal_factory,
):
    """Bug B3 (fixed): when a malformed id_code coexists with a valid one,
    the malformed entry is filtered out and the valid entry's number is
    used as the max (not a short-circuit to the fallback)."""
    animal_factory(id_code="PNGP24_ab")
    animal_factory(id_code="PNGP24_007")

    result = generate_animal_id_code("PNGP24_---_24_06_15_174811.jpg")

    assert result == "PNGP24_008"


# T29 -----------------------------------------------------------------------
@pytest.mark.django_db
def test_get_task_request_origin_valid_referer_returns_url_name():
    request = RequestFactory().get("/", HTTP_REFERER="http://testserver/unidentified/")

    result = get_task_request_origin(request)

    assert result == "unidentified-images"


# T30 -----------------------------------------------------------------
@pytest.mark.parametrize("http_referer", [None, ""])
def test_get_task_request_origin_no_referer_returns_none(http_referer):
    """Bug B1 (fixed): when request has no (or an empty/falsy) HTTP_REFERER,
    task_request_url_name is initialised to None instead of being left
    unassigned."""
    if http_referer is None:
        request = RequestFactory().get("/")
    else:
        request = RequestFactory().get("/", HTTP_REFERER=http_referer)

    result = get_task_request_origin(request)

    assert result is None


# T31 -----------------------------------------------------------------
@pytest.mark.django_db
def test_get_task_request_origin_unresolvable_path_returns_none():
    request = RequestFactory().get(
        "/", HTTP_REFERER="http://testserver/this-path-does-not-exist/"
    )

    result = get_task_request_origin(request)

    assert result is None


# T32 -------------------------------------------------------------------
def test_multi_task_url_view_branch():
    result = multi_task_url("view")
    assert result == ("core/multi_view.html", None)


@override_settings(LANDMARK_IMAGE_WIDTH=1600)
def test_multi_task_url_landmark_branch():
    template, context = multi_task_url("landmark")
    assert template == "simple_landmarks/multi_landmarking.html"
    assert context == {"display_width": 1600}


def test_multi_task_url_delete_branch_returns_none():
    assert multi_task_url("delete") is None


def test_multi_task_url_unknown_tool_returns_none():
    assert multi_task_url("not-a-real-tool") is None


# T33 -------------------------------------------------------------------------
@pytest.mark.django_db
def test_multi_task_url_locate_branch_with_gps(user_factory, region_factory):
    from types import SimpleNamespace

    owner = user_factory()
    region_factory(owner=owner, name="R1")
    location = Location.objects.create(latitude=46.0, longitude=8.0)
    image = SimpleNamespace(location=location)

    template, context = multi_task_url("locate", image=image, user=owner)

    assert template == "core/multi_location_create.html"
    assert context["location_id"] == location.id
    assert context["image_location"] == location
    assert list(context["regions"]) == list(Region.objects.filter(owner=owner))


@pytest.mark.django_db
def test_multi_task_url_locate_branch_without_gps_returns_none_location(user_factory):
    from types import SimpleNamespace

    owner = user_factory()
    location = Location.objects.create(latitude=None, longitude=None)
    image = SimpleNamespace(location=location)

    template, context = multi_task_url("locate", image=image, user=owner)

    assert template == "core/multi_location_create.html"
    assert context["image_location"] is None
    assert context["location_id"] == location.id
