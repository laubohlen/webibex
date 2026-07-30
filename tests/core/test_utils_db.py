"""T26-T33: core/utils.py functions that touch the database or a request."""

import pytest
from django.test import RequestFactory, override_settings

from core.models import Animal, Location, Region
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


# T17 ---------------------------------------------------------------------
@pytest.mark.django_db
def test_generate_animal_id_code_rollover_collision(animal_factory):
    """Bug (pinned, not fixed): `f"{id_number:03}"` formats numbers >= 1000
    without padding truncation, so both "PN24_999" -> 1000 and a pre-existing
    "PN24_1000" collide on the same rendered code -- a real duplicate
    id_code, not just a display artifact."""
    animal_factory(id_code="PN24_999")
    animal_factory(id_code="PN24_1000")

    result = generate_animal_id_code("PN24_---_24_06_15_174811.jpg")

    assert result == "PN24_1000"
    assert Animal.objects.filter(id_code=result).exists()  # proves the COLLISION


@pytest.mark.django_db
def test_generate_animal_id_code_no_collision_just_under_rollover(animal_factory):
    """Boundary control for T17: seeding only up to "PN24_999" (no existing
    "PN24_1000" row) means the new code is genuinely unique -- the collision
    above is specifically about the >=1000 rollover, not generation itself."""
    animal_factory(id_code="PN24_998")
    animal_factory(id_code="PN24_999")

    result = generate_animal_id_code("PN24_---_24_06_15_174811.jpg")

    assert result == "PN24_1000"


# T18/T19/T20 --------------------------------------------------------------
@pytest.mark.django_db
@pytest.mark.parametrize(
    "seed_id_code,filename,expected",
    [
        ("PN24_1000", "PN24_x.jpg", "PN24_101"),  # T18: rollover, simple form
        (
            "PN2024_001",
            "PN2024_---_24_06_15_174811.jpg",
            "PN2024_203",
        ),  # T19: first-3-digit-run misparse (matches "202" from the seeded
        # id_code's own "2024" prefix digits, before reaching "001")
        (
            "ZZZZ_050",
            "PN24_---_24_06_15_174811.jpg",
            "PN24_051",
        ),  # T20: no prefix scoping -- any existing "_"-containing id_code counts
    ],
)
def test_generate_animal_id_code_known_bugs(
    animal_factory, seed_id_code, filename, expected
):
    """Bugs (pinned, not fixed): the `re.findall(r"\\d{3}", id_code)` pattern
    grabs the FIRST 3-digit run in the whole id_code string -- not
    necessarily the trailing counter -- and `previous_generated_codes` is
    never scoped to the new filename's prefix, so any "_"-containing
    id_code in the table (regardless of location/year prefix) contributes
    to the max()."""
    animal_factory(id_code=seed_id_code)

    result = generate_animal_id_code(filename)

    assert result == expected


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
    # T06: regions are shared/unfiltered (not scoped to the calling user) --
    # see T01/T03/T05a below for the actual discriminating oracles; this
    # assertion alone would pass either way with only one Region row here.
    assert list(context["regions"]) == list(Region.objects.all())


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


# T01 -------------------------------------------------------------------
@pytest.mark.django_db
def test_multi_task_url_locate_branch_shows_region_owned_by_other_user(
    user_factory, region_factory
):
    from types import SimpleNamespace

    owner = user_factory(username="t01_owner")
    other = user_factory(username="t01_other")
    region = region_factory(owner=owner, name="T01Region")
    location = Location.objects.create(latitude=46.0, longitude=8.0)
    image = SimpleNamespace(location=location)

    _template, context = multi_task_url("locate", image=image, user=other)

    assert region in list(context["regions"])


# T03 -------------------------------------------------------------------
@pytest.mark.django_db
def test_multi_task_url_locate_branch_shows_all_regions_not_just_cross_owner(
    user_factory, region_factory
):
    """Discriminating counter: pre-fix only region_b (owned by the calling
    user) would show. region_a's presence kills a mutant that flips the
    filter direction (e.g. `owner != user`) instead of removing it."""
    from types import SimpleNamespace

    owner = user_factory(username="t03_owner")
    other = user_factory(username="t03_other")
    region_a = region_factory(owner=owner, name="T03RegionA")
    region_b = region_factory(owner=other, name="T03RegionB")
    location = Location.objects.create(latitude=46.0, longitude=8.0)
    image = SimpleNamespace(location=location)

    _template, context = multi_task_url("locate", image=image, user=other)

    regions = list(context["regions"])
    assert region_a in regions
    assert region_b in regions


# T05a --------------------------------------------------------------------
@pytest.mark.django_db
def test_multi_task_url_locate_branch_shows_orphaned_region_with_no_owner(
    user_factory, region_factory
):
    """Kills a naive partial fix like
    `filter(owner=user) | filter(owner__isnull=True)` -- only a true
    `Region.objects.all()` passes this."""
    from types import SimpleNamespace

    owner = user_factory(username="t05a_owner")
    orphan_region = region_factory(owner=None, name="T05A_ORPHAN")
    location = Location.objects.create(latitude=46.0, longitude=8.0)
    image = SimpleNamespace(location=location)

    _template, context = multi_task_url("locate", image=image, user=owner)

    assert orphan_region in list(context["regions"])
