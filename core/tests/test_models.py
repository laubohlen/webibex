"""T23-T25: core/models.py __str__ methods and constraints."""

import pytest
from django.db import IntegrityError, transaction

from core.models import Location, Region
from simple_landmarks.models import Landmark


# T23 -------------------------------------------------------------------
@pytest.mark.django_db
def test_region_str_returns_name(region_factory):
    region = region_factory(name="Alps")
    assert str(region) == "Alps"


# S5 (Region twin of B4) -----------------------------------------------------
@pytest.mark.django_db
def test_region_str_none_name_returns_fallback(region_factory):
    """Region twin of Bug B4 (fixed): __str__ returns a bracket-sentinel
    fallback instead of raising TypeError when name field is None."""
    region = region_factory(name=None)
    assert str(region) == "[No Name]"


@pytest.mark.django_db
def test_landmark_str_returns_label():
    landmark = Landmark.objects.create(label="horn_tip")
    assert str(landmark) == "horn_tip"


@pytest.mark.django_db
def test_location_str_fallback_when_no_ibeximage():
    location = Location.objects.create()
    assert str(location) == "Location for: [No IbexImage]"


# T24 ---------------------------------------------------------------------
@pytest.mark.django_db
def test_animal_str_returns_id_code(animal_factory):
    animal = animal_factory(id_code="PNGP24_001")
    assert str(animal) == "PNGP24_001"


@pytest.mark.django_db
def test_animal_str_none_id_code_returns_fallback(animal_factory):
    """Bug B4 (fixed): __str__ returns a bracket-sentinel fallback instead
    of raising TypeError when id_code field is None."""
    animal = animal_factory(id_code=None)
    assert str(animal) == "[No ID Code]"


# T25 -----------------------------------------------------------------------
@pytest.mark.django_db
def test_region_unique_name_per_owner_duplicate_raises_integrity_error(
    user_factory, region_factory
):
    owner = user_factory(username="owner1")
    region_factory(owner=owner, name="X")

    with pytest.raises(IntegrityError):
        with transaction.atomic():
            region_factory(owner=owner, name="X")


@pytest.mark.django_db
def test_region_same_name_different_owner_succeeds(user_factory, region_factory):
    owner_a = user_factory(username="owner_a")
    owner_b = user_factory(username="owner_b")
    region_factory(owner=owner_a, name="X")

    region_b = region_factory(owner=owner_b, name="X")

    assert region_b.pk is not None
