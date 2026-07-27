"""T36-T38: thin smoke-test slice over core/views.py."""

import pytest
from django.urls import reverse

from core.models import Location, Region


# T36 -------------------------------------------------------------------
def test_welcome_view_returns_200(client):
    response = client.get(reverse("welcome"))

    assert response.status_code == 200
    assert "core/welcome.html" in [t.name for t in response.templates]


# T37 -----------------------------------------------------------------------
@pytest.mark.django_db
def test_animals_overview_redirects_anonymous_user(client):
    response = client.get(reverse("animals"))

    assert response.status_code == 302
    assert "login" in response["Location"]


@pytest.mark.django_db
def test_animals_overview_authenticated_returns_200_with_expected_context(
    client, user_factory
):
    user = user_factory(username="dave")
    client.force_login(user)

    response = client.get(reverse("animals"))

    assert response.status_code == 200
    assert "core/animal_overview.html" in [t.name for t in response.templates]
    assert "observed_animals" in response.context
    assert "unobserved_animals" in response.context
    assert "nr_unidentified_images" in response.context


# T38 --------------------------------------------------------------------
@pytest.mark.django_db
def test_image_read_existing_image_returns_200(
    client, user_factory, ibex_image_factory
):
    user = user_factory(username="erin")
    image = ibex_image_factory(owner=user)
    client.force_login(user)

    response = client.get(reverse("read-image", kwargs={"oid": image.id}))

    assert response.status_code == 200
    assert "core/image_read_new.html" in [t.name for t in response.templates]


@pytest.mark.django_db
def test_image_read_missing_image_returns_404(client, user_factory):
    user = user_factory(username="frank")
    client.force_login(user)

    response = client.get(reverse("read-image", kwargs={"oid": 999999}))

    assert response.status_code == 404


# T02 -------------------------------------------------------------------
@pytest.mark.django_db
def test_create_loaction_view_shows_region_owned_by_other_user(
    client, user_factory, region_factory, ibex_image_factory
):
    user_a = user_factory(username="t02_user_a")
    user_b = user_factory(username="t02_user_b")
    region = region_factory(owner=user_a, name="T02Region")
    image = ibex_image_factory(owner=user_b)
    client.force_login(user_b)

    response = client.get(reverse("locate-image", kwargs={"oid": image.id}))

    assert response.status_code == 200
    assert region in list(response.context["regions"])


# T04 -------------------------------------------------------------------
@pytest.mark.django_db
def test_create_loaction_view_shows_all_regions_not_just_cross_owner(
    client, user_factory, region_factory, ibex_image_factory
):
    """Mirrors T03 at the view level -- both region_a (owned by user_a) and
    region_b (owned by the logged-in/image-owning user_b) must be visible."""
    user_a = user_factory(username="t04_user_a")
    user_b = user_factory(username="t04_user_b")
    region_a = region_factory(owner=user_a, name="T04RegionA")
    region_b = region_factory(owner=user_b, name="T04RegionB")
    image = ibex_image_factory(owner=user_b)
    client.force_login(user_b)

    response = client.get(reverse("locate-image", kwargs={"oid": image.id}))

    regions = list(response.context["regions"])
    assert region_a in regions
    assert region_b in regions


# T05b --------------------------------------------------------------------
@pytest.mark.django_db
def test_create_loaction_view_shows_orphaned_region_with_no_owner(
    client, user_factory, region_factory, ibex_image_factory
):
    user = user_factory(username="t05b_user")
    orphan_region = region_factory(owner=None, name="T05B_ORPHAN")
    image = ibex_image_factory(owner=user)
    client.force_login(user)

    response = client.get(reverse("locate-image", kwargs={"oid": image.id}))

    assert response.status_code == 200
    assert orphan_region in list(response.context["regions"])


# T07 -------------------------------------------------------------------
@pytest.mark.django_db
def test_create_loaction_region_visibility_same_for_existing_vs_new_location(
    client, user_factory, region_factory, ibex_image_factory
):
    """Branch-parity: region_qs is built unconditionally after the
    `if not image_location:` branch (views.py:656-660), so cross-owner
    region visibility must be identical whether that branch created a new
    Location or the image already had one attached."""
    user = user_factory(username="t07_user")
    other = user_factory(username="t07_other")
    region = region_factory(owner=other, name="T07Region")
    client.force_login(user)

    image_without_location = ibex_image_factory(owner=user, name="t07_no_loc")
    existing_location = Location.objects.create(latitude=1.0, longitude=2.0)
    image_with_location = ibex_image_factory(
        owner=user, name="t07_has_loc", location=existing_location
    )

    response_new = client.get(
        reverse("locate-image", kwargs={"oid": image_without_location.id})
    )
    response_existing = client.get(
        reverse("locate-image", kwargs={"oid": image_with_location.id})
    )

    assert response_new.status_code == 200
    assert response_existing.status_code == 200
    assert region in list(response_new.context["regions"])
    assert region in list(response_existing.context["regions"])


# T08 -------------------------------------------------------------------
@pytest.mark.django_db
def test_create_loaction_view_empty_region_list_returns_200(
    client, user_factory, ibex_image_factory
):
    user = user_factory(username="t08_user")
    image = ibex_image_factory(owner=user)
    client.force_login(user)

    response = client.get(reverse("locate-image", kwargs={"oid": image.id}))

    assert response.status_code == 200
    assert list(response.context["regions"]) == []


# T09 -------------------------------------------------------------------
@pytest.mark.django_db
def test_region_edit_permission_unchanged_for_non_owner(
    client, user_factory, region_factory
):
    """Regression guard for R3: the R1/R2 fix must NOT change EDIT
    permission scoping. All three mutate paths must keep rejecting a
    non-owner exactly as before this CR. Must pass both before and after
    the R1/R2 fix -- it is a standing guard, not something that should
    flip."""
    owner = user_factory(username="t09_owner")
    attacker = user_factory(username="t09_attacker")
    region = region_factory(owner=owner, name="T09ProtectedRegion")
    client.force_login(attacker)

    # 1. save-region (update path, views.py:534): get_object_or_404 filters
    #    by owner=request.user, so a non-owner's region-id doesn't resolve.
    response = client.post(
        reverse("save-region"),
        {
            "region-id": region.pk,
            "region-name": "hacked-name",
            "radius": 1000,
            "latitude": 1.0,
            "longitude": 2.0,
        },
    )
    assert response.status_code == 404
    region.refresh_from_db()
    assert region.name == "T09ProtectedRegion"

    # 2. update-region (views.py:604-606): explicit ownership check -> 403.
    response = client.get(reverse("update-region", kwargs={"oid": region.pk}))
    assert response.status_code == 403

    # 3. delete-region (views.py:593): get_object_or_404 filters by
    #    owner=request.user, so a non-owner's region-id doesn't resolve.
    response = client.post(reverse("delete-region", kwargs={"oid": region.pk}))
    assert response.status_code == 404
    assert Region.objects.filter(pk=region.pk).exists()
