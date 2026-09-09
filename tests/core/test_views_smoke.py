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


# ---------------------------------------------------------------------------
# Identification tab (images_overview) -- duplicated animals listing removal
# ---------------------------------------------------------------------------
# T01 -------------------------------------------------------------------
@pytest.mark.django_db
def test_images_overview_context_excludes_animals_key(
    client, user_factory, animal_factory, ibex_image_factory
):
    """T01 (P0, R1+R3): the 'animals' context key must be absent entirely
    (not just empty) once the dead duplicated-listing query is removed.
    Must be RED pre-fix, GREEN post-fix."""
    user = user_factory(username="t01_user")
    animal = animal_factory(id_code="IDCTX01")
    ibex_image_factory(owner=user, name="t01_identified", animal=animal)
    ibex_image_factory(owner=user, name="t01_unidentified")
    client.force_login(user)

    response = client.get(reverse("images-overview"))

    assert response.status_code == 200
    assert "core/images_overview.html" in [t.name for t in response.templates]
    assert "animals" not in response.context
    assert "no_id_count" in response.context


# T02 -------------------------------------------------------------------
@pytest.mark.django_db
@pytest.mark.parametrize("n", [0, 1, 2, 5])
def test_images_overview_no_id_count_matches_unidentified_count(
    client, user_factory, ibex_image_factory, n
):
    user = user_factory(username=f"t02_user_{n}")
    for i in range(n):
        ibex_image_factory(owner=user, name=f"t02_img_{i}")
    client.force_login(user)

    response = client.get(reverse("images-overview"))

    assert response.context["no_id_count"] == n


# T03 -------------------------------------------------------------------
@pytest.mark.django_db
def test_images_overview_no_id_count_excludes_other_users_images(
    client, user_factory, ibex_image_factory
):
    user = user_factory(username="t03_user")
    other = user_factory(username="t03_other")
    ibex_image_factory(owner=user, name="t03_own")
    ibex_image_factory(owner=other, name="t03_other_1")
    ibex_image_factory(owner=other, name="t03_other_2")
    client.force_login(user)

    response = client.get(reverse("images-overview"))

    assert response.context["no_id_count"] == 1


# T04 -------------------------------------------------------------------
@pytest.mark.django_db
def test_images_overview_no_id_count_excludes_own_identified_images(
    client, user_factory, animal_factory, ibex_image_factory
):
    user = user_factory(username="t04_user")
    animal = animal_factory(id_code="T04AN01")
    ibex_image_factory(owner=user, name="t04_identified", animal=animal)
    ibex_image_factory(owner=user, name="t04_unidentified")
    client.force_login(user)

    response = client.get(reverse("images-overview"))

    assert response.context["no_id_count"] == 1


# T05 -------------------------------------------------------------------
@pytest.mark.django_db
def test_images_overview_redirects_anonymous_user(client):
    """Mirrors test_animals_overview_redirects_anonymous_user."""
    response = client.get(reverse("images-overview"))

    assert response.status_code == 302
    assert "login" in response["Location"]


# T06 -------------------------------------------------------------------
@pytest.mark.django_db
def test_images_overview_empty_db_returns_zero_count_no_animals_key(
    client, user_factory
):
    user = user_factory(username="t06_user")
    client.force_login(user)

    response = client.get(reverse("images-overview"))

    assert response.status_code == 200
    assert response.context["no_id_count"] == 0
    assert "animals" not in response.context


# T07 -------------------------------------------------------------------
@pytest.mark.django_db
def test_images_overview_only_identified_images_zero_count_id_code_absent(
    client, user_factory, animal_factory, ibex_image_factory
):
    user = user_factory(username="t07_user")
    animal = animal_factory(id_code="ONLYID1")
    ibex_image_factory(owner=user, name="t07_identified", animal=animal)
    client.force_login(user)

    response = client.get(reverse("images-overview"))

    assert response.status_code == 200
    assert response.context["no_id_count"] == 0
    assert "ONLYID1" not in response.content.decode()


# T08 -------------------------------------------------------------------
@pytest.mark.django_db
def test_images_overview_does_not_render_identified_animal(
    client, user_factory, animal_factory, ibex_image_factory
):
    """Must be RED pre-fix (renders at template line 31 currently), GREEN
    post-fix. Never assert on `b"id_code" not in content` -- that's the
    literal <th> header text and must remain (see T10)."""
    user = user_factory(username="t08_user")
    animal = animal_factory(id_code="ZZTOP01", name="ZZTopAnimalName")
    ibex_image_factory(owner=user, name="t08_identified", animal=animal)
    client.force_login(user)

    response = client.get(reverse("images-overview"))
    content = response.content.decode()

    assert "ZZTOP01" not in content
    assert "ZZTopAnimalName" not in content


# T09 -------------------------------------------------------------------
@pytest.mark.django_db
@pytest.mark.parametrize("n", [0, 3])
def test_images_overview_unidentified_row_survives(
    client, user_factory, ibex_image_factory, n
):
    user = user_factory(username=f"t09_user_{n}")
    for i in range(n):
        ibex_image_factory(owner=user, name=f"t09_img_{i}")
    client.force_login(user)

    response = client.get(reverse("images-overview"))
    content = response.content.decode()
    normalized = " ".join(content.split())

    assert reverse("unidentified-images") in content
    # str(n) alone is vacuous: digits from unrelated boilerplate (viewport
    # meta, font/alpine.js version strings) would satisfy it regardless of
    # whether {{no_id_count}} actually rendered. Assert the rendered cell.
    assert f"<td>{n}</td>" in normalized


# T10 -------------------------------------------------------------------
@pytest.mark.django_db
def test_images_overview_table_header_row_unchanged(client, user_factory):
    user = user_factory(username="t10_user")
    client.force_login(user)

    response = client.get(reverse("images-overview"))
    normalized = " ".join(response.content.decode().split())

    assert "<th>id_code</th>" in normalized
    assert "<th>name</th>" in normalized
    assert "<th>capture date</th>" in normalized
    assert "<th>images</th>" in normalized


# T11 -------------------------------------------------------------------
@pytest.mark.django_db
def test_images_overview_no_animal_own_images_link_when_animal_exists(
    client, user_factory, animal_factory, ibex_image_factory
):
    """Redundant-by-construction with T08 but kills a different template
    mutant (the href, not the rendered id_code/name)."""
    user = user_factory(username="t11_user")
    animal = animal_factory(id_code="T11ANML")
    ibex_image_factory(owner=user, name="t11_identified", animal=animal)
    client.force_login(user)

    response = client.get(reverse("images-overview"))

    assert "/animal-own-images/" not in response.content.decode()


# T12 -------------------------------------------------------------------
@pytest.mark.django_db
def test_animals_overview_identified_animal_is_observed_and_rendered(
    client, user_factory, animal_factory, ibex_image_factory
):
    """R2 regression guard -- extends the existing context-key-presence
    test (test_animals_overview_authenticated_returns_200_with_expected_context
    above, kept as-is) with membership + rendering assertions."""
    user = user_factory(username="t12_user")
    animal = animal_factory(id_code="PARITY01")
    ibex_image_factory(owner=user, name="t12_identified", animal=animal)
    client.force_login(user)

    response = client.get(reverse("animals"))

    assert response.status_code == 200
    assert "core/animal_overview.html" in [t.name for t in response.templates]
    assert "observed_animals" in response.context
    assert "unobserved_animals" in response.context
    assert "nr_unidentified_images" in response.context
    assert animal in list(response.context["observed_animals"])
    assert "PARITY01" in response.content.decode()


# T13 -------------------------------------------------------------------
@pytest.mark.django_db
def test_results_over_view_shows_only_identified_images(
    client, user_factory, animal_factory, ibex_image_factory
):
    user = user_factory(username="t13_user")
    animal = animal_factory(id_code="T13AN001")
    identified = ibex_image_factory(
        owner=user, name="t13_identified", animal=animal
    )
    unidentified = ibex_image_factory(owner=user, name="t13_unidentified")
    client.force_login(user)

    response = client.get(reverse("results-overview"))

    images = list(response.context["images"])
    assert identified in images
    assert unidentified not in images


# T14 -------------------------------------------------------------------
@pytest.mark.django_db
def test_unidentified_images_view_returns_only_caller_own_unidentified_images(
    client, user_factory, ibex_image_factory
):
    user = user_factory(username="t14_user")
    other = user_factory(username="t14_other")
    own_1 = ibex_image_factory(owner=user, name="t14_own_1")
    own_2 = ibex_image_factory(owner=user, name="t14_own_2")
    ibex_image_factory(owner=other, name="t14_other")
    client.force_login(user)

    response = client.get(reverse("unidentified-images"))

    assert set(response.context["images"]) == {own_1, own_2}


# T16 -------------------------------------------------------------------
@pytest.mark.django_db
@pytest.mark.parametrize("k,m", [(0, 0), (1, 0), (0, 1), (3, 2)])
def test_images_overview_no_id_count_matches_unidentified_view_image_count(
    client, user_factory, animal_factory, ibex_image_factory, k, m
):
    """Metamorphic: no_id_count on /identification/ must equal
    len(images) on /unidentified/ for the same user, regardless of how
    many identified images that user (or another user) also owns."""
    user = user_factory(username=f"t16_user_{k}_{m}")
    other = user_factory(username=f"t16_other_{k}_{m}")
    for i in range(k):
        ibex_image_factory(owner=user, name=f"t16_unid_{i}")
    if m:
        animal = animal_factory(id_code=f"T16A{k}{m}")
        for i in range(m):
            ibex_image_factory(owner=user, name=f"t16_id_{i}", animal=animal)
    ibex_image_factory(owner=other, name="t16_other_owner")
    client.force_login(user)

    identification_response = client.get(reverse("images-overview"))
    unidentified_response = client.get(reverse("unidentified-images"))

    assert identification_response.context["no_id_count"] == len(
        unidentified_response.context["images"]
    )
