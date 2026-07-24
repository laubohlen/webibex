"""T36-T38: thin smoke-test slice over core/views.py."""

import pytest
from django.urls import reverse


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
def test_image_read_existing_image_returns_200(client, user_factory, ibex_image_factory):
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
