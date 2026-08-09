"""T01-T25: login_required gate coverage for the 6 previously-unauthenticated
views in core/views.py (save_landmarks_view, results_over_view,
default_chip_compare_view, project_chip_compare_view,
geographic_chip_compare_view, rerun_view). See
docs/security-remediation-plan.md's 2026-08-14 "New, more severe finding"
section for the vulnerability this closes.
"""

from datetime import datetime
from unittest import mock

import pytest
from django.contrib.auth.models import AnonymousUser
from django.contrib.contenttypes.models import ContentType
from django.http import HttpResponseRedirect
from django.template import TemplateDoesNotExist
from django.test import RequestFactory
from django.urls import reverse
from django.utils import timezone

from core import views as core_views
from core.models import Embedding, IbexChip, IbexImage
from core.views import geographic_chip_compare_view
from simple_landmarks.models import Landmark, LandmarkItem

# ---------------------------------------------------------------------------
# File-local fixtures.
# ---------------------------------------------------------------------------


@pytest.fixture
def embedding_factory(db):
    def _make(chip, vector=None):
        if vector is None:
            vector = [1.0, 2.0, 3.0]
        return Embedding.objects.create(ibex_chip=chip, embedding=vector)

    return _make


@pytest.fixture
def landmark_setup(ibex_image_factory):
    """Creates the two Landmark labels the view looks up by name, plus one
    LandmarkItem per label attached to `image` (new, via ibex_image_factory,
    unless `image` is passed explicitly). Returns (image, horn_item,
    eye_item)."""

    def _make(owner=None, image=None, **image_overrides):
        if image is None:
            image = ibex_image_factory(owner=owner, **image_overrides)
        horn_landmark = Landmark.objects.create(label="horn_tip")
        eye_landmark = Landmark.objects.create(label="eye_corner")
        content_type = ContentType.objects.get_for_model(IbexImage)
        horn_item = LandmarkItem.objects.create(
            landmark=horn_landmark, content_type=content_type, object_id=image.id
        )
        eye_item = LandmarkItem.objects.create(
            landmark=eye_landmark, content_type=content_type, object_id=image.id
        )
        return image, horn_item, eye_item

    return _make


@pytest.fixture
def chip_with_embedding(
    user_factory, ibex_image_factory, ibex_chip_factory, embedding_factory
):
    """Composite convenience: user -> image -> chip(ibex_image=image) ->
    embedding. Dimension is fixed ([1.0, 2.0, 3.0] by default) so two chips
    built with this fixture are always distance-computable together."""

    def _make(owner=None, vector=None, image=None, **image_overrides):
        if owner is None:
            owner = user_factory()
        if image is None:
            image = ibex_image_factory(owner=owner, **image_overrides)
        chip = ibex_chip_factory(owner=owner, ibex_image=image)
        embedding_factory(chip, vector=vector)
        return chip

    return _make


@pytest.fixture
def gate_scenario(
    user_factory, chip_with_embedding, region_factory, landmark_setup, animal_factory
):
    """Shared valid precondition for the T07/T08 URL sweeps: one chip (with
    embedding) whose image has an animal set and carries landmark items,
    plus a region -- satisfies every one of the 5 gated views' preconditions
    at once, so a pre-fix (decorator-removed) request would 200 or raise a
    known exception, never 404."""
    owner = user_factory()
    # side="L": create_folder_for_animal_on_change (core/signals.py:293-305)
    # raises UnboundLocalError when animal is assigned to an image whose
    # side is not L/R/O -- pre-existing bug, pinned separately in
    # tests/core/test_signals.py, not this CR's concern. Every animal
    # assignment in this file uses side="L" to route around it.
    image, _horn_item, _eye_item = landmark_setup(owner=owner, side="L")
    image.animal = animal_factory()
    image.save()
    chip = chip_with_embedding(owner=owner, image=image)
    region = region_factory(owner=owner, name="GateScenarioRegion")
    return {"owner": owner, "image": image, "chip": chip, "region": region}


_GATE_URL_NAMES = [
    "results-overview",
    "result-default",
    "result-refined",
    "run-again",
    "save-landmarks",
]


def _resolve_gate_url(name, scenario):
    if name in ("result-default", "result-refined", "run-again"):
        return reverse(name, kwargs={"oid": scenario["chip"].id})
    return reverse(name)


def _gate_post_payload(name, scenario):
    if name == "result-refined":
        return {"toggle": "false", "region": scenario["region"].id}
    if name == "save-landmarks":
        return {
            "image-id": scenario["image"].id,
            "horn_x": "400",
            "horn_y": "800",
            "eye_x": "800",
            "eye_y": "1600",
        }
    return {}


# ---------------------------------------------------------------------------
# P0 -- anonymous access must never reach any of the 6 gated view bodies.
# ---------------------------------------------------------------------------


# T01 -------------------------------------------------------------------
@pytest.mark.django_db
def test_results_over_view_anon_redirects_to_login(
    client, user_factory, ibex_image_factory, animal_factory
):
    owner = user_factory()
    # side="L": see gate_scenario's comment on create_folder_for_animal_on_change.
    image = ibex_image_factory(owner=owner, side="L")
    image.animal = animal_factory()
    image.save()

    response = client.get(reverse("results-overview"))

    assert response.status_code == 302
    assert "/accounts/login/" in response["Location"]
    assert response.templates == []


# T02 -------------------------------------------------------------------
@pytest.mark.django_db
def test_default_chip_compare_view_anon_redirects_to_login(client, chip_with_embedding):
    chip = chip_with_embedding()

    response = client.get(reverse("result-default", kwargs={"oid": chip.id}))

    assert response.status_code == 302
    assert "/accounts/login/" in response["Location"]
    assert response.templates == []


@pytest.mark.django_db
def test_default_chip_compare_view_anon_nonexistent_oid_still_redirects_not_404(client):
    """Counter-input: the decorator must precede get_object_or_404 -- a bad
    oid must never surface as a 404 for an anonymous caller."""
    response = client.get(reverse("result-default", kwargs={"oid": 999999}))

    assert response.status_code == 302
    assert response.status_code != 404


# T03 -------------------------------------------------------------------
@pytest.mark.django_db
def test_project_chip_compare_view_anon_post_toggle_false_redirects(
    client, chip_with_embedding, region_factory
):
    chip = chip_with_embedding()
    region = region_factory(owner=chip.owner, name="T03Region")

    response = client.post(
        reverse("result-refined", kwargs={"oid": chip.id}),
        {"toggle": "false", "region": region.id},
    )

    assert response.status_code == 302
    assert "/accounts/login/" in response["Location"]
    assert response.templates == []


@pytest.mark.django_db
@pytest.mark.parametrize(
    "region_field",
    [
        pytest.param({"region": "abc"}, id="non_int_region_would_valueerror"),
        pytest.param({}, id="omitted_region_would_typeerror"),
    ],
)
def test_project_chip_compare_view_anon_post_bad_region_still_redirects(
    client, chip_with_embedding, region_field
):
    """Counter-inputs: a non-int region ('abc', would raise ValueError from
    int()) or an omitted region (would raise TypeError from int(None)) must
    both still redirect for an anonymous caller, never surface either
    exception."""
    chip = chip_with_embedding()
    data = {"toggle": "false", **region_field}

    response = client.post(reverse("result-refined", kwargs={"oid": chip.id}), data)

    assert response.status_code == 302
    assert response.status_code != 404


# T04 -------------------------------------------------------------------
@pytest.mark.django_db
def test_project_chip_compare_view_anon_post_without_toggle_does_not_delegate(
    client, chip_with_embedding
):
    chip = chip_with_embedding()

    with mock.patch("core.views.geographic_chip_compare_view") as delegate_mock:
        response = client.post(reverse("result-refined", kwargs={"oid": chip.id}), {})

    delegate_mock.assert_not_called()
    assert response.status_code == 302


@pytest.mark.django_db
def test_project_chip_compare_view_anon_post_toggle_capital_false_does_not_delegate(
    client, chip_with_embedding
):
    """Counter-input: 'False' (capital F) fails the literal 'false' string
    match just like a missing toggle -- must still not delegate."""
    chip = chip_with_embedding()

    with mock.patch("core.views.geographic_chip_compare_view") as delegate_mock:
        response = client.post(
            reverse("result-refined", kwargs={"oid": chip.id}), {"toggle": "False"}
        )

    delegate_mock.assert_not_called()
    assert response.status_code == 302


# T05 -------------------------------------------------------------------
@pytest.mark.django_db
def test_rerun_view_anon_redirects_cleanly_not_template_does_not_exist(
    client, chip_with_embedding
):
    """Anon GET must be a clean redirect -- not the pinned
    TemplateDoesNotExist bug (see T20), which now only manifests for an
    authenticated caller."""
    chip = chip_with_embedding()

    response = client.get(reverse("run-again", kwargs={"oid": chip.id}))

    assert response.status_code == 302
    assert "/accounts/login/" in response["Location"]


# T06 -------------------------------------------------------------------
@pytest.mark.django_db
def test_save_landmarks_view_anon_post_redirects_to_login_not_success(
    client, landmark_setup
):
    image, _horn_item, _eye_item = landmark_setup()

    response = client.post(
        reverse("save-landmarks"),
        {
            "image-id": image.id,
            "horn_x": "400",
            "horn_y": "800",
            "eye_x": "800",
            "eye_y": "1600",
        },
    )

    assert response.status_code == 302
    assert "/accounts/login/" in response["Location"]
    assert response["Location"] != reverse("unidentified-images")


# T07 -------------------------------------------------------------------
@pytest.mark.django_db
@pytest.mark.parametrize("url_name", _GATE_URL_NAMES)
def test_all_gated_views_anon_get_never_200(client, gate_scenario, url_name):
    """GET sweep over all 5 URL-routed gated views. Doubles as 2
    counter-inputs baked into the parametrize list itself: 'result-refined'
    GET (empty POST body) would TypeError pre-fix (int(None) in the
    delegated geographic_chip_compare_view branch); 'save-landmarks' GET
    (no POST branch) would ValueError pre-fix (bare `else: pass` -> implicit
    None return, see T21). Both must be a clean 302 post-fix, never the
    pre-existing exception, and never 200."""
    url = _resolve_gate_url(url_name, gate_scenario)

    response = client.get(url)

    assert response.status_code == 302
    assert response.status_code != 200
    assert "/accounts/login/" in response["Location"]


# T08 -------------------------------------------------------------------
@pytest.mark.django_db
@pytest.mark.parametrize("url_name", _GATE_URL_NAMES)
def test_all_gated_views_anon_post_never_200(client, gate_scenario, url_name):
    """POST sweep over the same 5 URL-routed gated views."""
    url = _resolve_gate_url(url_name, gate_scenario)
    data = _gate_post_payload(url_name, gate_scenario)

    response = client.post(url, data)

    assert response.status_code == 302
    assert response.status_code != 200
    assert "/accounts/login/" in response["Location"]


@pytest.mark.django_db
def test_save_landmarks_view_anon_get_counter_input_redirects_not_valueerror(client):
    """Counter-input: a GET (not POST) to save-landmarks must still be
    gated before the view's own `else: pass` branch (which returns None and
    would raise ValueError, see T21) is ever reached."""
    response = client.get(reverse("save-landmarks"))

    assert response.status_code == 302
    assert "/accounts/login/" in response["Location"]


# T09 -------------------------------------------------------------------
@pytest.mark.django_db
def test_geographic_chip_compare_view_anon_direct_call_redirects(
    chip_with_embedding, region_factory
):
    """No URL route exists for this view -- call it directly via
    RequestFactory + an explicit AnonymousUser."""
    chip = chip_with_embedding()
    region = region_factory(owner=chip.owner, name="T09Region")

    request = RequestFactory().post(
        f"/geographic-test/{chip.id}/", {"region": region.id}
    )
    request.user = AnonymousUser()

    response = geographic_chip_compare_view(request, chip.id)

    assert isinstance(response, HttpResponseRedirect)
    assert response.status_code == 302
    assert response["Location"].startswith("/accounts/login/")


# T10 -------------------------------------------------------------------
@pytest.mark.django_db
def test_save_landmarks_view_anon_post_never_reaches_runpod_or_saves(
    client, landmark_setup, no_network
):
    """R2 proof: anonymous POST must never reach process_horn_chip, the
    real network boundary (no_network's post_patch), create an IbexChip, or
    mutate the LandmarkItem rows."""
    image, horn_item, eye_item = landmark_setup()
    post_patch, _resource_patch = no_network

    with mock.patch("core.views.utils.process_horn_chip") as process_mock:
        response = client.post(
            reverse("save-landmarks"),
            {
                "image-id": image.id,
                "horn_x": "400",
                "horn_y": "800",
                "eye_x": "800",
                "eye_y": "1600",
            },
        )

    assert response.status_code == 302
    process_mock.assert_not_called()
    post_patch.assert_not_called()
    assert IbexChip.objects.count() == 0
    horn_item.refresh_from_db()
    eye_item.refresh_from_db()
    assert horn_item.x_coordinate is None
    assert horn_item.y_coordinate is None
    assert eye_item.x_coordinate is None
    assert eye_item.y_coordinate is None


# T11 -------------------------------------------------------------------
@pytest.mark.django_db
def test_save_landmarks_view_anon_post_with_next_id_index_never_delegates(
    client, landmark_setup
):
    """With next_id_index set, save_landmarks_view would delegate to
    multi_task_view on success -- anon must never reach either callable."""
    image, _horn_item, _eye_item = landmark_setup()

    with (
        mock.patch("core.views.multi_task_view") as multi_task_mock,
        mock.patch("core.views.utils.process_horn_chip") as process_mock,
    ):
        response = client.post(
            reverse("save-landmarks"),
            {
                "image-id": image.id,
                "horn_x": "400",
                "horn_y": "800",
                "eye_x": "800",
                "eye_y": "1600",
                "next_id_index": "1",
            },
        )

    assert response.status_code == 302
    multi_task_mock.assert_not_called()
    process_mock.assert_not_called()


# ---------------------------------------------------------------------------
# P1 -- authenticated behavior unchanged; pinned pre-existing bugs untouched.
# ---------------------------------------------------------------------------


# T12 -------------------------------------------------------------------
@pytest.mark.django_db
def test_results_over_view_authenticated_returns_200(
    client, user_factory, ibex_image_factory, animal_factory
):
    user = user_factory()
    client.force_login(user)
    image = ibex_image_factory(owner=user, side="L")
    image.animal = animal_factory()
    image.save()

    response = client.get(reverse("results-overview"))

    assert response.status_code == 200
    assert "core/results_overview.html" in [t.name for t in response.templates]
    assert "images" in response.context
    assert image in list(response.context["images"])


# T13 -------------------------------------------------------------------
@pytest.mark.django_db
def test_default_chip_compare_view_authenticated_empty_gallery(
    client, chip_with_embedding
):
    chip = chip_with_embedding()
    client.force_login(chip.owner)

    response = client.get(reverse("result-default", kwargs={"oid": chip.id}))

    assert response.status_code == 200
    assert response.context["gallery_and_distances"] == []
    assert response.context["n_gallery_chips"] == 0
    assert response.context["threshold"] == 9.3


# T14 -------------------------------------------------------------------
@pytest.mark.django_db
def test_default_chip_compare_view_authenticated_with_matching_gallery_chip(
    client,
    user_factory,
    ibex_image_factory,
    ibex_chip_factory,
    embedding_factory,
    animal_factory,
):
    owner = user_factory()
    query_year = 2020
    query_image = ibex_image_factory(owner=owner, name="query")
    query_image.created_at = timezone.make_aware(datetime(query_year, 6, 15))
    query_image.save()
    query_chip = ibex_chip_factory(owner=owner, ibex_image=query_image)
    embedding_factory(query_chip, vector=[1.0, 2.0, 3.0])

    gallery_image = ibex_image_factory(owner=owner, name="gallery", side="L")
    gallery_image.animal = animal_factory()
    gallery_image.created_at = timezone.make_aware(datetime(query_year, 3, 1))
    gallery_image.save()
    gallery_chip = ibex_chip_factory(owner=owner, ibex_image=gallery_image)
    embedding_factory(gallery_chip, vector=[1.1, 2.1, 3.1])

    client.force_login(owner)

    response = client.get(reverse("result-default", kwargs={"oid": query_chip.id}))

    assert response.status_code == 200
    assert len(response.context["gallery_and_distances"]) == 1
    assert response.context["n_gallery_chips"] == 1


@pytest.mark.django_db
def test_default_chip_compare_view_authenticated_excludes_other_owner(
    client,
    user_factory,
    ibex_image_factory,
    ibex_chip_factory,
    embedding_factory,
    animal_factory,
):
    """Counter-input: a gallery chip owned by a DIFFERENT user must be
    excluded -- proves owner scoping still holds post-decoration."""
    owner = user_factory(username="t14_owner")
    other = user_factory(username="t14_other")
    query_year = 2020
    query_image = ibex_image_factory(owner=owner, name="query")
    query_image.created_at = timezone.make_aware(datetime(query_year, 6, 15))
    query_image.save()
    query_chip = ibex_chip_factory(owner=owner, ibex_image=query_image)
    embedding_factory(query_chip, vector=[1.0, 2.0, 3.0])

    other_image = ibex_image_factory(owner=other, name="other_gallery", side="L")
    other_image.animal = animal_factory()
    other_image.created_at = timezone.make_aware(datetime(query_year, 3, 1))
    other_image.save()
    other_chip = ibex_chip_factory(owner=other, ibex_image=other_image)
    embedding_factory(other_chip, vector=[1.1, 2.1, 3.1])

    client.force_login(owner)

    response = client.get(reverse("result-default", kwargs={"oid": query_chip.id}))

    assert response.status_code == 200
    assert response.context["n_gallery_chips"] == 0


@pytest.mark.django_db
def test_default_chip_compare_view_authenticated_excludes_year_boundary(
    client,
    user_factory,
    ibex_image_factory,
    ibex_chip_factory,
    embedding_factory,
    animal_factory,
):
    """Counter-input: gallery image year = query year + 5 is outside the
    [-4, +4] window -- boundary must exclude it."""
    owner = user_factory()
    query_year = 2020
    query_image = ibex_image_factory(owner=owner, name="query")
    query_image.created_at = timezone.make_aware(datetime(query_year, 6, 15))
    query_image.save()
    query_chip = ibex_chip_factory(owner=owner, ibex_image=query_image)
    embedding_factory(query_chip, vector=[1.0, 2.0, 3.0])

    far_image = ibex_image_factory(owner=owner, name="far_gallery", side="L")
    far_image.animal = animal_factory()
    far_image.created_at = timezone.make_aware(datetime(query_year + 5, 3, 1))
    far_image.save()
    far_chip = ibex_chip_factory(owner=owner, ibex_image=far_image)
    embedding_factory(far_chip, vector=[1.1, 2.1, 3.1])

    client.force_login(owner)

    response = client.get(reverse("result-default", kwargs={"oid": query_chip.id}))

    assert response.status_code == 200
    assert response.context["n_gallery_chips"] == 0


# T15 -------------------------------------------------------------------
@pytest.mark.django_db
def test_project_chip_compare_view_authenticated_toggle_false_returns_result_refined(
    client, chip_with_embedding, region_factory
):
    chip = chip_with_embedding()
    region = region_factory(owner=chip.owner, name="T15Region")
    client.force_login(chip.owner)

    response = client.post(
        reverse("result-refined", kwargs={"oid": chip.id}),
        {"toggle": "false", "region": region.id},
    )

    assert response.status_code == 200
    assert "core/result_refined.html" in [t.name for t in response.templates]
    assert response.context["region"] == region
    assert "n_regions" not in response.context


# T16 -------------------------------------------------------------------
@pytest.mark.django_db
def test_project_chip_compare_view_authenticated_delegates_to_geographic(
    client, chip_with_embedding, region_factory
):
    """Delegation A: region posted but no toggle -> real delegation to
    geographic_chip_compare_view, proven via a spy (wraps=), not a stub."""
    chip = chip_with_embedding()
    region = region_factory(owner=chip.owner, name="T16Region")
    client.force_login(chip.owner)

    with mock.patch(
        "core.views.geographic_chip_compare_view",
        wraps=core_views.geographic_chip_compare_view,
    ) as spy:
        response = client.post(
            reverse("result-refined", kwargs={"oid": chip.id}), {"region": region.id}
        )

    spy.assert_called_once()
    assert response.status_code == 200
    assert "core/result_refined.html" in [t.name for t in response.templates]
    assert "n_regions" in response.context


# T17 (P2) ----------------------------------------------------------------
@pytest.mark.django_db
def test_geographic_chip_compare_view_authenticated_direct_call_returns_200(
    chip_with_embedding, region_factory
):
    chip = chip_with_embedding()
    region = region_factory(owner=chip.owner, name="T17Region")

    request = RequestFactory().post(
        f"/geographic-test/{chip.id}/", {"region": region.id}
    )
    request.user = chip.owner

    response = geographic_chip_compare_view(request, chip.id)

    assert response.status_code == 200
    assert response.content


# T18 -------------------------------------------------------------------
@pytest.mark.django_db
def test_save_landmarks_view_authenticated_happy_path_scales_and_processes(
    client, landmark_setup
):
    """image.width == 4 (tiny_png_bytes fixture), LANDMARK_IMAGE_WIDTH ==
    1600 -> scale == 0.0025. horn (400, 800) -> (1, 2); eye (800, 1600) ->
    (2, 4)."""
    image, horn_item, eye_item = landmark_setup()
    client.force_login(image.owner)

    with mock.patch("core.views.utils.process_horn_chip") as process_mock:
        response = client.post(
            reverse("save-landmarks"),
            {
                "image-id": image.id,
                "horn_x": "400",
                "horn_y": "800",
                "eye_x": "800",
                "eye_y": "1600",
            },
        )

    assert response.status_code == 302
    assert response["Location"] == reverse("unidentified-images")

    horn_item.refresh_from_db()
    eye_item.refresh_from_db()
    assert (horn_item.x_coordinate, horn_item.y_coordinate) == (1, 2)
    assert (eye_item.x_coordinate, eye_item.y_coordinate) == (2, 4)
    # Counter-input: the UNSCALED POST values must NOT appear -- pins the
    # scale direction, catching a src/dst-argument inversion in
    # scale_coordinate's call sites.
    assert (horn_item.x_coordinate, horn_item.y_coordinate) != (400, 800)
    assert (eye_item.x_coordinate, eye_item.y_coordinate) != (800, 1600)

    process_mock.assert_called_once_with(image, 1, 2, 2, 4)


# T19 -------------------------------------------------------------------
@pytest.mark.django_db
def test_save_landmarks_view_authenticated_delegates_to_multi_task_view(
    client, user_factory, ibex_image_factory, landmark_setup
):
    """Delegation B: next_id_index set -> save_landmarks_view delegates
    into multi_task_view's 'landmark' branch for the SECOND selected image.
    process_horn_chip is mocked even though not part of the delegation
    assertion itself -- otherwise this authenticated happy path would
    attempt a real RunPod HTTP call and trip the autouse no_network guard
    (see T10's R2 proof for why that guard exists)."""
    owner = user_factory()
    image_a = ibex_image_factory(owner=owner, name="a")
    image_b = ibex_image_factory(owner=owner, name="b")
    _image, _horn_item, _eye_item = landmark_setup(owner=owner, image=image_a)
    client.force_login(owner)

    with mock.patch("core.views.utils.process_horn_chip"):
        response = client.post(
            reverse("save-landmarks"),
            {
                "image-id": image_a.id,
                "horn_x": "400",
                "horn_y": "800",
                "eye_x": "800",
                "eye_y": "1600",
                "next_id_index": "1",
                "task": "landmark",
                "selected-files": f"{image_a.id},{image_b.id}",
            },
        )

    assert response.status_code == 200
    template_names = [t.name for t in response.templates]
    assert "simple_landmarks/multi_landmarking.html" in template_names
    assert response.context["image"] == image_b
    assert response.context["current_id_index"] == 1
    assert response.context["next_id_index"] is None


# T20 (PIN) ---------------------------------------------------------------
@pytest.mark.django_db
def test_rerun_view_authenticated_still_raises_template_does_not_exist(
    client, chip_with_embedding
):
    """PIN: pre-existing bug, explicitly out of scope for this CR -- see
    docs/security-remediation-plan.md's rerun_view entry (renders
    'core/result.html', which does not exist as a file). login_required
    only gates anonymous access (T05); it does not change or mask this
    authenticated-path bug."""
    chip = chip_with_embedding()
    client.force_login(chip.owner)

    with pytest.raises(TemplateDoesNotExist):
        client.get(reverse("run-again", kwargs={"oid": chip.id}))


# T21 (PIN) ---------------------------------------------------------------
@pytest.mark.django_db
def test_save_landmarks_view_authenticated_get_still_raises_value_error(
    client, user_factory
):
    """PIN: pre-existing bug, explicitly out of scope -- the view's
    `else: pass` branch (no POST) implicitly returns None, and Django's
    BaseHandler raises ValueError for a view that returns no HttpResponse.
    See docs/security-remediation-plan.md."""
    user = user_factory()
    client.force_login(user)

    with pytest.raises(ValueError):
        client.get(reverse("save-landmarks"))


@pytest.mark.django_db
def test_save_landmarks_view_anon_get_contrast_redirects_no_exception(client):
    """Counter-input/contrast for T21: the SAME GET request, anonymous,
    must be a clean redirect -- the decorator intercepts before the buggy
    `else: pass` branch is ever reached."""
    response = client.get(reverse("save-landmarks"))

    assert response.status_code == 302
    assert "/accounts/login/" in response["Location"]


# T22 (P2, PIN) -------------------------------------------------------------
@pytest.mark.django_db
def test_project_chip_compare_view_authenticated_toggle_false_missing_region_raises(
    client, chip_with_embedding
):
    """PIN: pre-existing bug, out of scope -- toggle='false' without a
    region posted hits int(None) at views.py:271. Documents why the
    region-required precondition is mandatory across this file's fixtures,
    not an oversight."""
    chip = chip_with_embedding()
    client.force_login(chip.owner)

    with pytest.raises(TypeError):
        client.post(
            reverse("result-refined", kwargs={"oid": chip.id}), {"toggle": "false"}
        )


# T23 (P2, PIN) -------------------------------------------------------------
@pytest.mark.django_db
def test_default_chip_compare_view_authenticated_chip_without_embedding_raises(
    client, user_factory, ibex_image_factory, ibex_chip_factory
):
    """PIN: pre-existing bug, out of scope -- a chip with no Embedding row
    raises Embedding.DoesNotExist (RelatedObjectDoesNotExist) at
    query.embedding.embedding. Documents why embedding_factory is
    mandatory in every other fixture in this file, not an oversight."""
    owner = user_factory()
    image = ibex_image_factory(owner=owner)
    chip = ibex_chip_factory(owner=owner, ibex_image=image)
    client.force_login(owner)

    with pytest.raises(Embedding.DoesNotExist):
        client.get(reverse("result-default", kwargs={"oid": chip.id}))


# T24 -------------------------------------------------------------------
@pytest.mark.django_db
def test_login_redirect_exact_contract(client):
    assert reverse("account_login") == "/accounts/login/"

    response = client.get(reverse("results-overview"))

    expected_next = "/accounts/login/?next=" + reverse("results-overview")
    assert response["Location"] == expected_next


# T25 (P2, supplementary) ---------------------------------------------------
@pytest.mark.parametrize(
    "view",
    [
        core_views.save_landmarks_view,
        core_views.results_over_view,
        core_views.default_chip_compare_view,
        core_views.project_chip_compare_view,
        core_views.geographic_chip_compare_view,
        core_views.rerun_view,
    ],
)
def test_gated_views_are_login_required_wrapped(view):
    """Supplementary introspection only -- NOT a substitute for the
    behavioral T01-T09 redirect proofs above."""
    assert hasattr(view, "__wrapped__")
