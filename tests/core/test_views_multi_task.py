"""R1-R4: Tools-menu delete regression suite for core.views.multi_task_view."""

import re
from unittest import mock

import pytest
from django.urls import reverse

from core.models import IbexChip, IbexImage

# ---------------------------------------------------------------------------
# P0 -- core fix, red before fix / green after
# ---------------------------------------------------------------------------


@pytest.mark.django_db
def test_multi_task_view_delete_single_image_redirects_to_unidentified_images(
    client, user_factory, ibex_image_factory
):
    user = user_factory(username="p0_single")
    image = ibex_image_factory(owner=user)
    client.force_login(user)

    response = client.post(
        reverse("multi-task"),
        {"selected-files": str(image.id), "task": "delete"},
    )

    assert response.status_code == 302
    assert response["Location"] == reverse("unidentified-images")


@pytest.mark.django_db
def test_multi_task_view_delete_multiple_images_redirects_to_unidentified_images(
    client, user_factory, ibex_image_factory
):
    user = user_factory(username="p0_multi")
    image_a = ibex_image_factory(owner=user, name="a")
    image_b = ibex_image_factory(owner=user, name="b")
    client.force_login(user)

    response = client.post(
        reverse("multi-task"),
        {
            "selected-files": f"{image_a.id},{image_b.id}",
            "task": "delete",
        },
    )

    assert response.status_code == 302
    assert response["Location"] == reverse("unidentified-images")


@pytest.mark.django_db
def test_multi_task_view_delete_does_not_delete_any_image_or_chip(
    client, user_factory, ibex_image_factory, ibex_chip_factory
):
    """Load-bearing invariant: this CR hides the crash WITHOUT implementing
    real deletion. Nothing in the DB should be touched by task=delete."""
    user = user_factory(username="p0_no_delete")
    image_a = ibex_image_factory(owner=user, name="a")
    image_b = ibex_image_factory(owner=user, name="b")
    chip = ibex_chip_factory(owner=user, ibex_image=image_a)
    client.force_login(user)

    client.post(
        reverse("multi-task"),
        {
            "selected-files": f"{image_a.id},{image_b.id}",
            "task": "delete",
        },
    )

    assert (
        IbexImage.objects.filter(pk__in=[image_a.pk, image_b.pk]).count() == 2
    )
    assert IbexChip.objects.filter(pk=chip.pk).exists()


@pytest.mark.django_db
def test_multi_task_view_delete_never_calls_multi_task_url(
    client, user_factory, ibex_image_factory
):
    """Executable proof that `delete` no longer routes through
    `utils.multi_task_url` at all -- protects the untouched utils.py."""
    user = user_factory(username="p0_no_utils_call")
    image = ibex_image_factory(owner=user)
    client.force_login(user)

    with mock.patch("core.views.utils.multi_task_url") as mocked:
        response = client.post(
            reverse("multi-task"),
            {"selected-files": str(image.id), "task": "delete"},
        )

    mocked.assert_not_called()
    assert response.status_code == 302


@pytest.mark.django_db
def test_multi_task_view_delete_via_unidentified_images_url_redirects(
    client, user_factory, ibex_image_factory
):
    """The REAL UI path: the Tools-menu form posts to `unidentified-images`
    (with a `next` query param), not directly to `multi-task/`."""
    user = user_factory(username="p0_real_ui_path")
    image = ibex_image_factory(owner=user)
    client.force_login(user)

    response = client.post(
        reverse("unidentified-images") + "?next=/unidentified/",
        {"selected-files": str(image.id), "task": "delete"},
    )

    assert response.status_code == 302
    assert response["Location"] == reverse("unidentified-images")


@pytest.mark.django_db
def test_multi_task_view_delete_with_nonexistent_image_ids_still_redirects(
    client, user_factory
):
    """Proves the new branch is placed BEFORE get_object_or_404 (views.py:852),
    not after -- a nonexistent id must not 404."""
    user = user_factory(username="p0_nonexistent_ids")
    client.force_login(user)

    response = client.post(
        reverse("multi-task"),
        {"selected-files": "999999", "task": "delete"},
    )

    assert response.status_code == 302
    assert response["Location"] == reverse("unidentified-images")


@pytest.mark.django_db
def test_multi_task_view_delete_requires_login(client):
    """Every OTHER redirect test in this file asserts the exact Location
    header to prove the redirect target is `unidentified-images` (the fix),
    not this login-redirect 302 -- which points at the login page instead."""
    response = client.post(
        reverse("multi-task"),
        {"selected-files": "1", "task": "delete"},
    )

    assert response.status_code == 302
    assert "login" in response["Location"]
    assert response["Location"] != reverse("unidentified-images")


# ---------------------------------------------------------------------------
# P1 -- template
# ---------------------------------------------------------------------------


@pytest.mark.django_db
def test_unidentified_images_page_has_no_delete_option(
    client, user_factory
):
    user = user_factory(username="p1_no_delete_option")
    client.force_login(user)

    response = client.get(reverse("unidentified-images"))

    assert response.status_code == 200
    assert "core/unidentified_images.html" in [
        t.name for t in response.templates
    ]
    assert 'value="delete"' not in response.content.decode()


@pytest.mark.django_db
def test_unidentified_images_page_tools_options_are_exactly_the_expected_set(
    client, user_factory
):
    user = user_factory(username="p1_exact_options")
    client.force_login(user)

    response = client.get(reverse("unidentified-images"))
    html = response.content.decode()

    options = re.findall(r'<option value="([^"]*)"', html)

    assert options == [
        "",
        "tag_left",
        "tag_right",
        "tag_other",
        "locate",
        "landmark",
        "view",
    ]


# ---------------------------------------------------------------------------
# P1 -- no-regression on other Tools branches
# ---------------------------------------------------------------------------


@pytest.mark.django_db
@pytest.mark.parametrize(
    "task,expected_side",
    [("tag_left", "L"), ("tag_right", "R"), ("tag_other", "O")],
)
def test_multi_task_view_tag_branches_still_update_side_and_redirect(
    client, user_factory, ibex_image_factory, task, expected_side
):
    user = user_factory(username=f"p1_tag_{task}")
    image_a = ibex_image_factory(owner=user, name="a", side=None)
    image_b = ibex_image_factory(owner=user, name="b", side=None)
    client.force_login(user)

    response = client.post(
        reverse("multi-task"),
        {
            "selected-files": f"{image_a.id},{image_b.id}",
            "task": task,
        },
    )

    assert response.status_code == 302
    assert response["Location"] == reverse("unidentified-images")

    image_a.refresh_from_db()
    image_b.refresh_from_db()
    assert image_a.side == expected_side
    assert image_b.side == expected_side


@pytest.mark.django_db
def test_multi_task_view_view_branch_still_renders_multi_view_template(
    client, user_factory, ibex_image_factory
):
    user = user_factory(username="p1_view_branch")
    image_a = ibex_image_factory(owner=user, name="a")
    image_b = ibex_image_factory(owner=user, name="b")
    client.force_login(user)

    response = client.post(
        reverse("multi-task"),
        {
            "selected-files": f"{image_a.id},{image_b.id}",
            "task": "view",
            "next_id_index": "0",
        },
    )

    assert response.status_code == 200
    assert "core/multi_view.html" in [t.name for t in response.templates]
    assert response.context["image"] == image_a
    assert (
        response.context["selected_img_ids"]
        == f"{image_a.id},{image_b.id}"
    )


@pytest.mark.django_db
def test_multi_task_view_landmark_branch_still_renders_landmark_template(
    client, user_factory, ibex_image_factory
):
    user = user_factory(username="p1_landmark_branch")
    image = ibex_image_factory(owner=user)
    client.force_login(user)

    response = client.post(
        reverse("multi-task"),
        {
            "selected-files": str(image.id),
            "task": "landmark",
            "next_id_index": "0",
        },
    )

    assert response.status_code == 200
    assert "simple_landmarks/multi_landmarking.html" in [
        t.name for t in response.templates
    ]


@pytest.mark.django_db
def test_multi_task_view_locate_branch_still_renders_location_template(
    client, user_factory, ibex_image_factory, location_factory, region_factory
):
    user = user_factory(username="p1_locate_branch")
    loc = location_factory(latitude=46.0, longitude=8.0)
    image = ibex_image_factory(owner=user, location=loc)
    region_factory(owner=user)
    client.force_login(user)

    response = client.post(
        reverse("multi-task"),
        {
            "selected-files": str(image.id),
            "task": "locate",
            "next_id_index": "0",
        },
    )

    assert response.status_code == 200
    assert "core/multi_location_create.html" in [
        t.name for t in response.templates
    ]
    assert response.context["location_id"] == loc.id


@pytest.mark.django_db
@pytest.mark.parametrize("idx,exp_next_idx", [("0", 1), ("1", None)])
def test_multi_task_view_next_id_index_pagination_unchanged(
    client, user_factory, ibex_image_factory, idx, exp_next_idx
):
    user = user_factory(username=f"p1_pagination_{idx}")
    image_a = ibex_image_factory(owner=user, name="a")
    image_b = ibex_image_factory(owner=user, name="b")
    client.force_login(user)

    response = client.post(
        reverse("multi-task"),
        {
            "selected-files": f"{image_a.id},{image_b.id}",
            "task": "view",
            "next_id_index": idx,
        },
    )

    assert response.context["current_id_index"] == int(idx)
    assert response.context["next_id_index"] == exp_next_idx


# ---------------------------------------------------------------------------
# P2 -- counter-input pins (pre-existing behavior, explicitly NOT being fixed)
# ---------------------------------------------------------------------------


@pytest.mark.django_db
def test_multi_task_view_delete_with_missing_selected_files_still_raises_index_error(
    client, user_factory
):
    """Pins current pre-fix behavior. views.py:832-833 parses selected-files
    before the new delete branch is reached, so R1's guarantee holds only
    for well-formed submissions. Out of scope for this CR -- see
    docs/security-remediation-plan.md."""
    user = user_factory(username="p2_missing_selected_files")
    client.force_login(user)

    with pytest.raises(IndexError):
        client.post(reverse("multi-task"), {"task": "delete"})


@pytest.mark.django_db
def test_multi_task_view_delete_with_empty_selected_files_raises_value_error(
    client, user_factory
):
    """Pins current pre-fix behavior. views.py:832-833 parses selected-files
    before the new delete branch is reached, so R1's guarantee holds only
    for well-formed submissions. Out of scope for this CR -- see
    docs/security-remediation-plan.md."""
    user = user_factory(username="p2_empty_selected_files")
    client.force_login(user)

    with pytest.raises(ValueError):
        client.post(
            reverse("multi-task"), {"selected-files": "", "task": "delete"}
        )


@pytest.mark.django_db
def test_multi_task_view_delete_with_non_numeric_selected_files_raises_value_error(
    client, user_factory
):
    """Pins current pre-fix behavior. views.py:832-833 parses selected-files
    before the new delete branch is reached, so R1's guarantee holds only
    for well-formed submissions. Out of scope for this CR -- see
    docs/security-remediation-plan.md."""
    user = user_factory(username="p2_non_numeric_selected_files")
    client.force_login(user)

    with pytest.raises(ValueError):
        client.post(
            reverse("multi-task"),
            {"selected-files": "abc", "task": "delete"},
        )


@pytest.mark.django_db
@pytest.mark.parametrize(
    "task_value",
    ["Delete", "DELETE", " delete", "delete ", "del", "deleted"],
)
def test_multi_task_view_task_not_exactly_delete_falls_through_to_default_branch(
    client, user_factory, ibex_image_factory, task_value
):
    """Pins current pre-fix behavior. views.py:832-833 parses selected-files
    before the new delete branch is reached, so R1's guarantee holds only
    for well-formed submissions. Out of scope for this CR -- see
    docs/security-remediation-plan.md.

    Proves the equality check is exact, not loosened to
    case-insensitive/substring matching."""
    user = user_factory(username=f"p2_task_variant_{len(task_value)}")
    image = ibex_image_factory(owner=user)
    client.force_login(user)

    with pytest.raises(TypeError):
        client.post(
            reverse("multi-task"),
            {"selected-files": str(image.id), "task": task_value},
        )


@pytest.mark.django_db
def test_multi_task_view_missing_task_falls_through_and_raises_type_error(
    client, user_factory, ibex_image_factory
):
    """Pins current pre-fix behavior. views.py:832-833 parses selected-files
    before the new delete branch is reached, so R1's guarantee holds only
    for well-formed submissions. Out of scope for this CR -- see
    docs/security-remediation-plan.md."""
    user = user_factory(username="p2_missing_task")
    image = ibex_image_factory(owner=user)
    client.force_login(user)

    with pytest.raises(TypeError):
        client.post(
            reverse("multi-task"), {"selected-files": str(image.id)}
        )


@pytest.mark.django_db
def test_multi_task_view_delete_ignores_extra_selected_files_fields(
    client, user_factory, ibex_image_factory
):
    """Pins current pre-fix behavior. views.py:832-833 parses selected-files
    before the new delete branch is reached, so R1's guarantee holds only
    for well-formed submissions. Out of scope for this CR -- see
    docs/security-remediation-plan.md.

    Pins "first field wins" via getlist(...)[0]."""
    user = user_factory(username="p2_extra_selected_files_fields")
    image_a = ibex_image_factory(owner=user, name="a")
    image_b = ibex_image_factory(owner=user, name="b")
    client.force_login(user)

    response = client.post(
        reverse("multi-task"),
        {"selected-files": [f"{image_a.id},{image_b.id}", "9999"], "task": "delete"},
    )

    assert response.status_code == 302
