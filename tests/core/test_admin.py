"""Tests for core/admin.py -- LocationAdmin.ibeximage_name and
CustomFolderAdmin.tag_left/tag_right/tag_other actions.

Scope: TEST-ONLY. core/admin.py is not modified. Pre-existing typos in the
tag_* message_user() strings ("we're" for "were", "taged" for "tagged") are
asserted verbatim -- fixing them is out of scope for this change.
"""

import pytest
from django.contrib import admin
from django.contrib.messages import get_messages
from django.contrib.messages.storage.fallback import FallbackStorage
from django.contrib.sessions.middleware import SessionMiddleware
from django.test import RequestFactory
from filer.models import Folder

from core.admin import CustomFolderAdmin, LocationAdmin
from core.models import IbexImage, Location


def _admin_request():
    """Build a POST request wired with a session and Django's messages
    framework -- required because ModelAdmin.message_user() writes to
    request._messages.
    """
    request = RequestFactory().post("/")
    SessionMiddleware(lambda r: None).process_request(request)
    request.session.save()
    request._messages = FallbackStorage(request)
    return request


# ---------------------------------------------------------------------------
# T01/T02 -- LocationAdmin.ibeximage_name
# ---------------------------------------------------------------------------
@pytest.mark.django_db
def test_ibeximage_name_with_related_image_returns_image_name(
    location_factory, ibex_image_factory
):
    loc = location_factory()
    image = ibex_image_factory(location=loc, name="test-ibex")
    # A post_save signal (core.signals.process_uploaded_image) renames the
    # file deterministically based on animal/exif data -- re-fetch the
    # actual persisted name rather than assuming the `name=` factory arg
    # survives unchanged.
    persisted_name = IbexImage.objects.get(pk=image.pk).name
    loc.refresh_from_db()

    result = LocationAdmin(Location, admin.site).ibeximage_name(loc)

    assert result == persisted_name


@pytest.mark.django_db
def test_ibeximage_name_without_related_image_returns_dash(location_factory):
    loc = location_factory()

    result = LocationAdmin(Location, admin.site).ibeximage_name(loc)

    assert result == "-"


# ---------------------------------------------------------------------------
# T03 -- CustomFolderAdmin.tag_left / tag_right / tag_other, folded matrix
# ---------------------------------------------------------------------------
@pytest.mark.django_db
@pytest.mark.parametrize(
    "method_name, expected_side, expected_word",
    [
        ("tag_left", "L", "left"),
        ("tag_right", "R", "right"),
        ("tag_other", "O", "other"),
    ],
)
@pytest.mark.parametrize("count", [0, 1, 3])
def test_tag_action_updates_side_and_messages_user(
    method_name, expected_side, expected_word, count, ibex_image_factory, user_factory
):
    # Share one owner across all created images -- ibex_image_factory()
    # defaults to a fresh user_factory() call per invocation, which would
    # collide on the unique username when count > 1.
    owner = user_factory()
    created_pks = [ibex_image_factory(owner=owner).pk for _ in range(count)]
    qs = IbexImage.objects.filter(pk__in=created_pks)
    req = _admin_request()
    admin_obj = CustomFolderAdmin(Folder, admin.site)

    ret = getattr(admin_obj, method_name)(req, qs, IbexImage.objects.none())

    assert ret is None
    for pk in created_pks:
        assert IbexImage.objects.get(pk=pk).side == expected_side

    messages = list(get_messages(req))
    assert len(messages) == 1
    assert (
        str(messages[0])
        == f"{count} images we're successfully taged '{expected_word}'"
    )
