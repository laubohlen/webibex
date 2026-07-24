"""T34-T35: core/middleware.py RedirectToUserFolderMiddleware."""

from unittest import mock

import pytest
from django.http import Http404
from django.test import RequestFactory
from django.urls import reverse
from filer.models import Folder

from core.middleware import RedirectToUserFolderMiddleware


# T34 -----------------------------------------------------------------------
@pytest.mark.django_db
def test_middleware_redirects_non_superuser_to_their_folder(user_factory):
    user = user_factory(username="alice")
    # created automatically by the create_user_folders post_save signal
    folder = Folder.objects.get(name="alice_files", owner=user)
    get_response = mock.Mock()
    middleware = RedirectToUserFolderMiddleware(get_response=get_response)
    request = RequestFactory().get(reverse("admin:filer_folder_changelist"))
    request.user = user

    response = middleware(request)

    assert response.status_code == 302
    assert str(folder.id) in response["Location"]
    get_response.assert_not_called()


# T35a ------------------------------------------------------------------
@pytest.mark.django_db
def test_middleware_passes_through_for_superuser_on_changelist(user_factory):
    superuser = user_factory(username="admin_user", is_superuser=True, is_staff=True)
    get_response = mock.Mock(return_value="sentinel-response")
    middleware = RedirectToUserFolderMiddleware(get_response=get_response)
    request = RequestFactory().get(reverse("admin:filer_folder_changelist"))
    request.user = superuser

    response = middleware(request)

    assert response == "sentinel-response"
    get_response.assert_called_once_with(request)


# T35b ------------------------------------------------------------------
@pytest.mark.django_db
def test_middleware_passes_through_for_non_changelist_path(user_factory):
    user = user_factory(username="bob")
    get_response = mock.Mock(return_value="sentinel-response")
    middleware = RedirectToUserFolderMiddleware(get_response=get_response)
    request = RequestFactory().get("/some/other/path/")
    request.user = user

    response = middleware(request)

    assert response == "sentinel-response"
    get_response.assert_called_once_with(request)


# T35c ------------------------------------------------------------------
@pytest.mark.django_db
def test_middleware_missing_folder_raises_http404_uncaught(user_factory):
    """Bonus bug (pinned, not fixed): `except Folder.DoesNotExist` does not
    catch what get_object_or_404 actually raises (django.http.Http404) when
    the user's main folder is missing."""
    user = user_factory(username="carol")
    Folder.objects.filter(name="carol_files", owner=user).delete()
    get_response = mock.Mock()
    middleware = RedirectToUserFolderMiddleware(get_response=get_response)
    request = RequestFactory().get(reverse("admin:filer_folder_changelist"))
    request.user = user

    with pytest.raises(Http404):
        middleware(request)
