"""T20-T27: Django admin / django-filer routing+rendering smoke tests.

NOT staleness/collectstatic-drift oracles -- see test_static_assets_
collectstatic.py for that. These guard against the static-asset refresh
(R1) breaking request handling: routing and view rendering don't depend on
admin CSS/JS byte content, so these tests should already be GREEN before the
refresh and stay GREEN after it (R2's "no functional regression" evidence).

Route note (empirically verified against the installed django-filer==3.3.0
package and core/middleware.py -- see filer/urls.py and filer/settings.py:
FILER_CANONICAL_URL defaults to 'canonical/'): filer.urls, mounted at
webibex/urls.py's `path("filer/", include("filer.urls"))`, has exactly one
route (`canonical/<int:uploaded_at>/<int:file_id>/`) -- no root route. The
real filer admin UI lives under the Django admin site itself, at
reverse("admin:filer_folder_changelist") == "/webibex/filer/folder/", gated
by core.middleware.RedirectToUserFolderMiddleware (superusers pass through,
other staff get redirected to their own `<username>_files` folder).
"""

import pytest
from django.urls import reverse


# T20 -------------------------------------------------------------------
@pytest.mark.django_db
def test_anon_admin_login_returns_200(client):
    """django_db required: admin's LoginView renders get_current_site(),
    which queries django.contrib.sites' Site model even for an anonymous
    GET (allauth/sites integration), not because this test itself creates
    any data."""
    response = client.get(reverse("admin:login"))

    assert response.status_code == 200


# T21 -------------------------------------------------------------------
def test_anon_admin_index_redirects_to_login(client):
    response = client.get(reverse("admin:index"))

    assert response.status_code == 302


# T22 -------------------------------------------------------------------
@pytest.mark.django_db
def test_superuser_admin_index_returns_200(client, user_factory):
    superuser = user_factory(username="root", is_staff=True, is_superuser=True)
    client.force_login(superuser)

    response = client.get(reverse("admin:index"))

    assert response.status_code == 200


# T23 -------------------------------------------------------------------
@pytest.mark.django_db
def test_superuser_filer_folder_changelist_returns_200(client, user_factory):
    superuser = user_factory(username="root2", is_staff=True, is_superuser=True)
    client.force_login(superuser)

    response = client.get(reverse("admin:filer_folder_changelist"))

    assert response.status_code == 200


# T24 -------------------------------------------------------------------
@pytest.mark.django_db
def test_non_superuser_staff_filer_folder_changelist_redirects(client, user_factory):
    """Mirrors test_middleware.py's T34 pattern: non-superuser staff is
    redirected to their own folder by RedirectToUserFolderMiddleware.
    Deliberately does NOT follow the redirect -- following it may hit a
    separate, unrelated permission check that isn't this test's invariant.
    """
    staff_user = user_factory(username="staffer", is_staff=True, is_superuser=False)
    client.force_login(staff_user)

    response = client.get(reverse("admin:filer_folder_changelist"))

    assert response.status_code == 302


# T25 -------------------------------------------------------------------
def test_anon_bare_filer_path_returns_404_not_500(client):
    """Empirically confirmed: django-filer 3.3.0's filer.urls has no root
    route (only 'canonical/<int:uploaded_at>/<int:file_id>/'), so a bare
    GET /filer/ matches nothing and 404s -- it must not 500."""
    response = client.get("/filer/")

    assert response.status_code == 404


# T26 -------------------------------------------------------------------
@pytest.mark.django_db
def test_authenticated_non_staff_admin_index_redirects(client, user_factory):
    plain_user = user_factory()
    client.force_login(plain_user)

    response = client.get(reverse("admin:index"))

    assert response.status_code == 302


# T27 -------------------------------------------------------------------
@pytest.mark.django_db
@pytest.mark.parametrize(
    "path, role",
    [
        (reverse("admin:login"), "anon"),
        (reverse("admin:index"), "anon"),
        (reverse("admin:index"), "superuser"),
        (reverse("admin:filer_folder_changelist"), "superuser"),
        (reverse("admin:filer_folder_changelist"), "staff"),
        ("/filer/", "anon"),
        (reverse("admin:index"), "plain_user"),
    ],
)
def test_no_5xx_across_path_role_combinations(client, user_factory, path, role):
    """R2/R3 consolidated invariant: the static-asset refresh must not turn
    any of these routes into a server error, regardless of caller role."""
    if role == "superuser":
        user = user_factory(
            username=f"su_{path}".replace("/", "_"), is_staff=True, is_superuser=True
        )
        client.force_login(user)
    elif role == "staff":
        user = user_factory(
            username=f"staff_{path}".replace("/", "_"),
            is_staff=True,
            is_superuser=False,
        )
        client.force_login(user)
    elif role == "plain_user":
        user = user_factory(username=f"plain_{path}".replace("/", "_"))
        client.force_login(user)
    # role == "anon": no login

    response = client.get(path)

    assert response.status_code < 500
