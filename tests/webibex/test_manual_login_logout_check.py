"""Permanent regression tests that the real allauth login/logout/password-change
flows still work after the SECURE_* hardening change in webibex/settings.py --
both under the ambient ENVIRONMENT=test (unaffected code path) and under a
simulated production environment (the new SECURE_* gate active via
override_settings, which -- unlike importlib.reload -- correctly patches
django.conf.settings so the test client and middleware actually observe it).

Password change specifically exercises CSRF_COOKIE_SECURE on an
authenticated POST -- the one gap login (unauthenticated CSRF) and logout
(no meaningful form body) don't cover.

Both flows covered here (login/logout and password-change) were also
verified live in a real browser session on 2026-07-25, in addition to the
automated coverage in this file.
"""

from django.test import Client, override_settings
from django.urls import reverse


def test_real_login_then_logout_flow_under_ambient_test_environment(user_factory):
    user_factory(username="loginchecker", password="test-pass-12345")
    client = Client()

    login_page = client.get(reverse("account_login"))
    assert login_page.status_code == 200

    # ACCOUNT_AUTHENTICATION_METHOD = "email" (webibex/settings.py) -- the
    # "login" field must be the email address, not the username.
    login_response = client.post(
        reverse("account_login"),
        {"login": "loginchecker@example.invalid", "password": "test-pass-12345"},
        follow=True,
    )
    assert login_response.status_code == 200
    assert login_response.context["user"].is_authenticated

    protected = client.get(reverse("animals"))
    assert protected.status_code == 200

    logout_response = client.post(reverse("account_logout"), follow=True)
    assert logout_response.status_code == 200
    assert not logout_response.context["user"].is_authenticated

    protected_after_logout = client.get(reverse("animals"))
    assert protected_after_logout.status_code == 302
    assert "login" in protected_after_logout["Location"]


@override_settings(
    SESSION_COOKIE_SECURE=True,
    CSRF_COOKIE_SECURE=True,
    SECURE_SSL_REDIRECT=True,
    SECURE_PROXY_SSL_HEADER=("HTTP_X_FORWARDED_PROTO", "https"),
    SECURE_HSTS_SECONDS=3600,
)
def test_real_login_then_logout_flow_under_simulated_production_settings(user_factory):
    """Same flow, but with the new hardening settings actually active.

    override_settings (unlike importlib.reload) patches django.conf.settings
    for real, so SecurityMiddleware/SessionMiddleware/CsrfViewMiddleware all
    observe the change. secure=True on each request sets
    request.META['wsgi.url_scheme']='https', satisfying SECURE_SSL_REDIRECT
    without the test itself hitting a 301 redirect loop (mirrors what
    Railway's edge guarantees in real traffic).
    """
    user_factory(username="loginchecker2", password="test-pass-12345")
    client = Client()

    login_page = client.get(reverse("account_login"), secure=True)
    assert login_page.status_code == 200

    login_response = client.post(
        reverse("account_login"),
        {"login": "loginchecker2@example.invalid", "password": "test-pass-12345"},
        follow=True,
        secure=True,
    )
    assert login_response.status_code == 200
    assert login_response.context["user"].is_authenticated

    session_cookie = client.cookies.get("sessionid")
    assert session_cookie is not None
    assert session_cookie["secure"]

    protected = client.get(reverse("animals"), secure=True)
    assert protected.status_code == 200

    logout_response = client.post(reverse("account_logout"), follow=True, secure=True)
    assert logout_response.status_code == 200
    assert not logout_response.context["user"].is_authenticated

    protected_after_logout = client.get(reverse("animals"), secure=True)
    assert protected_after_logout.status_code == 302
    assert "login" in protected_after_logout["Location"]


def test_real_login_then_password_change_flow_under_ambient_test_environment(user_factory):
    """Authenticated POST via allauth's ChangePasswordForm -- exercises
    CSRF_COOKIE_SECURE under a live session, unlike the unauthenticated
    login POST or the (usually form-less) logout POST above.
    """
    user_factory(username="pwchecker", password="Old-Pass-12345")
    client = Client()

    client.post(
        reverse("account_login"),
        {"login": "pwchecker@example.invalid", "password": "Old-Pass-12345"},
        follow=True,
    )

    change_page = client.get(reverse("account_change_password"))
    assert change_page.status_code == 200

    # allauth's ChangePasswordForm fields: oldpassword, password1, password2.
    change_response = client.post(
        reverse("account_change_password"),
        {
            "oldpassword": "Old-Pass-12345",
            "password1": "New-Pass-67890",
            "password2": "New-Pass-67890",
        },
        follow=True,
    )
    assert change_response.status_code == 200
    # a successful change keeps the session valid (update_session_auth_hash),
    # it does not force a re-login.
    assert change_response.context["user"].is_authenticated
    assert change_response.context["user"].username == "pwchecker"

    client.post(reverse("account_logout"), follow=True)

    relogin_with_old_password = client.post(
        reverse("account_login"),
        {"login": "pwchecker@example.invalid", "password": "Old-Pass-12345"},
        follow=True,
    )
    assert not relogin_with_old_password.context["user"].is_authenticated

    relogin_with_new_password = client.post(
        reverse("account_login"),
        {"login": "pwchecker@example.invalid", "password": "New-Pass-67890"},
        follow=True,
    )
    assert relogin_with_new_password.context["user"].is_authenticated


@override_settings(
    SESSION_COOKIE_SECURE=True,
    CSRF_COOKIE_SECURE=True,
    SECURE_SSL_REDIRECT=True,
    SECURE_PROXY_SSL_HEADER=("HTTP_X_FORWARDED_PROTO", "https"),
    SECURE_HSTS_SECONDS=3600,
)
def test_real_login_then_password_change_flow_under_simulated_production_settings(
    user_factory,
):
    """Same flow, with the new hardening settings actually active -- the
    CSRF_COOKIE_SECURE case this whole file exists to cover: a Secure-flagged
    CSRF cookie must still round-trip correctly on an authenticated POST.
    """
    user_factory(username="pwchecker2", password="Old-Pass-12345")
    client = Client()

    client.post(
        reverse("account_login"),
        {"login": "pwchecker2@example.invalid", "password": "Old-Pass-12345"},
        follow=True,
        secure=True,
    )

    change_page = client.get(reverse("account_change_password"), secure=True)
    assert change_page.status_code == 200

    change_response = client.post(
        reverse("account_change_password"),
        {
            "oldpassword": "Old-Pass-12345",
            "password1": "New-Pass-67890",
            "password2": "New-Pass-67890",
        },
        follow=True,
        secure=True,
    )
    assert change_response.status_code == 200
    assert change_response.context["user"].is_authenticated
    assert change_response.context["user"].username == "pwchecker2"

    client.post(reverse("account_logout"), follow=True, secure=True)

    relogin_with_new_password = client.post(
        reverse("account_login"),
        {"login": "pwchecker2@example.invalid", "password": "New-Pass-67890"},
        follow=True,
        secure=True,
    )
    assert relogin_with_new_password.context["user"].is_authenticated


@override_settings(
    SESSION_COOKIE_SECURE=True,
    CSRF_COOKIE_SECURE=True,
    SECURE_SSL_REDIRECT=True,
    SECURE_PROXY_SSL_HEADER=("HTTP_X_FORWARDED_PROTO", "https"),
    SECURE_HSTS_SECONDS=3600,
)
def test_plain_http_request_redirects_to_https_under_simulated_production_settings():
    """SECURE_SSL_REDIRECT's actual redirect behavior. Every other
    production-simulation test in this file passes secure=True, which makes
    the request already look secure to Django and never exercises the
    redirect path itself.
    """
    client = Client()

    response = client.get(reverse("account_login"))

    assert response.status_code == 301
    assert response.url.startswith("https://")


@override_settings(
    SESSION_COOKIE_SECURE=True,
    CSRF_COOKIE_SECURE=True,
    SECURE_SSL_REDIRECT=True,
    SECURE_PROXY_SSL_HEADER=("HTTP_X_FORWARDED_PROTO", "https"),
    SECURE_HSTS_SECONDS=3600,
)
def test_strict_transport_security_header_present_on_secure_response(db):
    """Confirms SecurityMiddleware actually emits the HSTS header on a live
    response. test_settings_security_hardening.py only asserts
    SECURE_HSTS_SECONDS as a settings *value* via importlib.reload -- that
    module reload never runs SecurityMiddleware, so it can't prove the
    header itself shows up on a response.

    Needs `db`: rendering the login page (unlike the redirect test above,
    which returns before the view runs) touches the database via allauth's
    account_login view.
    """
    client = Client()

    response = client.get(reverse("account_login"), secure=True)

    assert response.status_code == 200
    assert response.headers["Strict-Transport-Security"] == "max-age=3600"


def test_password_change_rejects_post_with_invalid_csrf_token(user_factory):
    """The reject-on-invalid-token path for an authenticated POST.
    test_real_login_then_password_change_flow_under_ambient_test_environment
    only covers the happy path (valid CSRF, via the default Client's
    disabled CSRF enforcement).
    """
    user_factory(username="csrfchecker", password="Old-Pass-12345")
    login_client = Client()
    login_response = login_client.post(
        reverse("account_login"),
        {"login": "csrfchecker@example.invalid", "password": "Old-Pass-12345"},
        follow=True,
    )
    assert login_response.context["user"].is_authenticated

    strict_client = Client(enforce_csrf_checks=True)
    strict_client.cookies["sessionid"] = login_client.cookies["sessionid"].value
    # CsrfViewMiddleware rejects an invalid token before the view runs, so a
    # bad-token POST would 403 regardless of auth state -- this GET proves
    # the copied session cookie actually authenticated strict_client, so the
    # 403 below is a true "authenticated POST, bad CSRF token" rejection.
    change_page = strict_client.get(reverse("account_change_password"))
    assert change_page.status_code == 200

    response = strict_client.post(
        reverse("account_change_password"),
        {
            "oldpassword": "Old-Pass-12345",
            "password1": "New-Pass-67890",
            "password2": "New-Pass-67890",
            "csrfmiddlewaretoken": "invalid-token-value",
        },
    )

    assert response.status_code == 403
