"""Regression tests for the /health/ liveness + deployed-version probe
(core/views.py:health_view) -- see docs/security-remediation-plan.md.
"""

from django.urls import reverse


def test_health_view_returns_200_with_expected_shape(client):
    response = client.get(reverse("health"))

    assert response.status_code == 200
    assert response["Content-Type"] == "application/json"
    body = response.json()
    assert body["status"] == "ok"
    assert "commit" in body


def test_health_view_does_not_leak_django_version(client):
    # Regression guard: an unauthenticated framework-version leak makes CVE
    # targeting easier -- must never come back.
    response = client.get(reverse("health"))

    assert "django" not in response.json()


def test_health_view_falls_back_to_unknown_commit_without_railway_env(
    client, monkeypatch
):
    monkeypatch.delenv("RAILWAY_GIT_COMMIT_SHA", raising=False)

    response = client.get(reverse("health"))

    assert response.json()["commit"] == "unknown"


def test_health_view_reports_real_commit_when_railway_env_present(client, monkeypatch):
    monkeypatch.setenv("RAILWAY_GIT_COMMIT_SHA", "abc1234")

    response = client.get(reverse("health"))

    assert response.json()["commit"] == "abc1234"


def test_health_view_requires_no_authentication(client):
    # No client.force_login() call -- anonymous access must succeed, unlike
    # most other views in this app (login_required-gated).
    response = client.get(reverse("health"))

    assert response.status_code == 200
