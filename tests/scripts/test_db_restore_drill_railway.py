"""P0 tests for scripts/db_restore_drill.py: fetch_database_url.

Uses the `mock_railway_api` fixture (tests/scripts/conftest.py), which
reconfigures the existing no_network `post_patch` -- never double-patches
requests.post.
"""

from __future__ import annotations

import logging

import pytest

from scripts.db_restore_drill import fetch_database_url

pytestmark = pytest.mark.spec(
    ref="docs/security-remediation-plan.md#gate-restore-drill-required"
)


class _FakeResponse:
    def __init__(self, status_code=200, json_body=None, raises_on_json=False):
        self.status_code = status_code
        self._json_body = json_body
        self._raises_on_json = raises_on_json

    def json(self):
        if self._raises_on_json:
            raise ValueError("not valid JSON")
        return self._json_body


def _ok_response(variables: dict) -> _FakeResponse:
    return _FakeResponse(status_code=200, json_body={"data": {"variables": variables}})


@pytest.mark.parametrize("token_kind", ["account", "project"])
def test_fetch_database_url_happy_path_both_token_kinds(
    monkeypatch, mock_railway_api, token_kind
):
    monkeypatch.setenv("RAILWAY_API_TOKEN", "sekrit-token")
    mock_railway_api.return_value = _ok_response({"DATABASE_URL": "postgresql://x"})

    result = fetch_database_url(
        "proj-1",
        "env-1",
        "svc-1",
        token_kind=token_kind,
        variable_name="DATABASE_URL",
    )

    assert result == "postgresql://x"
    assert mock_railway_api.call_count == 1
    _args, kwargs = mock_railway_api.call_args
    headers = kwargs["headers"]
    if token_kind == "account":
        assert headers["Authorization"] == "Bearer sekrit-token"
    else:
        assert headers["Project-Access-Token"] == "sekrit-token"


def test_fetch_database_url_invalid_token_kind_raises_before_network_call(
    monkeypatch, mock_railway_api
):
    monkeypatch.setenv("RAILWAY_API_TOKEN", "sekrit-token")
    with pytest.raises(ValueError):
        fetch_database_url(
            "proj-1",
            "env-1",
            None,
            token_kind="bogus",
            variable_name="DATABASE_URL",
        )
    assert mock_railway_api.call_count == 0


def test_fetch_database_url_http_error_status_raises(monkeypatch, mock_railway_api):
    monkeypatch.setenv("RAILWAY_API_TOKEN", "sekrit-token")
    mock_railway_api.return_value = _FakeResponse(status_code=401, json_body={})
    with pytest.raises(RuntimeError):
        fetch_database_url(
            "proj-1", "env-1", None, token_kind="account", variable_name="DATABASE_URL"
        )


def test_fetch_database_url_graphql_errors_array_on_200_extracts_message_only(
    monkeypatch, mock_railway_api, caplog
):
    monkeypatch.setenv("RAILWAY_API_TOKEN", "sekrit-token")
    mock_railway_api.return_value = _FakeResponse(
        status_code=200,
        json_body={
            "data": {"variables": {"SECRET_FIELD": "sentinel-should-never-log"}},
            "errors": [{"message": "projectId not found"}],
        },
    )
    with (
        caplog.at_level(logging.ERROR),
        pytest.raises(RuntimeError, match="projectId not found"),
    ):
        fetch_database_url(
            "proj-1",
            "env-1",
            None,
            token_kind="account",
            variable_name="DATABASE_URL",
        )

    for record in caplog.records:
        assert "sentinel-should-never-log" not in record.getMessage()
        assert "sentinel-should-never-log" not in repr(record.args)


def test_fetch_database_url_requested_variable_absent_lists_names_not_values(
    monkeypatch, mock_railway_api
):
    monkeypatch.setenv("RAILWAY_API_TOKEN", "sekrit-token")
    mock_railway_api.return_value = _ok_response(
        {"OTHER_VAR": "should-not-appear-in-error"}
    )
    with pytest.raises(RuntimeError) as exc_info:
        fetch_database_url(
            "proj-1",
            "env-1",
            None,
            token_kind="account",
            variable_name="DATABASE_URL",
        )
    message = str(exc_info.value)
    assert "OTHER_VAR" in message
    assert "should-not-appear-in-error" not in message


@pytest.mark.parametrize(
    "response",
    [
        _FakeResponse(status_code=200, raises_on_json=True),  # non-JSON
        _FakeResponse(status_code=200, json_body={"data": None}),  # data: null
        _FakeResponse(
            status_code=200,
            json_body={"data": {"variables": {"DATABASE_URL": 12345}}},
        ),  # non-str value
    ],
)
def test_fetch_database_url_malformed_body_fails_closed(
    monkeypatch, mock_railway_api, response
):
    monkeypatch.setenv("RAILWAY_API_TOKEN", "sekrit-token")
    mock_railway_api.return_value = response
    with pytest.raises(RuntimeError):
        fetch_database_url(
            "proj-1",
            "env-1",
            None,
            token_kind="account",
            variable_name="DATABASE_URL",
        )


def test_fetch_database_url_explicit_timeout_present(monkeypatch, mock_railway_api):
    monkeypatch.setenv("RAILWAY_API_TOKEN", "sekrit-token")
    mock_railway_api.return_value = _ok_response({"DATABASE_URL": "postgresql://x"})
    fetch_database_url(
        "proj-1", "env-1", None, token_kind="account", variable_name="DATABASE_URL"
    )
    _args, kwargs = mock_railway_api.call_args
    assert "timeout" in kwargs
    assert kwargs["timeout"] is not None


def test_fetch_database_url_endpoint_substituted_into_url(
    monkeypatch, mock_railway_api
):
    monkeypatch.setenv("RAILWAY_API_TOKEN", "sekrit-token")
    mock_railway_api.return_value = _ok_response({"DATABASE_URL": "postgresql://x"})
    fetch_database_url(
        "proj-1",
        "env-1",
        None,
        token_kind="account",
        variable_name="DATABASE_URL",
        endpoint="custom.railway.example",
    )
    args, _kwargs = mock_railway_api.call_args
    called_url = args[0] if args else _kwargs.get("url")
    assert called_url == "https://custom.railway.example/graphql/v2"


def test_fetch_database_url_exactly_one_post_call(monkeypatch, mock_railway_api):
    monkeypatch.setenv("RAILWAY_API_TOKEN", "sekrit-token")
    mock_railway_api.return_value = _ok_response({"DATABASE_URL": "postgresql://x"})
    fetch_database_url(
        "proj-1", "env-1", None, token_kind="account", variable_name="DATABASE_URL"
    )
    assert mock_railway_api.call_count == 1
