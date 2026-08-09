"""P0 tests for scripts/db_restore_drill.py: redact_dsn, libpq_env,
_assert_local_target.

No DB, no network, no Docker -- pure-function / parsing-only scenarios.
"""

from __future__ import annotations

from urllib.parse import quote

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis import example

from scripts.db_restore_drill import (
    _assert_local_target,
    libpq_env,
    redact_dsn,
)

pytestmark = pytest.mark.spec(
    ref="docs/security-remediation-plan.md#gate-restore-drill-required"
)


# ---------------------------------------------------------------------------
# redact_dsn
# ---------------------------------------------------------------------------
def test_redact_dsn_happy_path_full_dsn():
    dsn = "postgresql://alice:s3cret@dbhost.example.invalid:5432/webibex"
    redacted = redact_dsn(dsn)
    assert redacted == "postgresql://<redacted>@dbhost.example.invalid:5432/webibex"
    assert "s3cret" not in redacted
    assert "alice" not in redacted


@pytest.mark.parametrize(
    "dsn,expected",
    [
        (
            "postgresql://alice:pw@dbhost/webibex",
            "postgresql://<redacted>@dbhost/webibex",
        ),  # no port
        (
            "postgresql://alice:pw@dbhost:5432",
            "postgresql://<redacted>@dbhost:5432",
        ),  # no path
        (
            "postgresql://dbhost:5432/webibex",
            "postgresql://dbhost:5432/webibex",
        ),  # no user/pw at all
        (
            "postgresql://alice:pw@dbhost:5432/webibex?sslmode=require",
            "postgresql://<redacted>@dbhost:5432/webibex",
        ),  # query string dropped
    ],
)
def test_redact_dsn_shape_variants(dsn, expected):
    assert redact_dsn(dsn) == expected


def test_redact_dsn_ipv6_bracketed_host_preserved():
    dsn = "postgresql://alice:pw@[::1]:5432/webibex"
    redacted = redact_dsn(dsn)
    assert redacted == "postgresql://<redacted>@[::1]:5432/webibex"


@pytest.mark.parametrize(
    "garbage",
    [
        "garbage",
        "",
        "not-a-dsn-at-all",
        "postgresql://alice:pw@dbhost:not-a-number/webibex",
    ],
)
def test_redact_dsn_garbage_input_raises_value_error_not_crash(garbage):
    with pytest.raises(ValueError):
        redact_dsn(garbage)


def test_redact_dsn_error_message_never_echoes_raw_credential_shaped_input():
    dsn = "postgresql://alice:supersecretpw@dbhost:notaport/webibex"
    with pytest.raises(ValueError) as exc_info:
        redact_dsn(dsn)
    assert "supersecretpw" not in str(exc_info.value)
    assert dsn not in str(exc_info.value)


_DSN_STATIC_TOKENS = frozenset({"user", "dbhost", "5432", "dbname", "postgresql"})


@given(
    pw=st.text(
        alphabet=st.characters(
            blacklist_characters="@:/%\n\r\x00", blacklist_categories=("Cs",)
        ),
        min_size=1,
        max_size=24,
    )
)
@example(pw="@")
@example(pw="%40")
@example(pw="")
@settings(deadline=None)
def test_redact_dsn_property_password_never_survives(pw):
    if pw in _DSN_STATIC_TOKENS:
        return
    dsn = f"postgresql://user:{pw}@dbhost:5432/dbname"
    try:
        redacted = redact_dsn(dsn)
    except ValueError:
        return  # malformed encodings are allowed to be rejected outright
    if pw and len(pw) > 1:
        # Single-char passwords (e.g. "@") can coincidentally collide with
        # the redaction format's own structural separator ("<redacted>@")
        # without that being a real credential leak -- only multi-char
        # substrings are a meaningful leak signal.
        assert pw not in redacted


# ---------------------------------------------------------------------------
# libpq_env
# ---------------------------------------------------------------------------
def test_libpq_env_full_dsn_exact_dict():
    dsn = "postgresql://alice:s3cret@dbhost:5432/webibex"
    assert libpq_env(dsn) == {
        "PGHOST": "dbhost",
        "PGPORT": "5432",
        "PGUSER": "alice",
        "PGPASSWORD": "s3cret",
        "PGDATABASE": "webibex",
        "PGSSLMODE": "require",
    }


@pytest.mark.parametrize(
    "raw,decoded",
    [
        ("s3%40cret", "s3@cret"),
        ("s3%3Acret", "s3:cret"),
        ("s3%2Fcret", "s3/cret"),
        ("s3%25cret", "s3%cret"),
    ],
)
def test_libpq_env_percent_decodes_password(raw, decoded):
    dsn = f"postgresql://alice:{raw}@dbhost:5432/webibex"
    assert libpq_env(dsn)["PGPASSWORD"] == decoded


@pytest.mark.parametrize(
    "dsn,missing_key",
    [
        ("postgresql://dbhost:5432/webibex", "PGUSER"),
        ("postgresql://dbhost:5432/webibex", "PGPASSWORD"),
        ("postgresql://alice@dbhost:5432/webibex", "PGPASSWORD"),
        ("postgresql://alice:pw@dbhost:5432", "PGDATABASE"),
        ("postgresql://alice:pw@dbhost/webibex", "PGPORT"),
    ],
)
def test_libpq_env_missing_component_key_omitted_not_empty_string(dsn, missing_key):
    result = libpq_env(dsn)
    assert missing_key not in result


def test_libpq_env_non_numeric_port_raises():
    with pytest.raises(ValueError):
        libpq_env("postgresql://alice:pw@dbhost:notaport/webibex")


def test_libpq_env_sslmode_defaults_to_require():
    result = libpq_env("postgresql://alice:pw@dbhost:5432/webibex")
    assert result["PGSSLMODE"] == "require"


@pytest.mark.parametrize("variant", ["disable", "DISABLE", "Disable"])
def test_libpq_env_explicit_sslmode_disable_rejected(variant):
    dsn = f"postgresql://alice:pw@dbhost:5432/webibex?sslmode={variant}"
    with pytest.raises(ValueError):
        libpq_env(dsn)


def test_libpq_env_explicit_non_disable_sslmode_honored():
    dsn = "postgresql://alice:pw@dbhost:5432/webibex?sslmode=verify-full"
    assert libpq_env(dsn)["PGSSLMODE"] == "verify-full"


def test_libpq_env_returns_fresh_dict_no_shared_state():
    dsn = "postgresql://alice:pw@dbhost:5432/webibex"
    first = libpq_env(dsn)
    first["PGHOST"] = "mutated"
    second = libpq_env(dsn)
    assert second["PGHOST"] == "dbhost"


@given(
    pw=st.text(
        alphabet=st.characters(
            blacklist_characters="@:/%\n\r\x00", blacklist_categories=("Cs",)
        ),
        min_size=0,
        max_size=24,
    )
)
@example(pw="%")
@example(pw="%zz")
@example(pw="100%")
@settings(deadline=None)
def test_libpq_env_password_round_trips_through_dsn(pw):
    quoted = quote(pw, safe="")
    dsn = f"postgresql://alice:{quoted}@dbhost:5432/webibex"
    result = libpq_env(dsn)
    if pw:
        assert result["PGPASSWORD"] == pw
    else:
        assert "PGPASSWORD" not in result


# ---------------------------------------------------------------------------
# _assert_local_target
# ---------------------------------------------------------------------------
_SOURCE_DSN = "postgresql://alice:pw@prod-host.example.invalid:5432/webibex"


@pytest.mark.parametrize(
    "host",
    ["127.0.0.1", "localhost", "::1", "127.0.0.2"],
)
def test_assert_local_target_accepts_loopback(host):
    target_host = f"[{host}]" if ":" in host else host
    target_dsn = f"postgresql://alice:pw@{target_host}:5432/scratch_db"
    _assert_local_target(target_dsn, _SOURCE_DSN)  # must not raise


@pytest.mark.parametrize(
    "host",
    [
        "127.0.0.1.evil.com",
        "localhost.attacker.tld",
        "2130706433",
        "0x7f000001",
        "10.0.0.5",
        "0.0.0.0",  # noqa: S104 -- test data asserting this is REJECTED, not a bind address
        "::",
    ],
)
def test_assert_local_target_rejects_non_loopback_including_bypass_shapes(host):
    target_dsn = f"postgresql://alice:pw@{host}:5432/scratch_db"
    with pytest.raises(ValueError):
        _assert_local_target(target_dsn, _SOURCE_DSN)


def test_assert_local_target_rejects_self_target_even_with_different_password():
    source_dsn = "postgresql://alice:pw1@localhost:5432/scratch_db"
    target_dsn = "postgresql://bob:pw2@localhost:5432/scratch_db"
    with pytest.raises(ValueError):
        _assert_local_target(target_dsn, source_dsn)


def test_assert_local_target_hostless_dsn_fails_closed():
    with pytest.raises(ValueError):
        _assert_local_target("not-a-dsn", _SOURCE_DSN)
    with pytest.raises(ValueError):
        _assert_local_target("", _SOURCE_DSN)


def test_assert_local_target_raised_message_is_redacted():
    target_dsn = "postgresql://alice:supersecretpw@10.0.0.5:5432/scratch_db"
    with pytest.raises(ValueError) as exc_info:
        _assert_local_target(target_dsn, _SOURCE_DSN)
    assert "supersecretpw" not in str(exc_info.value)
