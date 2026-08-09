"""P0 tests for scripts/db_restore_drill.py: EXPECTED_TABLES parity,
preflight_source, collect_expected.

Uses duck-typed FakeConnection/FakeCursor (tests/scripts/conftest.py) --
no real DB required, mirrors tests/conftest.py's RegionStub/ChipStub
precedent.
"""

from __future__ import annotations

import re

import pytest

from core.models import Animal, Embedding, IbexChip, IbexImage, Location, Region
from scripts.db_restore_drill import (
    EXPECTED_TABLES,
    collect_expected,
    preflight_source,
)

pytestmark = pytest.mark.spec(
    ref="docs/security-remediation-plan.md#gate-restore-drill-required"
)


# ---------------------------------------------------------------------------
# EXPECTED_TABLES parity
# ---------------------------------------------------------------------------
def test_expected_tables_equals_model_db_table_names():
    assert tuple(
        m._meta.db_table
        for m in (Animal, Region, Location, IbexImage, IbexChip, Embedding)
    ) == EXPECTED_TABLES


def test_expected_tables_literal_pin():
    assert EXPECTED_TABLES == (
        "core_animal",
        "core_region",
        "core_location",
        "core_ibeximage",
        "core_ibexchip",
        "core_embedding",
    )


def test_expected_tables_constant_hygiene():
    assert len(EXPECTED_TABLES) == len(set(EXPECTED_TABLES)) == 6
    for name in EXPECTED_TABLES:
        assert re.match(r"^[a-z][a-z0-9_]*$", name)


# ---------------------------------------------------------------------------
# preflight_source
# ---------------------------------------------------------------------------
_SOURCE_DSN = "postgresql://alice:pw@prod-host.example.invalid:5432/webibex"


def _full_tables_rows():
    return [(t,) for t in EXPECTED_TABLES]


def test_preflight_source_read_only_connect_kwargs_asserted_exactly(
    monkeypatch, fake_connection_cls, fake_cursor_cls
):
    import scripts.db_restore_drill as mod

    cursor = fake_cursor_cls(
        fetchone_results=[(160000,)],  # SHOW server_version_num -> pg16
        fetchall_results=[_full_tables_rows()],
    )
    conn = fake_connection_cls(cursor)
    connect_calls = []

    def fake_connect(*args, **kwargs):
        connect_calls.append((args, kwargs))
        return conn

    monkeypatch.setattr(mod.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(mod, "_pg_dump_major_version", lambda path: 16)
    monkeypatch.setattr(mod.psycopg2, "connect", fake_connect)

    preflight_source(_SOURCE_DSN)

    assert len(connect_calls) == 1
    args, kwargs = connect_calls[0]
    assert args[0] == _SOURCE_DSN
    assert kwargs["options"] == "-c default_transaction_read_only=on"
    assert "connect_timeout" in kwargs
    # Security regression guard: _connect_readonly must not silently fall
    # back to libpq's own sslmode=prefer default (which downgrades to
    # plaintext on a failed TLS handshake, e.g. a MITM) -- it must pass an
    # explicit sslmode, normalized the same way libpq_env() normalizes it
    # for the pg_dump/pg_restore subprocess paths.
    assert kwargs["sslmode"] == "require"


def test_connect_readonly_rejects_sslmode_disable_on_source_dsn():
    """`_connect_readonly` must apply the same sslmode=disable rejection as
    libpq_env() -- refusing before any connection attempt, not just for the
    subprocess-based dump/restore paths.
    """
    from scripts.db_restore_drill import _connect_readonly

    dsn = "postgresql://alice:pw@prod-host.example.invalid:5432/webibex?sslmode=disable"
    with pytest.raises(ValueError, match="sslmode=disable"):
        _connect_readonly(dsn)


def test_connect_readonly_passes_through_explicit_non_disable_sslmode(monkeypatch):
    """A DSN with an explicit, non-disable sslmode (e.g. verify-full) is
    passed through as-is, not silently overridden to "require".
    """
    import scripts.db_restore_drill as mod
    from scripts.db_restore_drill import _connect_readonly

    connect_calls = []

    def fake_connect(*args, **kwargs):
        connect_calls.append((args, kwargs))
        return object()

    monkeypatch.setattr(mod.psycopg2, "connect", fake_connect)

    dsn = "postgresql://alice:pw@prod-host.example.invalid:5432/webibex?sslmode=verify-full"
    _connect_readonly(dsn)

    assert connect_calls[0][1]["sslmode"] == "verify-full"


def test_preflight_source_missing_table_raises(
    monkeypatch, fake_connection_cls, fake_cursor_cls
):
    import scripts.db_restore_drill as mod

    incomplete_rows = [(t,) for t in EXPECTED_TABLES[:-1]]  # missing core_embedding
    cursor = fake_cursor_cls(
        fetchone_results=[(160000,)], fetchall_results=[incomplete_rows]
    )
    conn = fake_connection_cls(cursor)

    monkeypatch.setattr(mod.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(mod, "_pg_dump_major_version", lambda path: 16)
    monkeypatch.setattr(mod.psycopg2, "connect", lambda *a, **k: conn)

    with pytest.raises(RuntimeError, match="core_embedding"):
        preflight_source(_SOURCE_DSN)


@pytest.mark.parametrize(
    "server_major,pg_dump_major,expect_raise",
    [
        (16, 17, False),  # local newer -- fine
        (16, 16, False),  # equal majors -- boundary, fine
        (17, 16, True),  # local older -- must raise
    ],
)
def test_preflight_source_version_gate(
    monkeypatch,
    fake_connection_cls,
    fake_cursor_cls,
    server_major,
    pg_dump_major,
    expect_raise,
):
    import scripts.db_restore_drill as mod

    cursor = fake_cursor_cls(
        fetchone_results=[(server_major * 10000,)],
        fetchall_results=[_full_tables_rows()],
    )
    conn = fake_connection_cls(cursor)

    monkeypatch.setattr(mod.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(mod, "_pg_dump_major_version", lambda path: pg_dump_major)
    monkeypatch.setattr(mod.psycopg2, "connect", lambda *a, **k: conn)

    if expect_raise:
        with pytest.raises(RuntimeError):
            preflight_source(_SOURCE_DSN)
    else:
        info = preflight_source(_SOURCE_DSN)
        assert info.server_major_version == server_major


def test_preflight_source_pg_dump_absent_gives_actionable_message():
    # pg_dump genuinely isn't installed in this sandbox -- exercises the
    # real shutil.which(None) branch, no monkeypatching needed.
    with pytest.raises(RuntimeError, match="pg_dump"):
        preflight_source(_SOURCE_DSN)


def test_preflight_source_driver_error_message_is_redacted(monkeypatch):
    import psycopg2

    import scripts.db_restore_drill as mod

    def raising_connect(*args, **kwargs):
        raise psycopg2.OperationalError(
            f"could not connect: {_SOURCE_DSN}"  # simulate a leaky driver message
        )

    monkeypatch.setattr(mod.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(mod, "_pg_dump_major_version", lambda path: 16)
    monkeypatch.setattr(mod.psycopg2, "connect", raising_connect)

    with pytest.raises(RuntimeError) as exc_info:
        preflight_source(_SOURCE_DSN)
    assert "pw" not in str(exc_info.value)
    assert "alice" not in str(exc_info.value)


# ---------------------------------------------------------------------------
# collect_expected
# ---------------------------------------------------------------------------
def test_collect_expected_six_count_queries_in_expected_tables_order(
    fake_connection_cls, fake_cursor_cls
):
    cursor = fake_cursor_cls(
        fetchone_results=[(i,) for i in range(1, 7)]
        + [("PNGP24_001", "Bob", True, "2024-01-01", "24AB", "M")],
    )
    conn = fake_connection_cls(cursor)

    collect_expected(conn)

    count_calls = cursor.execute_calls[:6]
    assert len(count_calls) == 6
    for (query, _params), table in zip(count_calls, EXPECTED_TABLES, strict=True):
        assert "COUNT(*)" in str(query)
        assert table in str(query)


def test_collect_expected_default_spot_row_query_shape(
    fake_connection_cls, fake_cursor_cls
):
    cursor = fake_cursor_cls(
        fetchone_results=[(i,) for i in range(1, 7)]
        + [("PNGP24_001", "Bob", True, "2024-01-01", "24AB", "M")],
    )
    conn = fake_connection_cls(cursor)

    collect_expected(conn)

    spot_query, spot_params = cursor.execute_calls[-1]
    spot_query_str = str(spot_query)
    assert "id_code" in spot_query_str
    assert "name" in spot_query_str
    assert "marked" in spot_query_str
    assert "capture_date" in spot_query_str
    assert "cohort" in spot_query_str
    assert "sex" in spot_query_str
    assert "IS NOT NULL" in spot_query_str
    assert "ORDER BY" in spot_query_str
    assert "LIMIT 1" in spot_query_str
    assert spot_params is None


def test_collect_expected_empty_spot_check_domain_fails_loudly(
    fake_connection_cls, fake_cursor_cls
):
    cursor = fake_cursor_cls(
        fetchone_results=[(i,) for i in range(1, 7)] + [None],
    )
    conn = fake_connection_cls(cursor)

    with pytest.raises(RuntimeError):
        collect_expected(conn)


@pytest.mark.parametrize(
    "malicious_id_code",
    [
        "PNGP24_001'; DROP TABLE core_animal; --",
        "' OR '1'='1",
    ],
)
def test_collect_expected_spot_id_code_bound_parameter_not_interpolated(
    fake_connection_cls, fake_cursor_cls, malicious_id_code
):
    cursor = fake_cursor_cls(
        fetchone_results=[(i,) for i in range(1, 7)] + [None],
    )
    conn = fake_connection_cls(cursor)

    with pytest.raises(RuntimeError):
        collect_expected(conn, spot_id_code=malicious_id_code)

    spot_query, spot_params = cursor.execute_calls[-1]
    assert malicious_id_code not in str(spot_query)
    assert spot_params == (malicious_id_code,)
