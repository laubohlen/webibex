"""P0 tests for scripts/db_restore_drill.py: restore_and_verify (without a
real Docker/testcontainers install) and the pure comparison logic.

`testcontainers` is confirmed NOT installed in this sandbox -- that's
exercised for real in test_restore_and_verify_without_testcontainers_*.
Other tests inject a fake `testcontainers.postgres` module via
sys.modules so the guard-ordering / image-tag-validation logic is
testable without the real package.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest import mock

import pytest

import scripts.db_restore_drill as mod
from scripts.db_restore_drill import (
    ExpectedState,
    restore_and_verify,
)

pytestmark = pytest.mark.spec(
    ref="docs/security-remediation-plan.md#gate-restore-drill-required"
)


def test_importing_module_succeeds_without_testcontainers_installed():
    # `testcontainers` is confirmed not installed in this sandbox -- reload
    # to prove import-time module execution never touches it (must not
    # raise ImportError).
    import importlib

    reloaded = importlib.reload(mod)
    assert reloaded.restore_and_verify is not None


def test_restore_and_verify_without_testcontainers_gives_actionable_message(
    monkeypatch,
):
    # `testcontainers` the Python package IS installed in this sandbox's
    # venv now (confirmed) -- a real-absence test would silently stop
    # testing this branch. Force the ImportError path deterministically:
    # setting a sys.modules entry to None makes Python's import system
    # raise ImportError immediately for that module, regardless of
    # whether it's actually installed.
    monkeypatch.setitem(sys.modules, "testcontainers.postgres", None)
    monkeypatch.setitem(sys.modules, "testcontainers", None)

    expected = ExpectedState(counts={"core_animal": 1}, spot_row=("a", "b"))
    with pytest.raises(RuntimeError, match="testcontainers"):
        restore_and_verify(Path("dump.enc"), expected, 16)


_FAKE_CONTAINER_ID = "b" * 40


@pytest.fixture
def fake_testcontainers_module(monkeypatch):
    """Inject a fake testcontainers.postgres module so restore_and_verify's
    lazy import succeeds without the real dependency being installed.
    """
    fake_pkg = types.ModuleType("testcontainers")
    fake_postgres_mod = types.ModuleType("testcontainers.postgres")

    container_instance = mock.MagicMock()
    container_instance.get_connection_url.return_value = (
        "postgresql://test:test@127.0.0.1:55432/test"
    )
    container_instance.get_wrapped_container.return_value.id = _FAKE_CONTAINER_ID
    container_instance.__enter__ = mock.Mock(return_value=container_instance)
    container_instance.__exit__ = mock.Mock(return_value=False)

    container_cls = mock.Mock(return_value=container_instance)
    fake_postgres_mod.PostgresContainer = container_cls

    monkeypatch.setitem(sys.modules, "testcontainers", fake_pkg)
    monkeypatch.setitem(sys.modules, "testcontainers.postgres", fake_postgres_mod)
    return container_cls, container_instance


@pytest.mark.parametrize(
    "bad_version",
    [
        "16-alpine; rm -rf /",
        "latest",
        "",
        "16.2",
        "16\n",  # `$` in the old `^\d+$` pattern matches before a trailing
        # newline -- `.match()` accepted this. Must be rejected now.
        "１６",  # noqa: RUF001 -- fullwidth digits (intentional confusable): `\d` is Unicode-aware and accepted
        # these under the old pattern. Must be rejected now (ASCII-only).
    ],
)
def test_restore_and_verify_rejects_unsafe_prod_major_version(
    monkeypatch, fake_testcontainers_module, bad_version
):
    monkeypatch.setenv("DB_DUMP_PASSPHRASE", "pw")
    expected = ExpectedState(counts={"core_animal": 1}, spot_row=("a",))

    with pytest.raises(ValueError):
        restore_and_verify(Path("dump.enc"), expected, bad_version)

    container_cls, _instance = fake_testcontainers_module
    container_cls.assert_not_called()


def test_restore_and_verify_calls_assert_local_target_before_any_popen(
    monkeypatch, fake_testcontainers_module
):
    monkeypatch.setenv("DB_DUMP_PASSPHRASE", "pw")
    monkeypatch.setattr(mod.shutil, "which", lambda name: f"/usr/bin/{name}")

    popen_calls = []
    monkeypatch.setattr(
        mod.subprocess,
        "Popen",
        lambda *a, **k: popen_calls.append((a, k)) or mock.Mock(),
    )

    restore_target_env_calls = []
    original_restore_target_env = mod._restore_target_env

    def spy_restore_target_env(dsn):
        restore_target_env_calls.append(dsn)
        return original_restore_target_env(dsn)

    monkeypatch.setattr(mod, "_restore_target_env", spy_restore_target_env)

    def raising_guard(dsn, source_dsn=None):
        raise ValueError("guard tripped -- not a loopback target")

    monkeypatch.setattr(mod, "_assert_local_target", raising_guard)

    expected = ExpectedState(counts={"core_animal": 1}, spot_row=("a",))
    with pytest.raises(ValueError, match="guard tripped"):
        restore_and_verify(Path("dump.enc"), expected, 16)

    assert popen_calls == []
    # _restore_target_env (and, transitively, the container-id read) must
    # never run when the loopback-target guard has already tripped --
    # otherwise the guard would be validating a value manufactured AFTER
    # it already ran, making it vacuous.
    assert restore_target_env_calls == []


def test_restore_and_verify_happy_path_restore_leg_argv_env_and_verification_dsn(
    monkeypatch,
    fake_testcontainers_module,
    fake_popen_cls,
    fake_connection_cls,
    fake_cursor_cls,
    docker_env,
):
    monkeypatch.setenv("DB_DUMP_PASSPHRASE", "pw")
    monkeypatch.setattr(
        mod.shutil,
        "which",
        lambda name: {"docker": "/usr/bin/docker", "openssl": "/usr/bin/openssl"}.get(
            name
        ),
    )
    monkeypatch.setattr(mod.subprocess, "Popen", fake_popen_cls)

    spot_row = ("PNGP24_001", "Bob", True, "2024-01-01", "24AB", "M")
    cursor = fake_cursor_cls(
        fetchone_results=[(i,) for i in range(1, 7)] + [spot_row],
    )
    conn = fake_connection_cls(cursor)
    connect_calls = []

    def fake_connect(*args, **kwargs):
        connect_calls.append(args)
        return conn

    monkeypatch.setattr(mod.psycopg2, "connect", fake_connect)

    expected = ExpectedState(counts={"core_animal": 1}, spot_row=spot_row)

    restore_and_verify(Path("dump.enc"), expected, 16)

    p1, p2 = fake_popen_cls.instances  # p1 = openssl (host), p2 = docker pg_restore
    assert p1.argv[0] == "/usr/bin/openssl"
    assert p2.argv[0] == "/usr/bin/docker"

    # Full-list equality (not membership) -- the `docker_env` fixture pins
    # HOME/PATH/DOCKER_HOST and clears the other 5 allowlist members, so
    # the -e name set is fully deterministic here: the 3 docker-context
    # vars + the 6 PG* keys from the (rewritten) restore target env.
    # `--entrypoint pg_restore` is the fix for the live entrypoint-
    # dispatch bug; `--network container:<id>` (never `--network=none`,
    # mutually exclusive with the container-namespace join this leg
    # needs) reaches the ephemeral testcontainers Postgres; -t/--tty/-it
    # are never emitted even though interactive=True.
    assert p2.argv == [
        "/usr/bin/docker",
        "run",
        "--rm",
        "--pull=never",
        "--security-opt",
        "no-new-privileges",
        "-i",
        "--network",
        f"container:{_FAKE_CONTAINER_ID}",
        "-e",
        "DOCKER_HOST",
        "-e",
        "HOME",
        "-e",
        "PATH",
        "-e",
        "PGDATABASE",
        "-e",
        "PGHOST",
        "-e",
        "PGPASSWORD",
        "-e",
        "PGPORT",
        "-e",
        "PGSSLMODE",
        "-e",
        "PGUSER",
        "--entrypoint",
        "pg_restore",
        mod.DEFAULT_PG_CLIENT_IMAGE,
        "--no-owner",
        "--no-privileges",
        "--exit-on-error",
        "--single-transaction",
        "-d",
        "test",
    ]
    assert "--network=none" not in p2.argv
    assert "-t" not in p2.argv
    assert "--tty" not in p2.argv
    assert "-it" not in p2.argv

    # PGHOST/PGPORT/PGSSLMODE rewritten to the container-internal loopback
    # -- never the fake published port (55432) from the fixture's DSN.
    restore_docker_env = p2.kwargs["env"]
    assert restore_docker_env["PGHOST"] == "127.0.0.1"
    assert restore_docker_env["PGPORT"] == "5432"
    assert restore_docker_env["PGSSLMODE"] == "disable"

    # The post-restore verification psycopg2.connect() call must reuse
    # the container's PUBLISHED DSN (127.0.0.1:55432 per the fixture),
    # never the rewritten container-internal env -- reusing the internal
    # env here would dial 127.0.0.1:5432 on the HOST, a real
    # data-integrity hazard (could silently hit an unrelated local
    # Postgres instance).
    assert connect_calls == [("postgresql://test:test@127.0.0.1:55432/test",)]


def test_restore_and_verify_invalid_container_id_raises_no_popen_and_clean_exit(
    monkeypatch, fake_testcontainers_module
):
    monkeypatch.setenv("DB_DUMP_PASSPHRASE", "pw")
    monkeypatch.setattr(
        mod.shutil,
        "which",
        lambda name: {"docker": "/usr/bin/docker", "openssl": "/usr/bin/openssl"}.get(
            name
        ),
    )

    popen_calls = []
    monkeypatch.setattr(
        mod.subprocess,
        "Popen",
        lambda *a, **k: popen_calls.append((a, k)) or mock.Mock(),
    )

    _container_cls, container_instance = fake_testcontainers_module
    container_instance.get_wrapped_container.return_value.id = "not-a-valid-id"

    expected = ExpectedState(counts={"core_animal": 1}, spot_row=("a",))
    with pytest.raises(RuntimeError):
        restore_and_verify(Path("dump.enc"), expected, 16)

    assert popen_calls == []
    # the `with PostgresContainer(...) as container:` block must still
    # exit cleanly -- no leaked container.
    container_instance.__exit__.assert_called_once()


# ---------------------------------------------------------------------------
# Pure comparison logic -- directly testable, no Docker/subprocess involved.
# ---------------------------------------------------------------------------
def test_compare_results_all_match_passes():
    expected = ExpectedState(
        counts={"core_animal": 3, "core_region": 1},
        spot_row=("PNGP24_001", "Bob", True, "2024-01-01", "24AB", "M"),
    )
    actual_counts = {"core_animal": 3, "core_region": 1}
    actual_spot_row = ("PNGP24_001", "Bob", True, "2024-01-01", "24AB", "M")

    result = mod._compare_results(expected, actual_counts, actual_spot_row)

    assert result.passed is True
    assert result.count_mismatches == {}
    assert result.spot_check_ok is True


def test_compare_results_count_mismatch_fails():
    expected = ExpectedState(counts={"core_animal": 3}, spot_row=("a",))
    actual_counts = {"core_animal": 2}

    result = mod._compare_results(expected, actual_counts, ("a",))

    assert result.passed is False
    assert result.count_mismatches == {"core_animal": (3, 2)}


@pytest.mark.parametrize(
    "expected_row,actual_row",
    [
        (
            ("PNGP24_001", "Bob", True, "2024-01-01", "24AB", "M"),
            ("PNGP24_001", "Bob", "t", "2024-01-01", "24AB", "M"),  # bool as 't'
        ),
        (
            ("PNGP24_001", "Bob", True, "2024-01-01", "24AB", "M"),
            ("PNGP24_001", "Bob", 1, "2024-01-01", "24AB", "M"),  # bool as 1
        ),
    ],
)
def test_compare_results_normalizes_marked_type_variants(expected_row, actual_row):
    expected = ExpectedState(counts={}, spot_row=expected_row)
    result = mod._compare_results(expected, {}, actual_row)
    assert result.spot_check_ok is True


def test_compare_results_normalizes_capture_date_type_variants():
    import datetime

    expected = ExpectedState(
        counts={},
        spot_row=("PNGP24_001", "Bob", True, "2024-01-01", "24AB", "M"),
    )
    actual_row = (
        "PNGP24_001",
        "Bob",
        True,
        datetime.date(2024, 1, 1),
        "24AB",
        "M",
    )
    result = mod._compare_results(expected, {}, actual_row)
    assert result.spot_check_ok is True


def test_compare_results_spot_check_only_mismatch_fails_overall():
    expected = ExpectedState(
        counts={"core_animal": 3},
        spot_row=("PNGP24_001", "Bob", True, "2024-01-01", "24AB", "M"),
    )
    actual_row = ("PNGP24_002", "Bob", True, "2024-01-01", "24AB", "M")

    result = mod._compare_results(expected, {"core_animal": 3}, actual_row)

    assert result.passed is False
    assert result.count_mismatches == {}
    assert result.spot_check_ok is False
