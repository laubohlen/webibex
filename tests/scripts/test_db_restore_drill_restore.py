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
    # testcontainers genuinely isn't installed in this sandbox.
    monkeypatch.delitem(sys.modules, "testcontainers.postgres", raising=False)
    monkeypatch.delitem(sys.modules, "testcontainers", raising=False)

    expected = ExpectedState(counts={"core_animal": 1}, spot_row=("a", "b"))
    with pytest.raises(RuntimeError, match="testcontainers"):
        restore_and_verify(Path("dump.enc"), expected, 16)


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
    container_instance.__enter__ = mock.Mock(return_value=container_instance)
    container_instance.__exit__ = mock.Mock(return_value=False)

    container_cls = mock.Mock(return_value=container_instance)
    fake_postgres_mod.PostgresContainer = container_cls

    monkeypatch.setitem(sys.modules, "testcontainers", fake_pkg)
    monkeypatch.setitem(sys.modules, "testcontainers.postgres", fake_postgres_mod)
    return container_cls, container_instance


@pytest.mark.parametrize(
    "bad_version",
    ["16-alpine; rm -rf /", "latest", "", "16.2"],
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

    def raising_guard(dsn, source_dsn=None):
        raise ValueError("guard tripped -- not a loopback target")

    monkeypatch.setattr(mod, "_assert_local_target", raising_guard)

    expected = ExpectedState(counts={"core_animal": 1}, spot_row=("a",))
    with pytest.raises(ValueError, match="guard tripped"):
        restore_and_verify(Path("dump.enc"), expected, 16)

    assert popen_calls == []


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
