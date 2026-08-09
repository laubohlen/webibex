"""P0 tests for scripts/db_restore_drill.py: main() orchestration and the
shutil.which binary-resolution posture (T50).
"""

from __future__ import annotations

import shutil
from unittest import mock

import pytest

import scripts.db_restore_drill as mod
from scripts.db_restore_drill import ExpectedState, ServerInfo, VerifyResult

pytestmark = pytest.mark.spec(
    ref="docs/security-remediation-plan.md#gate-restore-drill-required"
)

_ARGV = [
    "--project-id",
    "proj-1",
    "--environment-id",
    "env-1",
    "--token-kind",
    "account",
]


def _passing_verify_result():
    return VerifyResult(
        passed=True,
        count_mismatches={},
        spot_check_ok=True,
        spot_check_expected=("PNGP24_001",),
        spot_check_actual=("PNGP24_001",),
    )


def _mock_all_collaborators(monkeypatch, verify_result, expected=None):
    if expected is None:
        expected = ExpectedState(counts={"core_animal": 1}, spot_row=("PNGP24_001",))

    monkeypatch.setattr(
        mod,
        "fetch_database_url",
        mock.Mock(return_value="postgresql://x:y@localhost/db"),
    )
    monkeypatch.setattr(
        mod,
        "preflight_source",
        mock.Mock(
            return_value=ServerInfo(
                server_major_version=16, tables_present=mod.EXPECTED_TABLES
            )
        ),
    )
    monkeypatch.setattr(mod, "_connect_readonly", mock.Mock(return_value=mock.Mock()))
    monkeypatch.setattr(mod, "collect_expected", mock.Mock(return_value=expected))
    monkeypatch.setattr(mod, "dump_encrypted", mock.Mock())
    monkeypatch.setattr(
        mod, "restore_and_verify", mock.Mock(return_value=verify_result)
    )
    return expected


def test_main_all_match_exits_zero_and_prints_pass_table(
    monkeypatch, capsys
):
    monkeypatch.setenv("RAILWAY_API_TOKEN", "tok")
    monkeypatch.setenv("DB_DUMP_PASSPHRASE", "pw")
    _mock_all_collaborators(monkeypatch, _passing_verify_result())

    exit_code = mod.main(_ARGV)

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "PASS" in captured.out


def test_main_single_table_mismatch_nonzero_exit_names_table(monkeypatch, capsys):
    monkeypatch.setenv("RAILWAY_API_TOKEN", "tok")
    monkeypatch.setenv("DB_DUMP_PASSPHRASE", "pw")
    expected = ExpectedState(
        counts={"core_animal": 3, "core_region": 1}, spot_row=("PNGP24_001",)
    )
    verify_result = VerifyResult(
        passed=False,
        count_mismatches={"core_region": (1, 0)},
        spot_check_ok=True,
        spot_check_expected=("PNGP24_001",),
        spot_check_actual=("PNGP24_001",),
    )
    _mock_all_collaborators(monkeypatch, verify_result, expected=expected)

    exit_code = mod.main(_ARGV)

    assert exit_code != 0
    captured = capsys.readouterr()
    assert "core_region" in captured.out


def test_main_spot_check_only_mismatch_nonzero_exit(monkeypatch, capsys):
    monkeypatch.setenv("RAILWAY_API_TOKEN", "tok")
    monkeypatch.setenv("DB_DUMP_PASSPHRASE", "pw")
    verify_result = VerifyResult(
        passed=False,
        count_mismatches={},
        spot_check_ok=False,
        spot_check_expected=("PNGP24_001",),
        spot_check_actual=("PNGP24_999",),
    )
    _mock_all_collaborators(monkeypatch, verify_result)

    exit_code = mod.main(_ARGV)

    assert exit_code != 0


@pytest.mark.parametrize("missing_var", ["RAILWAY_API_TOKEN", "DB_DUMP_PASSPHRASE"])
def test_main_missing_required_env_var_refuses_before_any_call(
    monkeypatch, capsys, missing_var
):
    all_vars = {"RAILWAY_API_TOKEN": "tok", "DB_DUMP_PASSPHRASE": "pw"}
    for name, value in all_vars.items():
        if name == missing_var:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)

    fetch_mock = mock.Mock()
    monkeypatch.setattr(mod, "fetch_database_url", fetch_mock)
    popen_mock = mock.Mock()
    monkeypatch.setattr(mod.subprocess, "Popen", popen_mock)

    exit_code = mod.main(_ARGV)

    assert exit_code != 0
    fetch_mock.assert_not_called()
    popen_mock.assert_not_called()
    captured = capsys.readouterr()
    assert missing_var in captured.err


def test_main_whitespace_only_env_var_treated_as_missing(monkeypatch, capsys):
    monkeypatch.setenv("RAILWAY_API_TOKEN", "   ")
    monkeypatch.setenv("DB_DUMP_PASSPHRASE", "pw")

    fetch_mock = mock.Mock()
    monkeypatch.setattr(mod, "fetch_database_url", fetch_mock)

    exit_code = mod.main(_ARGV)

    assert exit_code != 0
    fetch_mock.assert_not_called()


def test_main_missing_env_var_leaves_no_partial_artifact(monkeypatch, tmp_path):
    monkeypatch.delenv("RAILWAY_API_TOKEN", raising=False)
    monkeypatch.setenv("DB_DUMP_PASSPHRASE", "pw")

    out_path = tmp_path / "dump.enc"
    argv = [*_ARGV, "--out-path", str(out_path)]

    exit_code = mod.main(argv)

    assert exit_code != 0
    assert not out_path.exists()


def test_main_no_credential_appears_in_stdout_stderr_with_sentinel_passwords(
    monkeypatch, capsys, caplog
):
    monkeypatch.setenv("RAILWAY_API_TOKEN", "tok")
    monkeypatch.setenv("DB_DUMP_PASSPHRASE", "pw")

    source_sentinel = "SOURCE-SECRET-9f2a"
    container_sentinel = "CONTAINER-SECRET-7c1b"
    monkeypatch.setattr(
        mod,
        "fetch_database_url",
        mock.Mock(return_value=f"postgresql://user:{source_sentinel}@prod-host/db"),
    )
    monkeypatch.setattr(
        mod,
        "preflight_source",
        mock.Mock(
            return_value=ServerInfo(
                server_major_version=16, tables_present=mod.EXPECTED_TABLES
            )
        ),
    )
    monkeypatch.setattr(mod, "_connect_readonly", mock.Mock(return_value=mock.Mock()))
    expected = ExpectedState(counts={"core_animal": 1}, spot_row=("PNGP24_001",))
    monkeypatch.setattr(mod, "collect_expected", mock.Mock(return_value=expected))
    monkeypatch.setattr(mod, "dump_encrypted", mock.Mock())

    def fake_restore_and_verify(enc_path, exp, prod_major_version, source_dsn=None):
        assert container_sentinel not in str(source_dsn)
        return _passing_verify_result()

    monkeypatch.setattr(mod, "restore_and_verify", fake_restore_and_verify)

    exit_code = mod.main(_ARGV)

    assert exit_code == 0
    captured = capsys.readouterr()
    assert source_sentinel not in captured.out
    assert source_sentinel not in captured.err
    assert container_sentinel not in captured.out
    assert container_sentinel not in captured.err
    for record in caplog.records:
        assert source_sentinel not in record.getMessage()
        assert container_sentinel not in record.getMessage()


# ---------------------------------------------------------------------------
# T50 -- binary-path posture: shutil.which resolution, scoped ruff suppression
# ---------------------------------------------------------------------------
def test_binary_resolution_uses_shutil_which_not_hardcoded_paths():
    import inspect

    source = inspect.getsource(mod)
    assert "shutil.which(" in source
    # no hardcoded /usr/bin or /usr/local/bin absolute binary paths for
    # pg_dump/pg_restore/openssl anywhere in the production code path.
    for hardcoded in ("/usr/bin/pg_dump", "/usr/bin/pg_restore", "/usr/bin/openssl"):
        assert hardcoded not in source


def test_shutil_which_is_real_shutil_module_reference():
    assert mod.shutil is shutil
