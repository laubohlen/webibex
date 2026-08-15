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

    def fake_restore_and_verify(
        enc_path, exp, prod_major_version, source_dsn=None, *, image=None
    ):
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
# SOURCE_DSN env var: local dry-run bypass of the Railway GraphQL fetch.
# An env var, not a CLI flag -- a DSN can carry a password, and this
# script's own invariant is "secrets are NEVER accepted on argv".
# ---------------------------------------------------------------------------
def test_parse_args_requires_railway_args_unless_source_dsn_env_set(monkeypatch):
    monkeypatch.delenv("SOURCE_DSN", raising=False)
    with pytest.raises(SystemExit):
        mod.parse_args(["--token-kind", "account"])  # missing project/environment id


def test_parse_args_source_dsn_env_makes_railway_args_optional(monkeypatch):
    monkeypatch.setenv("SOURCE_DSN", "postgresql://u:p@localhost/fake")

    args = mod.parse_args([])

    assert args.project_id is None
    assert args.environment_id is None
    assert args.token_kind is None


def test_parse_args_full_railway_args_still_work_without_source_dsn(monkeypatch):
    monkeypatch.delenv("SOURCE_DSN", raising=False)

    args = mod.parse_args(_ARGV)

    assert args.project_id == "proj-1"
    assert args.environment_id == "env-1"
    assert args.token_kind == "account"


def test_main_source_dsn_env_bypasses_railway_fetch_and_token_requirement(
    monkeypatch, capsys
):
    # No RAILWAY_API_TOKEN at all -- SOURCE_DSN dry-run needs no Railway
    # account/token whatsoever.
    monkeypatch.delenv("RAILWAY_API_TOKEN", raising=False)
    monkeypatch.setenv("DB_DUMP_PASSPHRASE", "pw")
    fake_dsn = "postgresql://u:p@localhost/fake"
    monkeypatch.setenv("SOURCE_DSN", fake_dsn)

    fetch_mock = mock.Mock()
    monkeypatch.setattr(mod, "fetch_database_url", fetch_mock)
    preflight_mock = mock.Mock(
        return_value=ServerInfo(
            server_major_version=16, tables_present=mod.EXPECTED_TABLES
        )
    )
    monkeypatch.setattr(mod, "preflight_source", preflight_mock)
    monkeypatch.setattr(mod, "_connect_readonly", mock.Mock(return_value=mock.Mock()))
    expected = ExpectedState(counts={"core_animal": 1}, spot_row=("PNGP24_001",))
    monkeypatch.setattr(mod, "collect_expected", mock.Mock(return_value=expected))
    monkeypatch.setattr(mod, "dump_encrypted", mock.Mock())
    monkeypatch.setattr(
        mod, "restore_and_verify", mock.Mock(return_value=_passing_verify_result())
    )

    exit_code = mod.main([])

    assert exit_code == 0
    fetch_mock.assert_not_called()
    preflight_mock.assert_called_once_with(fake_dsn, image=mod.DEFAULT_PG_CLIENT_IMAGE)


def test_main_source_dsn_never_appears_in_argv_or_env_shape(monkeypatch):
    """Regression guard for the design invariant: SOURCE_DSN is read from
    os.environ only, never accepted as a CLI argument."""
    import inspect

    source = inspect.getsource(mod.parse_args)
    assert "source-dsn" not in source
    assert "source_dsn" not in source


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
    # pg_dump/pg_restore now run inside `docker run --rm` -- zero
    # shutil.which("pg_dump")/("pg_restore") remains in the production
    # path (only "docker" and "openssl" are still resolved via
    # shutil.which).
    assert 'shutil.which("pg_dump")' not in source
    assert 'shutil.which("pg_restore")' not in source


def test_shutil_which_is_real_shutil_module_reference():
    assert mod.shutil is shutil


# ---------------------------------------------------------------------------
# --pg-client-image: threaded identically into all 3 docker-aware
# collaborators (preflight_source, dump_encrypted, restore_and_verify)
# ---------------------------------------------------------------------------
def test_parse_args_pg_client_image_default_is_module_default():
    args = mod.parse_args(_ARGV)
    assert args.pg_client_image == mod.DEFAULT_PG_CLIENT_IMAGE


def test_parse_args_pg_client_image_custom_value_threads_verbatim():
    args = mod.parse_args([*_ARGV, "--pg-client-image", "registry.example/pg:16"])
    assert args.pg_client_image == "registry.example/pg:16"


def _mock_all_collaborators_capture_image(monkeypatch):
    """Same collaborator mocking as `_mock_all_collaborators`, but each
    mock records the `image` kwarg it received.
    """
    received_images: dict[str, object] = {}
    expected = ExpectedState(counts={"core_animal": 1}, spot_row=("PNGP24_001",))

    monkeypatch.setattr(
        mod,
        "fetch_database_url",
        mock.Mock(return_value="postgresql://x:y@localhost/db"),
    )

    def fake_preflight_source(dsn, *, image=None):
        received_images["preflight_source"] = image
        return ServerInfo(server_major_version=16, tables_present=mod.EXPECTED_TABLES)

    monkeypatch.setattr(mod, "preflight_source", fake_preflight_source)
    monkeypatch.setattr(mod, "_connect_readonly", mock.Mock(return_value=mock.Mock()))
    monkeypatch.setattr(mod, "collect_expected", mock.Mock(return_value=expected))

    def fake_dump_encrypted(dsn, out_path, *args, image=None, **kwargs):
        received_images["dump_encrypted"] = image

    monkeypatch.setattr(mod, "dump_encrypted", fake_dump_encrypted)

    def fake_restore_and_verify(
        enc_path, exp, prod_major_version, source_dsn=None, *, image=None
    ):
        received_images["restore_and_verify"] = image
        return _passing_verify_result()

    monkeypatch.setattr(mod, "restore_and_verify", fake_restore_and_verify)
    return received_images


def test_main_pg_client_image_default_threads_identically_into_all_collaborators(
    monkeypatch,
):
    monkeypatch.setenv("RAILWAY_API_TOKEN", "tok")
    monkeypatch.setenv("DB_DUMP_PASSPHRASE", "pw")
    received_images = _mock_all_collaborators_capture_image(monkeypatch)

    exit_code = mod.main(_ARGV)

    assert exit_code == 0
    assert received_images == {
        "preflight_source": mod.DEFAULT_PG_CLIENT_IMAGE,
        "dump_encrypted": mod.DEFAULT_PG_CLIENT_IMAGE,
        "restore_and_verify": mod.DEFAULT_PG_CLIENT_IMAGE,
    }


def test_main_pg_client_image_custom_value_threads_identically_into_all_collaborators(
    monkeypatch,
):
    monkeypatch.setenv("RAILWAY_API_TOKEN", "tok")
    monkeypatch.setenv("DB_DUMP_PASSPHRASE", "pw")
    received_images = _mock_all_collaborators_capture_image(monkeypatch)

    custom_image = "registry.example.com:5000/ns/postgres:16"
    exit_code = mod.main([*_ARGV, "--pg-client-image", custom_image])

    assert exit_code == 0
    assert received_images == {
        "preflight_source": custom_image,
        "dump_encrypted": custom_image,
        "restore_and_verify": custom_image,
    }


def test_main_invalid_pg_client_image_nonzero_exit_no_artifact_no_credential_leak(
    monkeypatch, capsys, tmp_path
):
    monkeypatch.setenv("RAILWAY_API_TOKEN", "tok")
    monkeypatch.setenv("DB_DUMP_PASSPHRASE", "pw")

    source_sentinel = "SOURCE-SECRET-image-test"
    monkeypatch.setattr(
        mod,
        "fetch_database_url",
        mock.Mock(return_value=f"postgresql://user:{source_sentinel}@prod-host/db"),
    )
    # preflight_source is the real function -- it's the one that validates
    # `image` via _docker_preflight's _IMAGE_REF_RE check.
    monkeypatch.setattr(mod, "_docker_path", lambda: "/usr/bin/docker")

    out_path = tmp_path / "dump.enc"
    # A single argv token (no shell involved) containing an embedded
    # space -- argparse assigns it whole to --pg-client-image, so this
    # exercises _IMAGE_REF_RE rejection, not argparse's own tokenizing.
    bad_image = "postgres:16 --privileged"
    argv = [*_ARGV, "--pg-client-image", bad_image, "--out-path", str(out_path)]

    exit_code = mod.main(argv)

    assert exit_code != 0
    assert not out_path.exists()
    captured = capsys.readouterr()
    assert source_sentinel not in captured.out
    assert source_sentinel not in captured.err
