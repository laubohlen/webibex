"""T19-T26, T35, T37, T38: scripts/verify_prod_parity.py (R3).

Slow, opt-in end-to-end production-dependency-parity gate. Creates an
isolated venv pinned to the repo's runtime.txt Python version, installs
ONLY requirements.txt (never requirements-dev.txt), runs `manage.py
check` + `manage.py migrate --run-syncdb` against a throwaway sqlite DB,
boots `manage.py runserver`, does one GET / and asserts HTTP 200 with no
traceback in the server log, then tears everything down.

Every test in this file EXCEPT the one at the bottom
(`test_verify_prod_parity_full_run_against_real_requirements_txt`, marked
`prod_parity` + skip-gated) is a FAST unit test using the `fake_run`/
`fake_popen_cls` fixtures (tests/scripts/conftest.py) -- no real venv,
subprocess, or network involved -- and runs in the DEFAULT `pytest` suite.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import pytest

import scripts.verify_prod_parity as mod
from tests.scripts.conftest import FakeCompletedProcess

_REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# T19 -- argv/env contract: no secret accepted on argv.
# ---------------------------------------------------------------------------
def test_no_argparse_argument_is_secret_shaped():
    args = mod.parse_args([])
    secret_markers = ("token", "password", "passphrase", "secret", "key", "credential")

    for name in vars(args):
        assert not any(marker in name.lower() for marker in secret_markers), (
            f"argparse destination {name!r} looks secret-shaped -- this "
            "script must never accept a secret on argv"
        )


# ---------------------------------------------------------------------------
# T20 -- venv creation uses the repo Python version from runtime.txt.
# ---------------------------------------------------------------------------
def test_read_runtime_python_version_parses_runtime_txt():
    version = mod.read_runtime_python_version()

    assert version == "3.12.5"


def test_resolve_pinned_python_looks_for_exact_major_minor(monkeypatch):
    calls = []

    def _fake_which(name):
        calls.append(name)
        return "/usr/bin/python3.12" if name == "python3.12" else None

    monkeypatch.setattr(mod.shutil, "which", _fake_which)

    resolved = mod.resolve_pinned_python("3.12.5")

    assert resolved == "/usr/bin/python3.12"
    assert calls == ["python3.12"]


def test_resolve_pinned_python_raises_when_not_found(monkeypatch):
    monkeypatch.setattr(mod.shutil, "which", lambda _name: None)

    with pytest.raises(RuntimeError, match=r"python3\.12"):
        mod.resolve_pinned_python("3.12.5")


def test_create_venv_argv_uses_resolved_python_exe(monkeypatch, fake_run, tmp_path):
    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    fake_run.queue_result(FakeCompletedProcess(returncode=0))
    scratch = tmp_path / "tmp" / "verify_prod_parity_venv"

    mod.create_venv("/usr/bin/python3.12", scratch)

    assert len(fake_run.calls) == 1
    (args, _kwargs) = fake_run.calls[0]
    assert args[0] == ["/usr/bin/python3.12", "-m", "venv", str(scratch)]


# ---------------------------------------------------------------------------
# T21 -- pip install argv is `-r requirements.txt` ONLY, full-argv equality.
# ---------------------------------------------------------------------------
def test_pip_install_requirements_full_argv_equality(monkeypatch, fake_run, tmp_path):
    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    fake_run.queue_result(FakeCompletedProcess(returncode=0))
    venv_python = tmp_path / "fake-venv" / "bin" / "python"

    mod.pip_install_requirements(venv_python)

    assert len(fake_run.calls) == 1
    (args, _kwargs) = fake_run.calls[0]
    assert args[0] == [
        str(venv_python),
        "-m",
        "pip",
        "install",
        "-r",
        str(_REPO_ROOT / "requirements.txt"),
    ]


def test_pip_install_requirements_never_mentions_requirements_dev(
    monkeypatch, fake_run, tmp_path
):
    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    fake_run.queue_result(FakeCompletedProcess(returncode=0))
    venv_python = tmp_path / "fake-venv" / "bin" / "python"

    mod.pip_install_requirements(venv_python)

    (args, _kwargs) = fake_run.calls[0]
    assert "requirements-dev.txt" not in " ".join(args[0])


def test_pip_install_requirements_raises_on_nonzero_rc(monkeypatch, fake_run, tmp_path):
    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    fake_run.queue_result(FakeCompletedProcess(returncode=1, stderr="boom"))

    with pytest.raises(RuntimeError, match="boom"):
        mod.pip_install_requirements(tmp_path / "fake-venv" / "bin" / "python")


# ---------------------------------------------------------------------------
# T22 -- child env is an explicit allowlist; ambient creds not inherited.
# ---------------------------------------------------------------------------
def test_build_child_env_is_an_explicit_allowlist_not_full_os_environ(monkeypatch):
    monkeypatch.setenv("HOME", "/home/tester")
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    monkeypatch.setenv("DECOY_AMBIENT_SECRET", "should-never-appear")

    result = mod._build_child_env({"ENVIRONMENT": "test"})

    assert result["HOME"] == "/home/tester"
    assert result["PATH"] == "/usr/bin:/bin"
    assert result["ENVIRONMENT"] == "test"
    assert "DECOY_AMBIENT_SECRET" not in result


def test_django_test_env_overrides_match_conftest_placeholders_exactly():
    """F12: the exact SAME placeholder values as root conftest.py -- never
    a diverging or newly-invented credential-shaped literal."""
    assert mod._DJANGO_TEST_ENV_OVERRIDES == {
        "DJANGO_SETTINGS_MODULE": "webibex.settings",
        "ENVIRONMENT": "test",
        "SECRET_KEY": "test-secret-key-not-for-production",
        "AWS_ACCESS_KEY_ID": "test-aws-access-key-id",
        "AWS_SECRET_ACCESS_KEY": "test-aws-secret-access-key",
        "AWS_S3_ENDPOINT_URL": "https://example-b2-endpoint.invalid",
        "AWS_STORAGE_BUCKET_NAME": "test-bucket",
        "AWS_S3_REGION_NAME": "us-west-000",
        "RUNPOD_ENDPOINT_ID": "test-runpod-endpoint-id",
        "RUNPOD_API_KEY": "test-runpod-api-key",
    }


# ---------------------------------------------------------------------------
# T23 -- probe targets 127.0.0.1, never localhost (F11: ALLOWED_HOSTS gap).
# ---------------------------------------------------------------------------
def test_probe_host_default_is_127_0_0_1_never_localhost():
    assert mod._PROBE_HOST == "127.0.0.1"
    assert mod.default_config().host == "127.0.0.1"


def test_probe_server_requests_127_0_0_1_url(monkeypatch):
    requested_urls = []

    class _FakeResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return False

    def _fake_urlopen(url, timeout=5):
        requested_urls.append(url)
        return _FakeResponse()

    monkeypatch.setattr(mod, "urlopen", _fake_urlopen)

    status = mod.probe_server("127.0.0.1", 8734, attempts=1)

    assert status == 200
    assert requested_urls == ["http://127.0.0.1:8734/"]
    assert "localhost" not in requested_urls[0]


# ---------------------------------------------------------------------------
# T24 -- non-200 / traceback-in-log -> non-zero exit (fail-secure).
# ---------------------------------------------------------------------------
def _mock_run_collaborators(monkeypatch, *, probe_status, has_traceback):
    monkeypatch.setattr(mod, "refuse_if_production", lambda: None)
    monkeypatch.setattr(mod, "resolve_pinned_python", lambda _v: "/usr/bin/python3.12")
    monkeypatch.setattr(mod, "copy_repo_tree", lambda _dest: None)
    monkeypatch.setattr(
        mod, "create_venv", lambda _exe, _dir: Path("/fake/venv/bin/python")
    )
    monkeypatch.setattr(mod, "pip_install_requirements", lambda _py: None)
    monkeypatch.setattr(mod, "run_manage_check", lambda _py, _cwd, _env: None)
    monkeypatch.setattr(mod, "run_manage_migrate", lambda _py, _cwd, _env: None)

    class _FakeProcess:
        def poll(self):
            return None

        def terminate(self):
            pass

        def wait(self, timeout=None):
            return 0

    monkeypatch.setattr(
        mod,
        "start_runserver",
        lambda _py, _cwd, _env, _cfg: (_FakeProcess(), Path("/fake/log")),
    )
    monkeypatch.setattr(mod, "probe_server", lambda _host, _port, **_kw: probe_status)
    monkeypatch.setattr(mod, "log_contains_traceback", lambda _path: has_traceback)
    monkeypatch.setattr(mod, "teardown_scratch", lambda _dir: None)


def test_run_non_200_status_returns_nonzero_exit(monkeypatch):
    _mock_run_collaborators(monkeypatch, probe_status=500, has_traceback=False)

    exit_code = mod.run(mod.default_config())

    assert exit_code != 0


def test_run_traceback_in_log_returns_nonzero_exit_even_if_200(monkeypatch):
    _mock_run_collaborators(monkeypatch, probe_status=200, has_traceback=True)

    exit_code = mod.run(mod.default_config())

    assert exit_code != 0


def test_run_happy_path_200_no_traceback_returns_zero(monkeypatch):
    _mock_run_collaborators(monkeypatch, probe_status=200, has_traceback=False)

    exit_code = mod.run(mod.default_config())

    assert exit_code == 0


# ---------------------------------------------------------------------------
# T25 -- teardown always runs, even on a mid-run exception.
# ---------------------------------------------------------------------------
def test_teardown_runs_even_when_run_manage_check_raises(monkeypatch):
    _mock_run_collaborators(monkeypatch, probe_status=200, has_traceback=False)
    monkeypatch.setattr(
        mod,
        "run_manage_check",
        lambda _py, _cwd, _env: (_ for _ in ()).throw(RuntimeError("check failed")),
    )
    teardown_calls = []
    monkeypatch.setattr(
        mod, "teardown_scratch", lambda scratch_dir: teardown_calls.append(scratch_dir)
    )

    with pytest.raises(RuntimeError, match="check failed"):
        mod.run(mod.default_config())

    assert len(teardown_calls) == 1


def test_teardown_runs_and_stops_server_when_probe_raises(monkeypatch):
    _mock_run_collaborators(monkeypatch, probe_status=200, has_traceback=False)
    monkeypatch.setattr(
        mod,
        "probe_server",
        lambda *_a, **_kw: (_ for _ in ()).throw(RuntimeError("unreachable")),
    )
    teardown_calls = []
    stop_calls = []
    monkeypatch.setattr(
        mod, "teardown_scratch", lambda scratch_dir: teardown_calls.append(scratch_dir)
    )
    monkeypatch.setattr(
        mod, "stop_server", lambda process, **_kw: stop_calls.append(process)
    )

    with pytest.raises(RuntimeError, match="unreachable"):
        mod.run(mod.default_config())

    assert len(teardown_calls) == 1
    assert len(stop_calls) == 1


# ---------------------------------------------------------------------------
# T37 -- venv/scratch path is gitignored / outside the repo (F14).
# ---------------------------------------------------------------------------
def test_validate_scratch_dir_accepts_tmp_subdir():
    mod._validate_scratch_dir(_REPO_ROOT / "tmp" / "verify_prod_parity_scratch")


def test_validate_scratch_dir_accepts_path_outside_repo(tmp_path):
    mod._validate_scratch_dir(tmp_path / "outside_scratch")


def test_validate_scratch_dir_rejects_in_repo_non_tmp_path():
    with pytest.raises(ValueError, match="tmp"):
        mod._validate_scratch_dir(_REPO_ROOT / "core" / "scratch")


def test_default_config_scratch_dir_is_under_tmp():
    config = mod.default_config()

    resolved = config.scratch_dir.resolve()
    assert resolved.is_relative_to(_REPO_ROOT / "tmp")


# ---------------------------------------------------------------------------
# T38 -- refuses to run under ENVIRONMENT=production, before any side effect.
# ---------------------------------------------------------------------------
def test_refuse_if_production_raises(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")

    with pytest.raises(RuntimeError, match="production"):
        mod.refuse_if_production()


def test_refuse_if_production_allows_non_production(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "test")

    mod.refuse_if_production()  # must not raise


def test_run_refuses_before_any_side_effect_under_production(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    side_effect_calls = []
    monkeypatch.setattr(
        mod,
        "resolve_pinned_python",
        lambda _v: side_effect_calls.append("resolve_pinned_python"),
    )
    monkeypatch.setattr(
        mod, "copy_repo_tree", lambda _d: side_effect_calls.append("copy_repo_tree")
    )
    monkeypatch.setattr(
        mod, "create_venv", lambda _e, _d: side_effect_calls.append("create_venv")
    )

    with pytest.raises(RuntimeError, match="production"):
        mod.run(mod.default_config())

    assert side_effect_calls == []


# ---------------------------------------------------------------------------
# T35 -- no new test/tooling file contains a real credential-shaped literal.
# ---------------------------------------------------------------------------
_KNOWN_PLACEHOLDER_VALUES = frozenset(
    {
        "test-secret-key-not-for-production",
        "test-aws-access-key-id",
        "test-aws-secret-access-key",
        "https://example-b2-endpoint.invalid",
        "test-bucket",
        "us-west-000",
        "test-runpod-endpoint-id",
        "test-runpod-api-key",
    }
)

_CREDENTIAL_SHAPED_RE = re.compile(
    r'(?i)(secret|password|passphrase|api[_-]?key|token)["\']?\s*[:=]\s*["\']([^"\']+)["\']'
)


def test_no_new_r3_file_contains_a_non_placeholder_credential_shaped_literal():
    for path in (_REPO_ROOT / "scripts" / "verify_prod_parity.py", Path(__file__)):
        text = path.read_text()
        for match in _CREDENTIAL_SHAPED_RE.finditer(text):
            value = match.group(2)
            assert value in _KNOWN_PLACEHOLDER_VALUES, (
                f"{path}: possible real-credential-shaped literal: {match.group(0)!r}"
            )


# ---------------------------------------------------------------------------
# T26 -- live wrapper: skip-gated, prod_parity-marked.
# ---------------------------------------------------------------------------
def _pinned_python_available() -> bool:
    try:
        version = mod.read_runtime_python_version()
    except (FileNotFoundError, ValueError):
        return False
    major_minor = ".".join(version.split(".")[:2])
    return shutil.which(f"python{major_minor}") is not None


def _pypi_reachable() -> bool:
    try:
        with urlopen("https://pypi.org/simple/", timeout=5) as response:
            return response.status == 200
    except (URLError, OSError, ValueError):
        return False


@pytest.mark.prod_parity
@pytest.mark.skipif(
    not (_pinned_python_available() and _pypi_reachable()),
    reason=(
        "prod_parity live tier needs BOTH a pinned python3.12 interpreter "
        "on PATH AND PyPI network reachability -- neither is guaranteed in "
        "this sandbox (T26). This session's manual proof (moved "
        "mypy_boto3_s3 out of .venv, ran check + migrate + runserver, got "
        "HTTP 200) stands as evidence the underlying mechanism works; this "
        "test's job is to make that repeatable once unblocked, not to "
        "prove it for the first time here."
    ),
)
def test_verify_prod_parity_full_run_against_real_requirements_txt():
    exit_code = mod.main([])

    assert exit_code == 0
