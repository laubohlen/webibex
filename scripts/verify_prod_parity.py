#!/usr/bin/env python3
"""Slow, opt-in end-to-end production-dependency-parity gate (R3).

Follow-up to the production `ModuleNotFoundError` outage: `requirements.
txt` alone (the set Railway actually installs) must be sufficient to
import every module Django touches on startup AND serve one real HTTP
request. `tests/webibex/test_prod_dependency_parity.py` (R2) proves the
"import" half of that fast, on every default `pytest` run, by poisoning
dev-only package names in-process. This script proves the FULL claim,
slowly, end to end, in a genuinely isolated environment:

  1. Resolve the runtime.txt-pinned Python interpreter (`python3.12`) on
     PATH -- never `sys.executable`, which could silently be a different
     major.minor and invalidate the whole parity claim.
  2. Copy the repo tree into an isolated scratch directory (never the
     developer's real `db.sqlite3` -- `webibex/settings.py` hardcodes
     `BASE_DIR / "db.sqlite3"` unconditionally, and R8 forbids adding an
     env-var override there for this CR).
  3. Create a venv there with that pinned interpreter, `pip install -r
     requirements.txt` ONLY (never requirements-dev.txt).
  4. Run `manage.py check` + `manage.py migrate --run-syncdb` against the
     throwaway sqlite DB.
  5. Boot `manage.py runserver`, GET `/` once, assert HTTP 200 with no
     traceback in the server log.
  6. Always tear down the scratch dir (venv + sqlite + repo copy) and stop
     the server, even on a mid-run exception.

No secret is ever accepted on argv (this script needs none -- only
paths/versions/network config). The child subprocess env is an explicit
allowlist (see `_build_child_env`), NEVER `{**os.environ, **overrides}` --
protects against a future `.env` (none exists in this repo today, see
`root conftest.py`'s module docstring) silently leaking real Railway/B2
creds into this drill. Refuses to run under `ENVIRONMENT=production`,
before any side effect. The scratch directory always lives under the
repo's gitignored `tmp/` (or entirely outside the repo) -- never under
version control.

Usage:
    scripts/verify_prod_parity.py [--python-version 3.12.5]
        [--scratch-dir tmp/verify_prod_parity_scratch] [--host 127.0.0.1]
        [--port 8734] [--timeout 180]
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_RUNTIME_TXT = _REPO_ROOT / "runtime.txt"
_REQUIREMENTS_TXT = _REPO_ROOT / "requirements.txt"
_TMP_DIR = _REPO_ROOT / "tmp"
_DEFAULT_SCRATCH_DIR = _TMP_DIR / "verify_prod_parity_scratch"

# F11: webibex/settings.py's ALLOWED_HOSTS = ["wibex.up.railway.app",
# "127.0.0.1", "localhost:8000"] -- plain "localhost" is ABSENT (Django
# strips the port before comparing, so "localhost:8000" never matches a
# bare "localhost" Host header). This tool must probe 127.0.0.1 or it
# gets HTTP 400, never "localhost".
_PROBE_HOST = "127.0.0.1"
_DEFAULT_PORT = 8734
_DEFAULT_TIMEOUT = 180

_TRACEBACK_MARKER = "Traceback (most recent call last)"

_PYTHON_VERSION_RE = re.compile(r"\Apython-(\d+\.\d+\.\d+)\Z")

# F12: matches root conftest.py's placeholder values EXACTLY -- never a
# diverging or newly-invented credential-shaped literal. These are the
# ONLY env vars this script's Django child process needs; nothing else is
# ever read from the parent's ambient os.environ (see _build_child_env).
_DJANGO_TEST_ENV_OVERRIDES: dict[str, str] = {
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

# Explicit allowlist of ambient os.environ vars forwarded to every child
# subprocess -- NEVER `{**os.environ, **overrides}` (T22's counter-input:
# that exact pattern would silently forward any real secret a developer's
# shell/.env happens to have set).
_CHILD_ENV_ALLOWLIST: tuple[str, ...] = ("HOME", "PATH", "LANG", "LC_ALL")

# Repo-tree copy excludes (see copy_repo_tree) -- venvs, caches, media,
# git metadata, and the developer's real sqlite DB never get copied into
# the isolated scratch tree.
_COPY_EXCLUDES = frozenset(
    {
        ".git",
        ".venv",
        ".venv-mac",
        "tmp",
        "node",
        "media",
        "staticfiles",
        "__pycache__",
        ".pytest_cache",
        ".worktrees",
        "db.sqlite3",
    }
)


def _emit(message: str, *, err: bool = False) -> None:
    """CLI entry-point output meant for the user -- the documented
    print() exception in python.md, mirroring scripts/db_restore_drill.py
    and scripts/run_local_e2e_server.py's own print-based user-facing
    output. Suppression scoped to this single helper."""
    print(message, file=sys.stderr if err else sys.stdout, flush=True)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ParityConfig:
    python_version: str
    scratch_dir: Path
    host: str = _PROBE_HOST
    port: int = _DEFAULT_PORT
    timeout: int = _DEFAULT_TIMEOUT

    @property
    def venv_dir(self) -> Path:
        return self.scratch_dir / "venv"

    @property
    def repo_dir(self) -> Path:
        return self.scratch_dir / "repo"


def read_runtime_python_version(runtime_txt: Path = _RUNTIME_TXT) -> str:
    text = runtime_txt.read_text().strip()
    match = _PYTHON_VERSION_RE.match(text)
    if not match:
        raise ValueError(f"unrecognized runtime.txt contents: {text!r}")
    return match.group(1)


def default_config() -> ParityConfig:
    return ParityConfig(
        python_version=read_runtime_python_version(),
        scratch_dir=_DEFAULT_SCRATCH_DIR,
    )


# ---------------------------------------------------------------------------
# Refusal guards -- run BEFORE any side effect.
# ---------------------------------------------------------------------------
def refuse_if_production() -> None:
    if os.environ.get("ENVIRONMENT") == "production":
        raise RuntimeError(
            "refusing to run: ENVIRONMENT=production -- this tool creates "
            "a throwaway venv + sqlite DB + dev server; never point it at "
            "a real deployment"
        )


def _validate_scratch_dir(scratch_dir: Path) -> None:
    """Refuse a scratch_dir that would land under version control.
    Accepts any path under the repo's gitignored `tmp/` directory, or any
    path entirely outside the repo -- both are safe from `git add`.
    Rejects every other in-repo path (F14)."""
    resolved = scratch_dir.resolve()
    try:
        resolved.relative_to(_REPO_ROOT)
    except ValueError:
        return  # outside the repo entirely -- safe

    try:
        resolved.relative_to(_TMP_DIR)
    except ValueError as exc:
        raise ValueError(
            f"scratch_dir must live under {_TMP_DIR} or entirely outside "
            f"the repo (gitignored) -- got: {scratch_dir}"
        ) from exc


# ---------------------------------------------------------------------------
# Repo copy -- isolates BASE_DIR (and thus db.sqlite3) from the real repo.
# ---------------------------------------------------------------------------
def _ignore_scratch_copy(_dir: str, names: list[str]) -> set[str]:
    return {n for n in names if n in _COPY_EXCLUDES or n.endswith(".dump.enc")}


def copy_repo_tree(dest: Path) -> Path:
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(_REPO_ROOT, dest, ignore=_ignore_scratch_copy)
    return dest


# ---------------------------------------------------------------------------
# Pinned interpreter + venv.
# ---------------------------------------------------------------------------
def resolve_pinned_python(version: str) -> str:
    """Locate a `pythonX.Y` executable on PATH matching the pinned
    version's major.minor. Never falls back to `sys.executable` -- that
    could silently be a different major.minor and invalidate the whole
    parity claim."""
    major_minor = ".".join(version.split(".")[:2])
    candidate = shutil.which(f"python{major_minor}")
    if candidate is None:
        raise RuntimeError(
            f"python{major_minor} not found on PATH -- install it to run "
            f"the prod_parity gate (runtime.txt pins {version})"
        )
    return candidate


def create_venv(python_exe: str, venv_dir: Path) -> Path:
    _validate_scratch_dir(venv_dir)
    if venv_dir.exists():
        shutil.rmtree(venv_dir)
    venv_dir.parent.mkdir(parents=True, exist_ok=True)
    argv = [python_exe, "-m", "venv", str(venv_dir)]
    result = subprocess.run(  # noqa: S603 -- python_exe from resolve_pinned_python (shutil.which), fixed argv, shell=False
        argv, capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"venv creation failed (rc={result.returncode}): {result.stderr}"
        )
    return venv_dir / "bin" / "python"


def pip_install_requirements(
    venv_python: Path, requirements_path: Path = _REQUIREMENTS_TXT
) -> None:
    """`-r requirements.txt` ONLY -- NEVER requirements-dev.txt (T21).
    Full argv, no extra flags -- this exact call proves the exact set
    Railway installs is sufficient."""
    argv = [str(venv_python), "-m", "pip", "install", "-r", str(requirements_path)]
    result = subprocess.run(  # noqa: S603 -- venv_python from create_venv, fixed argv, shell=False
        argv, capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"pip install failed (rc={result.returncode}): {result.stderr}"
        )


# ---------------------------------------------------------------------------
# Child env.
# ---------------------------------------------------------------------------
def _build_child_env(overrides: dict[str, str]) -> dict[str, str]:
    """Explicit allowlist union with `overrides` -- NEVER
    `{**os.environ, **overrides}` (T22). Only the 4 vars in
    `_CHILD_ENV_ALLOWLIST` are read from the parent's `os.environ`;
    everything Django itself needs is always supplied explicitly via
    `overrides`, never inherited."""
    result: dict[str, str] = {
        name: value
        for name in _CHILD_ENV_ALLOWLIST
        if (value := os.environ.get(name)) is not None
    }
    result.update(overrides)
    return result


# ---------------------------------------------------------------------------
# manage.py check / migrate.
# ---------------------------------------------------------------------------
def run_manage_check(venv_python: Path, cwd: Path, env: dict[str, str]) -> None:
    argv = [str(venv_python), "manage.py", "check"]
    result = subprocess.run(  # noqa: S603 -- venv_python from create_venv, fixed argv, shell=False
        argv, cwd=str(cwd), env=env, capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"manage.py check failed (rc={result.returncode}): {result.stderr}"
        )


def run_manage_migrate(venv_python: Path, cwd: Path, env: dict[str, str]) -> None:
    argv = [str(venv_python), "manage.py", "migrate", "--run-syncdb"]
    result = subprocess.run(  # noqa: S603 -- venv_python from create_venv, fixed argv, shell=False
        argv, cwd=str(cwd), env=env, capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"manage.py migrate failed (rc={result.returncode}): {result.stderr}"
        )


# ---------------------------------------------------------------------------
# runserver + probe.
# ---------------------------------------------------------------------------
def start_runserver(
    venv_python: Path, cwd: Path, env: dict[str, str], config: ParityConfig
) -> tuple[subprocess.Popen[bytes], Path]:
    log_path = config.scratch_dir / "runserver.log"
    log_fh = open(log_path, "wb")  # noqa: SIM115 -- fh must outlive this function; owned by the caller's teardown
    argv = [
        str(venv_python),
        "manage.py",
        "runserver",
        f"{config.host}:{config.port}",
        "--noreload",
    ]
    process = subprocess.Popen(  # noqa: S603 -- venv_python from create_venv, fixed argv, shell=False
        argv, cwd=str(cwd), env=env, stdout=log_fh, stderr=subprocess.STDOUT
    )
    log_fh.close()  # child inherited its own fd; this process no longer needs it open
    return process, log_path


def probe_server(
    host: str, port: int, *, attempts: int = 30, delay: float = 1.0
) -> int:
    url = f"http://{host}:{port}/"
    last_exc: Exception | None = None
    for _ in range(attempts):
        try:
            with urlopen(url, timeout=5) as response:
                return response.status
        except HTTPError as exc:
            # server IS reachable, just returned non-2xx -- not a retry case
            return exc.code
        except URLError as exc:
            last_exc = exc
            time.sleep(delay)
    raise RuntimeError(f"server never became reachable at {url}: {last_exc}")


def log_contains_traceback(log_path: Path) -> bool:
    if not log_path.exists():
        return False
    return _TRACEBACK_MARKER in log_path.read_text(errors="replace")


def stop_server(process: subprocess.Popen[bytes], *, timeout: int = 10) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=timeout)


def teardown_scratch(scratch_dir: Path) -> None:
    _validate_scratch_dir(scratch_dir)
    shutil.rmtree(scratch_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Orchestration.
# ---------------------------------------------------------------------------
def run(config: ParityConfig | None = None) -> int:
    refuse_if_production()  # before ANY side effect (T38)
    if config is None:
        config = default_config()
    _validate_scratch_dir(config.scratch_dir)  # before ANY side effect (T37)

    process: subprocess.Popen[bytes] | None = None
    try:
        python_exe = resolve_pinned_python(config.python_version)
        copy_repo_tree(config.repo_dir)
        venv_python = create_venv(python_exe, config.venv_dir)
        pip_install_requirements(venv_python)

        child_env = _build_child_env(_DJANGO_TEST_ENV_OVERRIDES)
        run_manage_check(venv_python, config.repo_dir, child_env)
        run_manage_migrate(venv_python, config.repo_dir, child_env)

        process, log_path = start_runserver(
            venv_python, config.repo_dir, child_env, config
        )
        status = probe_server(
            config.host, config.port, attempts=max(1, config.timeout // 2)
        )

        if status != 200:
            _emit(f"FAIL: GET / returned HTTP {status}", err=True)
            return 1
        if log_contains_traceback(log_path):
            _emit("FAIL: traceback found in runserver log", err=True)
            return 1

        _emit(
            "PASS: prod-dependency-parity gate (requirements.txt alone is sufficient)"
        )
        return 0
    finally:
        if process is not None:
            stop_server(process)
        teardown_scratch(config.scratch_dir)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n", 1)[0])
    parser.add_argument("--python-version", default=None, help="override runtime.txt")
    parser.add_argument("--scratch-dir", type=Path, default=None)
    parser.add_argument("--host", default=_PROBE_HOST)
    parser.add_argument("--port", type=int, default=_DEFAULT_PORT)
    parser.add_argument("--timeout", type=int, default=_DEFAULT_TIMEOUT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = default_config()
    if args.python_version:
        config = dataclasses.replace(config, python_version=args.python_version)
    if args.scratch_dir:
        config = dataclasses.replace(config, scratch_dir=args.scratch_dir)
    config = dataclasses.replace(
        config, host=args.host, port=args.port, timeout=args.timeout
    )
    return run(config)


if __name__ == "__main__":
    sys.exit(main())
