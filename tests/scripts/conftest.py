"""Shared fixtures for the `tests/scripts` package (db_restore_drill.py)."""

from __future__ import annotations

import subprocess
from typing import Any, ClassVar
from unittest import mock

import pytest


@pytest.fixture
def mock_railway_api(no_network):
    """Reconfigure the (already-patched, autouse) requests.post mock.

    Mirrors tests/conftest.py::mock_runpod -- reuses the same patch object
    installed by the root `no_network` fixture instead of double-patching
    `requests.post`.
    """
    post_patch, _resource_patch = no_network
    post_patch.side_effect = None
    post_patch.return_value = mock.Mock(status_code=200)
    return post_patch


# ---------------------------------------------------------------------------
# Duck-typed psycopg2 stand-ins (mirrors tests/conftest.py's RegionStub /
# ChipStub precedent) -- no real DB or psycopg2 connection needed for P0.
# ---------------------------------------------------------------------------
class FakeCursor:
    """Records every execute() call; replays queued fetchone()/fetchall()."""

    def __init__(self, fetchone_results=None, fetchall_results=None):
        self.execute_calls: list[tuple[str, Any]] = []
        self._fetchone_results = list(fetchone_results or [])
        self._fetchall_results = list(fetchall_results or [])

    def execute(self, query, params=None):
        self.execute_calls.append((query, params))

    def fetchone(self):
        if not self._fetchone_results:
            return None
        return self._fetchone_results.pop(0)

    def fetchall(self):
        if not self._fetchall_results:
            return []
        return self._fetchall_results.pop(0)

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


class FakeConnection:
    """Duck-typed stand-in for a psycopg2 connection."""

    def __init__(self, cursor: FakeCursor):
        self._cursor = cursor
        self.closed = False

    def cursor(self):
        return self._cursor

    def close(self):
        self.closed = True


@pytest.fixture
def fake_cursor_cls():
    return FakeCursor


@pytest.fixture
def fake_connection_cls():
    return FakeConnection


class FakeCompletedProcess:
    """Duck-typed stand-in for subprocess.CompletedProcess."""

    def __init__(self, stdout="", returncode=0, stderr=""):
        self.stdout = stdout
        self.returncode = returncode
        self.stderr = stderr


class FakePipe:
    """Duck-typed stand-in for a Popen.stdout pipe -- records close()."""

    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class FakePopen:
    """Records constructor args; replays a configured returncode on wait()."""

    instances: ClassVar[list[FakePopen]] = []

    def __init__(self, argv, returncode=0, **kwargs):
        self.argv = argv
        self.kwargs = kwargs
        self.returncode = returncode
        self.stdout = FakePipe() if kwargs.get("stdout") == subprocess.PIPE else None
        self.wait_called = False
        self.killed = False
        FakePopen.instances.append(self)

    def wait(self, timeout=None):
        self.wait_called = True
        return self.returncode

    def poll(self):
        return self.returncode if self.wait_called else None

    def kill(self):
        self.killed = True


@pytest.fixture
def fake_popen_cls():
    FakePopen.instances = []
    return FakePopen


class FakeRun:
    """Duck-typed stand-in for `subprocess.run`. Records every call
    (positional args, kwargs) and replays a queue of `FakeCompletedProcess`
    results -- or raises a queued exception instead, for simulating
    `TimeoutExpired`/`PermissionError`/`FileNotFoundError` from the
    `docker` binary itself.
    """

    def __init__(self):
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self._queue: list[Any] = []

    def queue_result(self, result: FakeCompletedProcess) -> None:
        self._queue.append(result)

    def queue_exception(self, exc: BaseException) -> None:
        self._queue.append(exc)

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if not self._queue:
            return FakeCompletedProcess()
        item = self._queue.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


@pytest.fixture
def fake_run():
    return FakeRun()


_DOCKER_ALLOWLIST_ONLY_VARS = (
    "DOCKER_CONTEXT",
    "DOCKER_CONFIG",
    "DOCKER_CERT_PATH",
    "DOCKER_TLS_VERIFY",
    "XDG_RUNTIME_DIR",
)


@pytest.fixture
def docker_env(monkeypatch):
    """Sets HOME/PATH/DOCKER_HOST + an unrelated sentinel var, deletes the
    other 5 docker-context allowlist members -- a controlled `os.environ`
    slice for `_docker_child_env` tests.
    """
    monkeypatch.setenv("HOME", "/home/tester")
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    monkeypatch.setenv("DOCKER_HOST", "unix:///var/run/docker.sock")
    monkeypatch.setenv("SENTINEL_UNRELATED", "should-not-appear")
    for name in _DOCKER_ALLOWLIST_ONLY_VARS:
        monkeypatch.delenv(name, raising=False)
    return {
        "HOME": "/home/tester",
        "PATH": "/usr/bin:/bin",
        "DOCKER_HOST": "unix:///var/run/docker.sock",
    }
