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

    def __init__(self, stdout="", returncode=0):
        self.stdout = stdout
        self.returncode = returncode


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
