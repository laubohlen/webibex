"""P0 tests for scripts/db_restore_drill.py: dump_encrypted.

Uses a monkeypatched recording FakePopen (tests/scripts/conftest.py) --
no real pg_dump/openssl process is ever spawned.
"""

from __future__ import annotations


import pytest

from scripts.db_restore_drill import dump_encrypted

pytestmark = pytest.mark.spec(
    ref="docs/security-remediation-plan.md#gate-restore-drill-required"
)

_SOURCE_DSN = "postgresql://alice:s3cret@dbhost.example.invalid:5432/webibex"


def _patch_binaries(monkeypatch):
    import scripts.db_restore_drill as mod

    monkeypatch.setattr(
        mod.shutil,
        "which",
        lambda name: f"/usr/bin/{name}" if name in ("pg_dump", "openssl") else None,
    )


def test_dump_encrypted_exact_argv_and_shell_false(
    tmp_path, monkeypatch, fake_popen_cls
):
    import scripts.db_restore_drill as mod

    monkeypatch.setenv("DB_DUMP_PASSPHRASE", "correct-horse-battery-staple")
    _patch_binaries(monkeypatch)
    monkeypatch.setattr(mod.subprocess, "Popen", fake_popen_cls)

    out_path = tmp_path / "dump.enc"
    dump_encrypted(_SOURCE_DSN, out_path)

    assert len(fake_popen_cls.instances) == 2
    p1, p2 = fake_popen_cls.instances

    assert p1.argv == ["/usr/bin/pg_dump", "-Fc", "-w", "--no-owner", "--no-privileges"]
    assert isinstance(p1.argv, list)
    assert "shell" not in p1.kwargs or p1.kwargs["shell"] is False

    assert p2.argv[:5] == [
        "/usr/bin/openssl",
        "enc",
        "-aes-256-cbc",
        "-pbkdf2",
        "-iter",
    ]
    assert "600000" in p2.argv
    assert "-salt" in p2.argv
    assert "-pass" in p2.argv
    assert "env:DB_DUMP_PASSPHRASE" in p2.argv
    assert "-out" in p2.argv
    assert str(out_path) in p2.argv
    assert "shell" not in p2.kwargs or p2.kwargs["shell"] is False


def test_dump_encrypted_env_separation(tmp_path, monkeypatch, fake_popen_cls):
    import scripts.db_restore_drill as mod

    monkeypatch.setenv("DB_DUMP_PASSPHRASE", "correct-horse-battery-staple")
    _patch_binaries(monkeypatch)
    monkeypatch.setattr(mod.subprocess, "Popen", fake_popen_cls)

    out_path = tmp_path / "dump.enc"
    dump_encrypted(_SOURCE_DSN, out_path)

    p1, p2 = fake_popen_cls.instances
    pg_env = p1.kwargs["env"]
    ssl_env = p2.kwargs["env"]

    assert pg_env["PGHOST"] == "dbhost.example.invalid"
    assert pg_env["PGPASSWORD"] == "s3cret"
    assert "DB_DUMP_PASSPHRASE" not in pg_env

    assert ssl_env["DB_DUMP_PASSPHRASE"] == "correct-horse-battery-staple"
    assert "PGPASSWORD" not in ssl_env
    assert "PGHOST" not in ssl_env

    # neither secret ever appears in argv
    for argv in (p1.argv, p2.argv):
        assert "s3cret" not in argv
        assert "correct-horse-battery-staple" not in argv


def test_dump_encrypted_pg_dump_failure_raises_and_unlinks(
    tmp_path, monkeypatch, fake_popen_cls
):
    import scripts.db_restore_drill as mod

    monkeypatch.setenv("DB_DUMP_PASSPHRASE", "pw")
    _patch_binaries(monkeypatch)

    def make_popen(argv, **kwargs):
        rc = 1 if "pg_dump" in argv[0] else 0
        return fake_popen_cls(argv, returncode=rc, **kwargs)

    monkeypatch.setattr(mod.subprocess, "Popen", make_popen)

    out_path = tmp_path / "dump.enc"
    with pytest.raises(RuntimeError):
        dump_encrypted(_SOURCE_DSN, out_path)
    assert not out_path.exists()


def test_dump_encrypted_openssl_failure_raises_and_unlinks(
    tmp_path, monkeypatch, fake_popen_cls
):
    import scripts.db_restore_drill as mod

    monkeypatch.setenv("DB_DUMP_PASSPHRASE", "pw")
    _patch_binaries(monkeypatch)

    def make_popen(argv, **kwargs):
        rc = 1 if "openssl" in argv[0] else 0
        return fake_popen_cls(argv, returncode=rc, **kwargs)

    monkeypatch.setattr(mod.subprocess, "Popen", make_popen)

    out_path = tmp_path / "dump.enc"
    with pytest.raises(RuntimeError):
        dump_encrypted(_SOURCE_DSN, out_path)
    assert not out_path.exists()


def test_dump_encrypted_unlink_tolerates_already_absent_file(
    tmp_path, monkeypatch, fake_popen_cls
):
    """Failure branch must not itself raise even if the artifact never
    made it to disk (e.g. pg_dump crashed before openssl wrote anything)."""
    import scripts.db_restore_drill as mod

    monkeypatch.setenv("DB_DUMP_PASSPHRASE", "pw")
    _patch_binaries(monkeypatch)

    def make_popen(argv, **kwargs):
        return fake_popen_cls(argv, returncode=1, **kwargs)

    monkeypatch.setattr(mod.subprocess, "Popen", make_popen)

    out_path = tmp_path / "dump.enc"
    with pytest.raises(RuntimeError):
        dump_encrypted(_SOURCE_DSN, out_path)
    # calling again (file still absent after cleanup) must not crash on unlink
    with pytest.raises(RuntimeError):
        dump_encrypted(_SOURCE_DSN, out_path)


def test_dump_encrypted_closes_parent_stdout_copy_before_wait(
    tmp_path, monkeypatch, fake_popen_cls
):
    import scripts.db_restore_drill as mod

    monkeypatch.setenv("DB_DUMP_PASSPHRASE", "pw")
    _patch_binaries(monkeypatch)
    monkeypatch.setattr(mod.subprocess, "Popen", fake_popen_cls)

    out_path = tmp_path / "dump.enc"
    dump_encrypted(_SOURCE_DSN, out_path)

    p1, _p2 = fake_popen_cls.instances
    assert p1.stdout is not None
    assert p1.stdout.closed is True


def test_dump_encrypted_artifact_mode_0600(tmp_path, monkeypatch, fake_popen_cls):
    import scripts.db_restore_drill as mod

    monkeypatch.setenv("DB_DUMP_PASSPHRASE", "pw")
    _patch_binaries(monkeypatch)
    monkeypatch.setattr(mod.subprocess, "Popen", fake_popen_cls)

    out_path = tmp_path / "dump.enc"
    dump_encrypted(_SOURCE_DSN, out_path)

    mode = out_path.stat().st_mode & 0o777
    assert mode == 0o600


def test_dump_encrypted_refuses_to_clobber_existing_file(
    tmp_path, monkeypatch, fake_popen_cls
):
    import scripts.db_restore_drill as mod

    monkeypatch.setenv("DB_DUMP_PASSPHRASE", "pw")
    _patch_binaries(monkeypatch)
    monkeypatch.setattr(mod.subprocess, "Popen", fake_popen_cls)

    out_path = tmp_path / "dump.enc"
    out_path.write_bytes(b"pre-existing content")

    with pytest.raises(FileExistsError):
        dump_encrypted(_SOURCE_DSN, out_path)
    # must not have touched/deleted the pre-existing file
    assert out_path.read_bytes() == b"pre-existing content"
    assert len(fake_popen_cls.instances) == 0
