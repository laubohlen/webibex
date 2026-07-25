"""R1 GREEN oracle: staticfiles/admin/* and staticfiles/filer/* must byte-match
what `collectstatic` produces for the currently pinned Django 5.2.16 /
django-filer 3.3.0.

`compare_subtree` is a from-scratch recursive walk (NOT `filecmp.dircmp`,
which is shallow and mtime-based) that compares only relative paths and byte
content -- mtime and mode bits are deliberately ignored so a real
`collectstatic` refresh (which always rewrites mtimes) doesn't produce false
drift, while genuine content/path drift is still caught.
"""

import os
from pathlib import Path

import pytest
from django.core.management import call_command
from django.test import override_settings

TRACKED_STATICFILES = Path(__file__).resolve().parent.parent.parent / "staticfiles"


def collect_into(tmp_path):
    """Run a real `collectstatic` into an empty tmp dir and return its Path."""
    with override_settings(STATIC_ROOT=tmp_path):
        call_command("collectstatic", interactive=False, verbosity=0)
    return tmp_path


def compare_subtree(tracked_dir, fresh_dir):
    """Recursively compare two directory trees by relative path + byte content.

    Returns a list of relative path strings that differ (missing on either
    side, or present on both with different bytes). Empty list means the
    subtrees are identical. Mtime and file mode bits are ignored by design.
    """
    tracked_dir = Path(tracked_dir)
    fresh_dir = Path(fresh_dir)

    tracked_files = {
        p.relative_to(tracked_dir).as_posix()
        for p in tracked_dir.rglob("*")
        if p.is_file()
    }
    fresh_files = {
        p.relative_to(fresh_dir).as_posix() for p in fresh_dir.rglob("*") if p.is_file()
    }

    diffs = set(tracked_files) ^ set(fresh_files)

    for rel_path in tracked_files & fresh_files:
        tracked_bytes = (tracked_dir / rel_path).read_bytes()
        fresh_bytes = (fresh_dir / rel_path).read_bytes()
        if tracked_bytes != fresh_bytes:
            diffs.add(rel_path)

    return sorted(diffs)


# ---------------------------------------------------------------------------
# T01-T02: the R1 GREEN oracle (RED before the real refresh, GREEN after).
# ---------------------------------------------------------------------------
@pytest.mark.django_db
def test_admin_staticfiles_match_fresh_collectstatic(tmp_path):
    fresh = collect_into(tmp_path)

    diffs = compare_subtree(TRACKED_STATICFILES / "admin", fresh / "admin")

    assert diffs == []


@pytest.mark.django_db
def test_filer_staticfiles_match_fresh_collectstatic(tmp_path):
    fresh = collect_into(tmp_path)

    diffs = compare_subtree(TRACKED_STATICFILES / "filer", fresh / "filer")

    assert diffs == []


# ---------------------------------------------------------------------------
# T03-T11: meta-oracles -- prove compare_subtree itself is a real detector,
# not a vacuously-passing stub. These do not depend on real tree staleness.
# ---------------------------------------------------------------------------
def test_compare_subtree_detects_extra_file_in_fresh(tmp_path):
    tracked = tmp_path / "tracked"
    fresh = tmp_path / "fresh"
    tracked.mkdir()
    fresh.mkdir()
    (tracked / "a.txt").write_bytes(b"same")
    (fresh / "a.txt").write_bytes(b"same")
    (fresh / "extra.txt").write_bytes(b"new")

    diffs = compare_subtree(tracked, fresh)

    assert diffs == ["extra.txt"]


def test_compare_subtree_detects_orphan_file_in_tracked(tmp_path):
    tracked = tmp_path / "tracked"
    fresh = tmp_path / "fresh"
    tracked.mkdir()
    fresh.mkdir()
    (tracked / "a.txt").write_bytes(b"same")
    (fresh / "a.txt").write_bytes(b"same")
    (tracked / "orphan.txt").write_bytes(b"stale")

    diffs = compare_subtree(tracked, fresh)

    assert diffs == ["orphan.txt"]


def test_compare_subtree_detects_content_mismatch_same_path(tmp_path):
    tracked = tmp_path / "tracked"
    fresh = tmp_path / "fresh"
    tracked.mkdir()
    fresh.mkdir()
    (tracked / "a.txt").write_bytes(b"old content")
    (fresh / "a.txt").write_bytes(b"new content")

    diffs = compare_subtree(tracked, fresh)

    assert diffs == ["a.txt"]


def test_compare_subtree_detects_nested_subdirectory_drift(tmp_path):
    tracked = tmp_path / "tracked"
    fresh = tmp_path / "fresh"
    (tracked / "css").mkdir(parents=True)
    (fresh / "css").mkdir(parents=True)
    (tracked / "js").mkdir(parents=True)
    (fresh / "js").mkdir(parents=True)
    (tracked / "css" / "a.css").write_bytes(b"old")
    (fresh / "css" / "a.css").write_bytes(b"new")
    (tracked / "js" / "b.js").write_bytes(b"identical")
    (fresh / "js" / "b.js").write_bytes(b"identical")

    diffs = compare_subtree(tracked, fresh)

    assert diffs == ["css/a.css"]


def test_compare_subtree_identical_dirs_returns_empty(tmp_path):
    tracked = tmp_path / "tracked"
    fresh = tmp_path / "fresh"
    tracked.mkdir()
    fresh.mkdir()
    (tracked / "a.txt").write_bytes(b"same")
    (fresh / "a.txt").write_bytes(b"same")

    diffs = compare_subtree(tracked, fresh)

    assert diffs == []


def test_compare_subtree_ignores_mtime_difference(tmp_path):
    tracked = tmp_path / "tracked"
    fresh = tmp_path / "fresh"
    tracked.mkdir()
    fresh.mkdir()
    (tracked / "a.txt").write_bytes(b"same")
    (fresh / "a.txt").write_bytes(b"same")

    # Touch only the fresh copy to a different mtime -- must not register as drift.
    os.utime(fresh / "a.txt", (1_000_000, 1_000_000))

    diffs = compare_subtree(tracked, fresh)

    assert diffs == []


def test_compare_subtree_scope_confinement_ignores_sibling_drift(tmp_path):
    """Drift injected in a sibling subtree not passed to compare_subtree
    must not affect the result of comparing a separate, clean pair."""
    root = tmp_path
    admin_tracked = root / "admin_tracked"
    admin_fresh = root / "admin_fresh"
    filer_tracked = root / "filer_tracked"
    filer_fresh = root / "filer_fresh"
    for d in (admin_tracked, admin_fresh, filer_tracked, filer_fresh):
        d.mkdir()

    (admin_tracked / "a.js").write_bytes(b"same")
    (admin_fresh / "a.js").write_bytes(b"same")

    # Sibling "filer"-like pair has drift -- must not leak into the admin comparison.
    (filer_tracked / "b.js").write_bytes(b"old")
    (filer_fresh / "b.js").write_bytes(b"new")

    diffs = compare_subtree(admin_tracked, admin_fresh)

    assert diffs == []


def test_compare_subtree_identical_bytes_different_mode_returns_empty(tmp_path):
    tracked = tmp_path / "tracked"
    fresh = tmp_path / "fresh"
    tracked.mkdir()
    fresh.mkdir()
    tracked_file = tracked / "a.sh"
    fresh_file = fresh / "a.sh"
    tracked_file.write_bytes(b"same bytes")
    fresh_file.write_bytes(b"same bytes")

    tracked_file.chmod(0o755)
    fresh_file.chmod(0o644)

    diffs = compare_subtree(tracked, fresh)

    assert diffs == []


# ---------------------------------------------------------------------------
# T12: orphan-cleanup regression guard (the reason `--clear` matters).
# ---------------------------------------------------------------------------
@pytest.mark.django_db
@pytest.mark.parametrize("subtree", ["admin", "filer"])
def test_tracked_and_fresh_path_sets_match(tmp_path, subtree):
    fresh = collect_into(tmp_path)

    tracked_paths = {
        p.relative_to(TRACKED_STATICFILES / subtree).as_posix()
        for p in (TRACKED_STATICFILES / subtree).rglob("*")
        if p.is_file()
    }
    fresh_paths = {
        p.relative_to(fresh / subtree).as_posix()
        for p in (fresh / subtree).rglob("*")
        if p.is_file()
    }

    assert tracked_paths == fresh_paths


# ---------------------------------------------------------------------------
# T10: structural invariant -- no symlinks under the tracked admin/filer trees.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("subtree", ["admin", "filer"])
def test_tracked_staticfiles_have_no_symlinks(subtree):
    root = TRACKED_STATICFILES / subtree
    symlinks = [str(p) for p in root.rglob("*") if p.is_symlink()]

    assert symlinks == []


# ---------------------------------------------------------------------------
# T13: advisory only -- see docstring. Downgraded to skip per code-analyst's
# own guidance: building an exhaustive "known good" allowlist of every path
# STATICFILES_DIRS/app static can produce (including third-party apps like
# django-allauth, easy_thumbnails, django_extensions, debug_toolbar) would be
# a brittle, high-maintenance allowlist for near-zero incremental signal --
# T01/T02/T12 already give a genuine, exhaustive oracle for the admin/filer
# subtrees this CR cares about.
# ---------------------------------------------------------------------------
@pytest.mark.skip(
    reason=(
        "Advisory only (code-analyst T13): cross-checking `git ls-files "
        "staticfiles/` against a hand-maintained 'known good' allowlist of "
        "every app that can contribute static files (django-allauth, "
        "easy_thumbnails, django_extensions, debug_toolbar, this project's "
        "own static/) would require an allowlist that silently goes stale "
        "every time a new app is added -- the same brittle-fixture failure "
        "mode documented in mutation-testing.md's stale-fixture pattern. "
        "T01/T02 (byte-exact) + T12 (path-set exact) already give a genuine, "
        "exhaustive R1 oracle for the admin/filer subtrees in scope for this "
        "CR; this advisory check adds no additional signal."
    )
)
def test_advisory_git_ls_files_staticfiles_explainable():
    pass
