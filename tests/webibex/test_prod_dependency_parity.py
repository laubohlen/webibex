"""T09-T16: fast, always-on production-dependency-parity gate (R2).

Follow-up to the production `ModuleNotFoundError` outage (`mypy_boto3_s3`,
a dev-only stub package, was silently required at runtime). This gate
dynamically enumerates every "dev-only" importable name in the current
environment, then poisons each one (see `poison_modules`,
tests/conftest.py) and forces a clean re-import sweep of every module
under `core`/`webibex`/`simple_landmarks`/`db_management` -- if any
first-party module secretly depends on a dev-only package, this test
fails loudly, in seconds, on every default `pytest` run (no venv/network
needed -- see R3's slow opt-in tier for the real end-to-end proof).

Dev-only name resolution -- IMPORTANT, read before touching this file:
the plan originally specified inverting `packages_distributions()`
against `requirements-dev.txt`'s declared names. That algorithm is
PROVABLY BROKEN in this repo (F1, empirically confirmed this session):
`boto3-stubs` has NO installed dist-info at all in `.venv` (declared but
never actually resolved as a top-level pip install target the way
`boto3-stubs[s3]==1.26.0` implies) -- the real provider of
`mypy_boto3_s3` is a SEPARATE, UNDECLARED transitive dist,
`mypy-boto3-s3==1.26.163`. Inverting against requirements-dev.txt NAMES
would silently no-op on exactly the package that caused the outage.

APPROVED FIX (complement derivation, user-approved this session): every
INSTALLED top-level importable name, MINUS the prod transitive closure
(every dist reachable from requirements.txt via installed `Requires-Dist`
metadata, direct + transitive), MINUS first-party repo top-level names
(F4: `scripts` collides with both this repo's own `scripts/` package and
a `types-awscrt` dev dependency), MINUS non-identifier names (F5:
`botocore-stubs` etc. can never be a real `sys.modules` key).

Expected noise: T13's full sweep fresh-reimports `core.models` /
`simple_landmarks.models`, which triggers Django's own
`RuntimeWarning: Model '...' was already registered` (app-registry
re-registration notice) -- harmless, does not fail the test, not
something this test suite can or should silence (it comes from Django's
model metaclass, not from any of this file's own code).
"""

from __future__ import annotations

import importlib
import importlib.metadata as im
import logging
import pkgutil
import sys
from pathlib import Path

import pytest
from django.contrib import admin
from packaging.requirements import InvalidRequirement, Requirement

import core
import db_management
import simple_landmarks
import webibex
from core.models import Animal, Location, Region, User
from simple_landmarks.models import Landmark

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SWEPT_PACKAGES = (core, webibex, simple_landmarks, db_management)

# F3 (empirically confirmed this session): this module does REAL DATABASE
# WRITES (`IbexImage.objects.all()` + a loop with `.save(update_fields=...)`)
# at IMPORT TIME, and always raises against this repo's real data
# (`ValueError: time data 'SM01_noexifdata' does not match format
# '%y_%m_%d_%H%M%S'`) -- it is a one-shot data-migration script, not a
# library module. Documented exclusion, mirrors root conftest.py's
# `_MOTO_S3_MISUSE_ALLOWLIST` pattern. `db_management` itself (its
# `__init__.py`) stays in the sweep.
_SWEEP_EXCLUSIONS: dict[str, str] = {
    "db_management.populate_created_at_field": (
        "module is a one-shot data-migration script, not a library "
        "module -- executes DB writes at import time"
    ),
}

_FIRST_PARTY_TOP_LEVEL: frozenset[str] = frozenset(
    {"core", "webibex", "simple_landmarks", "db_management", "scripts", "tests"}
)


def _normalize(name: str) -> str:
    return name.lower().replace("_", "-")


def _prod_transitive_closure() -> set[str]:
    """Normalized dist names reachable from requirements.txt, direct and
    transitive, via INSTALLED-metadata `Requires-Dist` -- deliberately NOT
    requirements-dev.txt's declared names (see module docstring / F1)."""
    seed = []
    for line in (_REPO_ROOT / "requirements.txt").read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        seed.append(Requirement(stripped).name)

    seen: set[str] = set()
    queue = list(seed)
    while queue:
        name = queue.pop()
        key = _normalize(name)
        if key in seen:
            continue
        seen.add(key)
        try:
            dist = im.distribution(name)
        except im.PackageNotFoundError:
            continue
        for req_str in dist.requires or []:
            try:
                req = Requirement(req_str)
            except InvalidRequirement:
                logger.debug("skipping malformed Requires-Dist entry: %r", req_str)
                continue
            queue.append(req.name)
    return seen


def resolve_dev_only_importable_names() -> frozenset[str]:
    """Complement derivation (F1 fix, user-approved). Fail-secure: raises
    `RuntimeError` instead of ever returning an empty set -- an empty
    result here would silently no-op the entire sweep below (a
    "green with zero items checked" false pass)."""
    closure = _prod_transitive_closure()
    pd = im.packages_distributions()

    prod_importables: set[str] = {
        importable
        for importable, dists in pd.items()
        if any(_normalize(d) in closure for d in dists)
    }

    all_importables = set(pd.keys())
    dev_only = {
        name
        for name in (all_importables - prod_importables - _FIRST_PARTY_TOP_LEVEL)
        if name.isidentifier()
    }
    if not dev_only:
        raise RuntimeError(
            "dev-only importable-name resolution produced an EMPTY set -- "
            "refusing to silently no-op the prod-dependency-parity sweep"
        )
    return frozenset(dev_only)


def _discover_sweep_targets() -> list[str]:
    """Every module under the 4 first-party packages via
    `pkgutil.walk_packages`, minus `_SWEEP_EXCLUSIONS` (F3)."""
    names = []
    for pkg in _SWEPT_PACKAGES:
        names.append(pkg.__name__)
        names.extend(
            info.name
            for info in pkgutil.walk_packages(pkg.__path__, prefix=pkg.__name__ + ".")
        )
    return [name for name in names if name not in _SWEEP_EXCLUSIONS]


# `core.admin` registers User/Animal/Region/Location via the bare
# `@admin.register(...)` decorator (no preceding `unregister()`) -- a
# fresh re-import of these 4 collides with `AlreadyRegistered` against
# the registration already done once by Django's own startup (root
# conftest.py's `django.setup()` -> admin autodiscover). `simple_
# landmarks.admin` registers Landmark the same bare way. `core.admin`'s
# Folder/Image entries, by contrast, are always `unregister()`-then-
# `register()`'d in the SAME statement pair, so they're already
# reimport-safe and must be LEFT ALONE here -- popping them first would
# break the `unregister()` call, which requires the model to already be
# registered (confirmed empirically this session: this is a genuine,
# module-reimport-specific Django-admin gotcha, unrelated to dev-only-
# package poisoning).
_ADMIN_REGISTRY_RESET_TARGETS: dict[str, tuple] = {
    "core.admin": (User, Animal, Region, Location),
    "simple_landmarks.admin": (Landmark,),
}


def _reimport_sweep_target(target: str) -> None:
    """Evict + fresh-import a single sweep target, popping only the exact
    admin-registry entries (if any) that `target`'s own bare
    `@admin.register(...)` calls are about to re-add -- see
    `_ADMIN_REGISTRY_RESET_TARGETS`."""
    for model in _ADMIN_REGISTRY_RESET_TARGETS.get(target, ()):
        admin.site._registry.pop(model, None)
    sys.modules.pop(target, None)
    importlib.import_module(target)


@pytest.fixture(scope="module")
def dev_only_names() -> frozenset[str]:
    return resolve_dev_only_importable_names()


@pytest.fixture(scope="module")
def sweep_targets() -> list[str]:
    return _discover_sweep_targets()


# T09 -------------------------------------------------------------------
def test_dev_only_names_contains_mypy_boto3_s3(dev_only_names):
    assert dev_only_names, "empty dev-only set -- resolver is broken (see F1)"
    assert "mypy_boto3_s3" in dev_only_names


# T10 -------------------------------------------------------------------
@pytest.mark.parametrize(
    "prod_name", ["django", "boto3", "botocore", "requests", "numpy", "cv2", "PIL"]
)
def test_dev_only_names_excludes_prod_names(dev_only_names, prod_name):
    # Fail-secure counter-input: an empty dev_only set would make this
    # exclusion assertion vacuously true -- must never pass "for free".
    assert dev_only_names, "empty dev-only set: exclusion check would be vacuous"
    assert prod_name not in dev_only_names


# T11 -------------------------------------------------------------------
def test_dev_only_names_are_all_valid_identifiers(dev_only_names):
    assert dev_only_names
    for name in dev_only_names:
        assert name.isidentifier(), f"non-identifier name leaked through: {name!r}"


def test_dev_only_names_filters_non_identifier_dist_names(monkeypatch):
    """F5, with a fabricated collision: a hyphenated dist-derived name
    (`botocore-stubs`, never a valid `sys.modules` key) must be filtered
    out even when `packages_distributions()` itself reports it."""

    def _fake_packages_distributions():
        return {
            "botocore-stubs": ["botocore-stubs"],
            "mypy_boto3_s3": ["mypy-boto3-s3"],
        }

    monkeypatch.setattr(im, "packages_distributions", _fake_packages_distributions)

    result = resolve_dev_only_importable_names()

    assert "botocore-stubs" not in result
    assert "mypy_boto3_s3" in result


# T12 -------------------------------------------------------------------
@pytest.mark.parametrize(
    "first_party_name",
    ["scripts", "core", "webibex", "simple_landmarks", "db_management"],
)
def test_dev_only_names_excludes_first_party_names(dev_only_names, first_party_name):
    assert dev_only_names
    assert first_party_name not in dev_only_names


# T13 -------------------------------------------------------------------
def test_full_sweep_imports_clean_under_every_dev_only_poison(
    dev_only_names, sweep_targets, poison_modules
):
    # Fail-secure: both input sets must be non-empty, or this sweep would
    # be a vacuous "zero items checked" pass.
    assert dev_only_names, "empty dev-only set -- sweep would be vacuous"
    assert sweep_targets, "empty sweep-target set -- sweep would be vacuous"

    # All dev-only names poisoned TOGETHER in one pass (rather than one
    # pass per name) -- O(names + targets) instead of O(names * targets),
    # keeping this "fast, always-on" gate actually fast (F15: ~44 dev-only
    # names x 51 swept modules would otherwise mean ~2244 re-imports).
    # Equivalent regression coverage: a first-party module depending on
    # ANY dev-only package still fails to import with all of them
    # poisoned at once.
    admin_registry_snapshot = dict(admin.site._registry)
    try:
        with poison_modules(list(dev_only_names)):
            for target in sweep_targets:
                _reimport_sweep_target(target)
    finally:
        admin.site._registry.clear()
        admin.site._registry.update(admin_registry_snapshot)


# T14 -------------------------------------------------------------------
def test_sweep_excludes_populate_created_at_field_with_documented_reason(
    sweep_targets,
):
    excluded_name = "db_management.populate_created_at_field"
    assert excluded_name not in sweep_targets
    assert excluded_name in _SWEEP_EXCLUSIONS
    assert _SWEEP_EXCLUSIONS[excluded_name].strip()
    # db_management itself (the package's __init__.py) stays swept.
    assert "db_management" in sweep_targets


# T15 -------------------------------------------------------------------
def test_control_sweep_harness_detects_a_broken_module(
    poison_modules, tmp_path, dev_only_names, monkeypatch
):
    """Meta-test: proves the sweep harness itself would catch a real
    regression -- a synthetic first-party-shaped module that imports a
    dev-only name unconditionally must fail the sweep."""
    victim_name = next(iter(dev_only_names))
    pkg_dir = tmp_path / "victim_pkg_for_sweep_control"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text(f"import {victim_name}\n")
    monkeypatch.syspath_prepend(str(tmp_path))

    with poison_modules(list(dev_only_names)):
        sys.modules.pop("victim_pkg_for_sweep_control", None)
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("victim_pkg_for_sweep_control")


# T16 -------------------------------------------------------------------
def test_sys_modules_restored_when_sweep_raises_mid_iteration(
    poison_modules, dev_only_names
):
    snapshot = dict(sys.modules)
    victim_name = next(iter(dev_only_names))

    with pytest.raises(ModuleNotFoundError), poison_modules(list(dev_only_names)):
        raise ModuleNotFoundError(f"synthetic mid-sweep failure: {victim_name}")

    assert dict(sys.modules) == snapshot
