"""T01-T08: regression coverage for core/b2_utils.py's TYPE_CHECKING guard.

Follow-up to the production `ModuleNotFoundError` outage: `boto3-stubs[s3]`
(which provides the `mypy_boto3_s3` stub package) is dev-only
(requirements-dev.txt), never installed where Railway deploys
(requirements.txt alone). `core/b2_utils.py` imports `mypy_boto3_s3` names
only under `if TYPE_CHECKING:` (never executed at runtime) with quoted
forward-ref annotations, so importing the module must never require the
stub package to be present.

This file proves that guard actually works, by literally simulating "the
stub package is not installed" -- poisoning `mypy_boto3_s3` in
`sys.modules` (see `poison_modules`, tests/conftest.py) and forcing a
fresh import of `core.b2_utils`. A companion "control" test (T02) proves
the poison mechanism itself is lethal against an UN-guarded module, so a
green T01 can't be explained by a broken/no-op poison -- and T04 proves
the poison mechanism also correctly handles an already-cached submodule
(a real gap: poisoning only the parent name does NOT block
`from parent.sub import X` when `parent.sub` is already imported).
"""

from __future__ import annotations

import ast
import importlib
import inspect
import sys
from pathlib import Path

import pytest

_B2_UTILS_PATH = Path(__file__).resolve().parents[2] / "core" / "b2_utils.py"


def _fresh_import_b2_utils():
    sys.modules.pop("core.b2_utils", None)
    return importlib.import_module("core.b2_utils")


# T01 -------------------------------------------------------------------
def test_b2_utils_imports_clean_with_mypy_boto3_s3_poisoned(poison_modules):
    with poison_modules("mypy_boto3_s3"):
        module = _fresh_import_b2_utils()

    assert module.get_b2_resource is not None


# T02 -------------------------------------------------------------------
def test_control_unguarded_import_of_poisoned_name_raises(poison_modules, tmp_path):
    """Control: a module that imports the poisoned name UNCONDITIONALLY
    (no TYPE_CHECKING guard) must raise -- proves the poison mechanism
    itself is lethal, so T01's green result isn't explained by a vacuous
    no-op poison."""
    victim = tmp_path / "unguarded_b2_utils_victim.py"
    victim.write_text("from mypy_boto3_s3 import S3ServiceResource\n")
    sys.path.insert(0, str(tmp_path))
    try:
        with poison_modules("mypy_boto3_s3"):
            sys.modules.pop("unguarded_b2_utils_victim", None)
            with pytest.raises(ModuleNotFoundError):
                importlib.import_module("unguarded_b2_utils_victim")
    finally:
        sys.path.remove(str(tmp_path))
        sys.modules.pop("unguarded_b2_utils_victim", None)


# T03 -------------------------------------------------------------------
def test_cache_bust_required_for_poison_to_take_effect(poison_modules):
    """Cache-bust is real: without evicting `core.b2_utils` from
    `sys.modules` first, re-importing it just returns the already-cached
    object -- the poison never gets exercised at all. Only after evicting
    it does a genuinely fresh import happen."""
    module_before = importlib.import_module("core.b2_utils")

    with poison_modules("mypy_boto3_s3"):
        module_same = importlib.import_module("core.b2_utils")
        assert module_same is module_before

        sys.modules.pop("core.b2_utils", None)
        module_fresh = importlib.import_module("core.b2_utils")
        assert module_fresh is not module_before


# T04 -------------------------------------------------------------------
def test_poison_recursive_covers_cached_submodules(poison_modules):
    """F2: `sys.modules['mypy_boto3_s3'] = None` alone does NOT block
    `from mypy_boto3_s3.type_defs import X` when that submodule is already
    cached -- `core.b2_utils` imports exactly that submodule
    (`ObjectIdentifierTypeDef`). The `poison_modules` fixture's
    `recursive=True` default must also poison every already-cached
    `f"{name}."`-prefixed key, or this whole regression suite would give
    false confidence."""
    import mypy_boto3_s3.type_defs  # noqa: F401 -- pre-cache the real submodule

    with poison_modules("mypy_boto3_s3", recursive=False):
        # Non-recursive poison: the submodule is still cached and importable.
        assert importlib.import_module("mypy_boto3_s3.type_defs") is not None

    with (
        poison_modules("mypy_boto3_s3", recursive=True),
        pytest.raises(ModuleNotFoundError),
    ):
        importlib.import_module("mypy_boto3_s3.type_defs")


# T05 -------------------------------------------------------------------
def test_public_api_intact_under_poison(poison_modules):
    with poison_modules("mypy_boto3_s3"):
        module = _fresh_import_b2_utils()

    assert callable(module.get_b2_resource)
    assert callable(module.download_file)
    assert callable(module.delete_files)
    assert callable(module.check_file_exists)
    assert list(inspect.signature(module.get_b2_resource).parameters) == [
        "endpoint",
        "key_id",
        "application_key",
    ]


# T06 -------------------------------------------------------------------
def test_sys_modules_restored_after_poison_context(poison_modules):
    snapshot = dict(sys.modules)

    with poison_modules("mypy_boto3_s3"):
        _fresh_import_b2_utils()
        assert sys.modules["mypy_boto3_s3"] is None

    assert dict(sys.modules) == snapshot


# T07 -------------------------------------------------------------------
def test_stub_imports_are_type_checking_only_and_quoted():
    source = _B2_UTILS_PATH.read_text()
    tree = ast.parse(source)

    type_checking_import_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and isinstance(node.test, ast.Name):
            if node.test.id != "TYPE_CHECKING":
                continue
            for stmt in node.body:
                if isinstance(stmt, ast.ImportFrom):
                    type_checking_import_names.update(
                        alias.name for alias in stmt.names
                    )

    assert type_checking_import_names, "no TYPE_CHECKING import block found at all"
    assert "S3ServiceResource" in type_checking_import_names
    assert "ObjectIdentifierTypeDef" in type_checking_import_names

    # S3ServiceResource is used as a function return-type annotation --
    # those ARE evaluated eagerly at def time regardless of TYPE_CHECKING,
    # so it must stay a quoted forward-ref or the module would NameError at
    # import time. Verified: `def f() -> Undefined: ...` raises immediately.
    assert '"S3ServiceResource"' in source

    # ObjectIdentifierTypeDef, by contrast, is used only as a local
    # variable annotation inside a function body -- CPython never evaluates
    # those at runtime (verified empirically: `def f(): x: Undefined = 1`
    # does not raise), so quoting it would be redundant. ruff's UP037 rule
    # agrees and flags/auto-fixes a quoted local-variable annotation --
    # assert it stays unquoted so this test and ruff never fight each other.
    assert "list[ObjectIdentifierTypeDef]" in source
    assert '"ObjectIdentifierTypeDef"' not in source


# T08 -------------------------------------------------------------------
def test_poisoning_unrelated_name_is_a_noop_control(poison_modules):
    """Negative control: poisoning a name core.b2_utils never references
    at all must have zero effect on the import -- if it somehow broke the
    import, that would indicate the test harness itself is broken, not
    core.b2_utils, and would mean T01's green result can't be trusted."""
    with poison_modules("this_name_does_not_exist_anywhere_xyz_b2_utils_control"):
        module = _fresh_import_b2_utils()

    assert module.get_b2_resource is not None
