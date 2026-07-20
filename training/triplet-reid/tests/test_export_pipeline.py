"""Sandbox-executable regression guards for the triplet-reid TF2.18 export pipeline rebuild.

Covers the 7 test-spec scenarios that code-analyst flagged as runnable
without real TensorFlow or Docker: T06, T09, T10, T11, T12, T14, T22. All
other spec scenarios (T01/T02/T03/T05/T08/T13/T15-T19/T23/T24) require real
TF/Docker/network and are covered by the host runbook instead (see the
session report / ADR-export-pipeline.md).

None of these tests import tensorflow — they operate on source text (via
`ast`) and file hashes only.
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import pytest

# tests/ -> training/triplet-reid/ (this repo dir; nets/ and export_saved_model.py
# live here). T14/T22's fixture data (triplet-reid/_archive/, wibex_model_v03/)
# was never moved here -- it remains gitignored scratch under tmp/inference/,
# a sibling of training/, not of this directory. Hardcoded (not a relative
# .parent chain) because the two trees no longer share a parent directory.
REPO_ROOT = Path(__file__).resolve().parent.parent
INFERENCE_ROOT = REPO_ROOT.parents[1] / "tmp" / "inference"
NETS_TF2_DIR = REPO_ROOT / "nets"
EXPORT_SCRIPT = REPO_ROOT / "export_saved_model.py"
TRIPLET_REID_DIR = INFERENCE_ROOT / "triplet-reid"
WIBEX_MODEL_V03_DIR = INFERENCE_ROOT / "wibex_model_v03"


def _contrib_references(source: str, filename: str) -> list[str]:
    """Return AST-level `tf.contrib` / `tensorflow.contrib` references.

    AST-based (not substring/grep) deliberately: several ported files carry
    inline comments and a module docstring documenting the OLD
    `tf.contrib.slim` API being replaced (e.g. "PORTED: `tf.contrib.slim` ->
    `tf_slim`") — a naive text search over comments/docstrings would
    false-positive on those. `ast.walk` only inspects code nodes, so
    comments are invisible to it and docstring string contents are never
    interpreted as attribute/import references.
    """
    tree = ast.parse(source, filename=filename)
    hits: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == "contrib":
            hits.append(f"{filename}:{node.lineno}: attribute access '.contrib'")
        if isinstance(node, ast.ImportFrom) and node.module and "contrib" in node.module:
            hits.append(f"{filename}:{node.lineno}: 'from {node.module} import ...'")
        if isinstance(node, ast.Import):
            for alias in node.names:
                if "contrib" in alias.name:
                    hits.append(f"{filename}:{node.lineno}: 'import {alias.name}'")
    return hits


def _nets_tf2_py_files() -> list[Path]:
    return sorted(NETS_TF2_DIR.glob("*.py"))


def _dotted_call_name(node: ast.AST) -> str | None:
    """Reconstruct a dotted name (`tf.saved_model.save`) from a Call's func node."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted_call_name(node.value)
        return f"{base}.{node.attr}" if base is not None else None
    return None


def _called_dotted_names(source: str, filename: str) -> set[str]:
    """Dotted names of every function *call* in `source` (AST-based, not text search).

    Used instead of substring search for T10/T11 so that explanatory prose
    in comments/docstrings (e.g. this file's own module docstring naming
    `SavedModelBuilder` as the thing NOT to use, or a code comment
    explaining the `trainable_variables()` bug being fixed) can't produce a
    false failure — only an actual call in code counts.
    """
    tree = ast.parse(source, filename=filename)
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            dotted = _dotted_call_name(node.func)
            if dotted:
                names.add(dotted)
    return names


@pytest.mark.spec(ref="ADR-export-pipeline.md#r3-nets-port")
def test_nets_tf2_has_no_tf_contrib_references() -> None:
    """T06: zero `tf.contrib` code references remain across nets_tf2/*.py."""
    # Arrange
    py_files = _nets_tf2_py_files()
    assert py_files, "nets_tf2/ contains no .py files — staging dir not populated"

    # Act
    offenders: dict[str, list[str]] = {}
    for py_file in py_files:
        hits = _contrib_references(py_file.read_text(encoding="utf-8"), py_file.name)
        if hits:
            offenders[py_file.name] = hits

    # Assert
    assert not offenders, f"tf.contrib references remain: {offenders}"


@pytest.mark.spec(ref="ADR-export-pipeline.md#r3-mobilenet-completeness")
@pytest.mark.parametrize("filename", ["mobilenet_v1.py", "mobilenet_v1_1_224.py"])
def test_mobilenet_port_is_complete(filename: str) -> None:
    """T09: mobilenet files specifically (the ones v2adapted left unported) are clean.

    Subset of T06 scoped to the two files code-analyst found were
    byte-identical to the TF1 originals in `triplet-reid_v2adapted/nets/`
    (i.e. never actually ported there) — kept as a separate test so a
    regression here fails with an unambiguous, targeted name.
    """
    # Arrange
    py_file = NETS_TF2_DIR / filename
    assert py_file.exists(), f"expected ported file missing: {py_file}"

    # Act
    hits = _contrib_references(py_file.read_text(encoding="utf-8"), filename)

    # Assert
    assert not hits, f"{filename} still has tf.contrib references: {hits}"


@pytest.mark.spec(ref="ADR-export-pipeline.md#r4-savedmodel-save")
def test_export_uses_saved_model_save_not_builder() -> None:
    """T10: export_saved_model.py CALLS tf.saved_model.save(), never instantiates SavedModelBuilder.

    `tf.compat.v1.saved_model.builder.SavedModelBuilder` produces no
    `fingerprint.pb` and was implicated in historical export bugs (see ADR).
    AST-based (not raw substring search): this file's own module docstring
    names `SavedModelBuilder` as the thing NOT to use — a text-only check
    would false-fail on that legitimate documentation.
    """
    # Arrange
    source = EXPORT_SCRIPT.read_text(encoding="utf-8")

    # Act
    called = _called_dotted_names(source, EXPORT_SCRIPT.name)

    # Assert
    assert not any(name.endswith("SavedModelBuilder") for name in called), called
    assert any(name.endswith("saved_model.save") for name in called), called


@pytest.mark.spec(ref="ADR-export-pipeline.md#r4-r7-global-variables-capture")
def test_export_captures_global_variables_for_restore() -> None:
    """T11: export_saved_model.py CALLS tf.compat.v1.global_variables(), never trainable_variables().

    Root-cause fix for the historical FailedPreconditionError on
    resnet_v1_50/.../BatchNorm/moving_variance (non-trainable, silently
    unrestored by a trainable-only var_list). AST-based for the same reason
    as T10 above — this file's docstrings discuss `trainable_variables()`
    by name while explaining the fix.
    """
    # Arrange
    source = EXPORT_SCRIPT.read_text(encoding="utf-8")

    # Act
    called = _called_dotted_names(source, EXPORT_SCRIPT.name)

    # Assert
    assert any(name.endswith("global_variables") for name in called), called
    assert not any(name.endswith("trainable_variables") for name in called), called


@pytest.mark.spec(ref="ADR-export-pipeline.md#preprocessing-contract")
def test_export_preprocessing_contract_matches_spec() -> None:
    """T12: the exact confirmed preprocessing contract is present in source.

    Signature input key `bytes_inputs`, signature name `serving_default`,
    net input size 288x144, JPEG decode via decode_jpeg. No normalization
    (no `/255` style scaling) — triplet-reid/test.ipynb has that explicitly
    commented out.
    """
    # Arrange
    source = EXPORT_SCRIPT.read_text(encoding="utf-8")

    # Assert — required tokens present
    for expected in ("288", "144", "decode_jpeg", "bytes_inputs", "serving_default"):
        assert expected in source, f"missing expected preprocessing token: {expected!r}"

    # Assert — no normalization introduced
    for forbidden in ("/255", "/ 255", "255.0", "/127.5", "/ 127.5"):
        assert forbidden not in source, f"unexpected normalization token found: {forbidden!r}"


@pytest.mark.skipif(
    not TRIPLET_REID_DIR.exists(),
    reason=f"{TRIPLET_REID_DIR} is gitignored scratch data, not present on this machine",
)
@pytest.mark.spec(ref="ADR-export-pipeline.md#r4-quarantine-superseded-scripts")
@pytest.mark.parametrize(
    "script_name",
    ["freeze_as_saved_model.py", "migrate_checkpoint.py", "format_saved_model.py"],
)
def test_superseded_scripts_are_quarantined(script_name: str) -> None:
    """T14: the 3 collapsed scripts are absent from the active triplet-reid/ tree."""
    # Arrange
    active_path = TRIPLET_REID_DIR / script_name
    archived_path = TRIPLET_REID_DIR / "_archive" / script_name

    # Assert
    assert not active_path.exists(), f"{script_name} still present in active triplet-reid/ tree"
    assert archived_path.exists(), f"{script_name} missing from triplet-reid/_archive/"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# R7 must-not-break invariant. Recorded 2026-07-19 via `sha256sum` on
# wibex_model_v03/{saved_model.pb,fingerprint.pb,variables/*} — this is the
# production numeric baseline and must never change as a side effect of this
# export-pipeline rebuild. Re-run this test after any host-side work too.
_EXPECTED_SHA256 = {
    "fingerprint.pb": "727da118afe5da7c540dd8d802894544925bd76eb6f1227be8dae88f9f03c6bf",
    "saved_model.pb": "00e7f5d2ec83cb611d0700089cc70fd9ef6664b2d8abf6105d2ec09d4a25ebe0",
    "variables/variables.data-00000-of-00001": (
        "f74f84bfaae01460ebd38fce5bc0b39a27d4e8e48f2e35b10cb3bf317156433b"
    ),
    "variables/variables.index": "39f7447b02f49c71c2ff376722ab8ca88ab1ee14ad531739fbeea677a319001c",
}


@pytest.mark.skipif(
    not WIBEX_MODEL_V03_DIR.exists(),
    reason=f"{WIBEX_MODEL_V03_DIR} is gitignored scratch data, not present on this machine",
)
@pytest.mark.spec(ref="ADR-export-pipeline.md#r7-must-not-break-baseline")
@pytest.mark.parametrize("relative_path", sorted(_EXPECTED_SHA256))
def test_wibex_model_v03_baseline_checksum_unchanged(relative_path: str) -> None:
    """T22: wibex_model_v03 (production numeric baseline) files are byte-unchanged."""
    # Arrange
    file_path = WIBEX_MODEL_V03_DIR / relative_path
    assert file_path.exists(), f"baseline file missing: {file_path}"

    # Act
    actual_sha256 = _sha256(file_path)

    # Assert
    expected_sha256 = _EXPECTED_SHA256[relative_path]
    assert actual_sha256 == expected_sha256, (
        f"{relative_path} checksum drifted — expected {expected_sha256}, got {actual_sha256}"
    )
