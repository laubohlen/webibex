"""Shared fixtures/config for the sandbox-executable export-pipeline test suite.

These tests are pure Python (AST/grep/hash-based) — none of them import
TensorFlow. See ../ADR-export-pipeline.md and the session report for the
host-only tests (T01/T02/T03/T05/T08/T13/T15-T19/T23/T24) that DO require
real TF/Docker and are covered by the host runbook instead.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# tests/ -> training/triplet-reid/ (REPO_ROOT). tmp/inference/ (where
# triplet-reid/_archive/ and wibex_model_v03/ live) is gitignored scratch,
# a sibling of training/, not of this directory -- see the matching
# constants in test_export_pipeline.py.
REPO_ROOT = Path(__file__).resolve().parent.parent
INFERENCE_ROOT = REPO_ROOT.parents[1] / "tmp" / "inference"


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "spec(ref): links test to a section of ADR-export-pipeline.md"
    )


@pytest.fixture(scope="session")
def inference_root() -> Path:
    return INFERENCE_ROOT
