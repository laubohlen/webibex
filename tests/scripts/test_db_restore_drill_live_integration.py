"""P1/integration tier for scripts/db_restore_drill.py -- NOT implemented.

These need Docker + real `pg_dump`/`pg_restore`/`openssl` binaries + a real
Railway network round-trip, none of which are available in this sandbox
(confirmed: no Docker, no postgresql-client on PATH, no network egress).
Left as skip-gated stubs (mirroring the `live_b2` marker's precedent) so
the user has a concrete, runnable target once unblocked -- not filled in
speculatively per the code-analyst spec ("Do NOT write the P1/integration
tests").

Run manually once postgresql-client + Docker + Railway credentials are
available: `pytest tests/scripts/test_db_restore_drill_live_integration.py
-m live_pg_restore -v`
"""

from __future__ import annotations

import shutil

import pytest

pytestmark = [
    pytest.mark.live_pg_restore,
    pytest.mark.skipif(
        shutil.which("pg_restore") is None,
        reason="pg_restore not on PATH -- install postgresql-client to run this tier",
    ),
]


def test_preflight_source_against_real_read_only_postgres():
    """T30-I: real Postgres connection actually enforces
    default_transaction_read_only=on (attempt a real INSERT/UPDATE against
    the read-only session and confirm it's rejected by the server, not
    just requested by the client)."""
    pytest.skip(
        "T30-I not implemented -- needs a real reachable Postgres instance. "
        "See scripts/db_restore_drill.py:preflight_source / _connect_readonly."
    )


def test_dump_encrypted_real_openssl_round_trip():
    """T40-I: dump_encrypted() against a real local Postgres, followed by a
    real `openssl enc -d` decrypt, confirms the artifact is a genuinely
    valid encrypted pg_dump (not just that the subprocess pipeline exited
    0 with mocked processes)."""
    pytest.skip(
        "T40-I not implemented -- needs real pg_dump + openssl binaries and "
        "a real local Postgres to dump from. See scripts/db_restore_drill.py:"
        "dump_encrypted."
    )


def test_full_restore_drill_end_to_end_against_real_railway():
    """T45-I: the actual GATE evidence run -- fetch_database_url() against
    real Railway credentials, preflight_source() against the real
    production DB, dump_encrypted() + restore_and_verify() into a real
    testcontainers Postgres, full PASS/FAIL report. This is the run that
    satisfies docs/security-remediation-plan.md's GATE section -- it must
    be executed manually once Railway project/environment IDs and a real
    RAILWAY_API_TOKEN are available; it cannot run in CI/sandbox."""
    pytest.skip(
        "T45-I not implemented -- needs Docker (testcontainers), a real "
        "RAILWAY_API_TOKEN, and real --project-id/--environment-id values. "
        "This is the actual GATE evidence run -- see "
        "docs/security-remediation-plan.md GATE section."
    )
