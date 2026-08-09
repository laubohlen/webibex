#!/usr/bin/env python3
"""Manual DB restore-drill tool -- proves a Postgres backup mechanism
actually *restores*, not just runs and uploads.

Why this exists: `docs/security-remediation-plan.md`'s GATE section
("restore drill required before the id_code max_length migration ships")
blocks any future production migration until a restore has been proven to
work, not merely produced. This script automates the drill's 4-item
checklist end to end:

  1. Produce a backup (via Railway's GraphQL API + `pg_dump`) against the
     real production database.
  2. Actually restore that backup into a separate/scratch local Postgres
     (a throwaway `testcontainers` container -- never anything reachable
     from outside this machine).
  3. Verify the restored DB is usable: row counts match on `Animal`,
     `Region`, `Location`, `IbexImage`, `IbexChip`, `Embedding`, and an
     `Animal` spot-check row matches field-for-field.
  4. Print a PASS/FAIL report; exit 0 only if everything matched.

Scope: this script only dumps + restores + verifies. It does NOT schedule
anything, does NOT upload to B2, and does NOT resume the paused
`backup_db` management command -- see the paused-backup TODO directly
above the GATE section in the same doc. Running this once, successfully,
satisfies the GATE for the current migration; it is not meant to run on
a schedule.

Required environment variables (secrets are NEVER accepted on argv):
    RAILWAY_API_TOKEN      Railway account or project API token, used to
                            fetch the source DATABASE_URL via Railway's
                            GraphQL API. Never printed or logged.
    DB_DUMP_PASSPHRASE     Passphrase used to encrypt/decrypt the pg_dump
                            artifact (openssl -aes-256-cbc -pbkdf2
                            -iter 600000 -salt). Never printed or logged.

Example invocation:
    RAILWAY_API_TOKEN=... DB_DUMP_PASSPHRASE=... \\
        scripts/db_restore_drill.py \\
            --project-id <railway-project-id> \\
            --environment-id <railway-environment-id> \\
            --token-kind account

To manually decrypt the resulting artifact later (e.g. to inspect it, or
to hand it to someone else for a manual restore):
    openssl enc -d -aes-256-cbc -pbkdf2 -iter 600000 \\
        -in webibex_restore_drill.dump.enc -out webibex_restore_drill.dump
"""

from __future__ import annotations

import argparse
import datetime
import ipaddress
import logging
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, unquote, urlsplit

import psycopg2
import requests
from environ import Env
from psycopg2 import sql
from psycopg2.extensions import connection as PGConnection

logger = logging.getLogger(__name__)

# django-environ ships no type stubs; RAILWAY_API_TOKEN is read at CALL
# TIME inside fetch_database_url (not here) so importing this module never
# requires the env var to be set -- tests need to import without it, and
# the fail-secure crash (env(X), no default) should only happen when the
# token is actually needed. Mirrors core/b2_utils.py's fail-secure env()
# pattern, just deferred to call time instead of import time.
env = Env()

EXPECTED_TABLES: tuple[str, ...] = (
    "core_animal",
    "core_region",
    "core_location",
    "core_ibeximage",
    "core_ibexchip",
    "core_embedding",
)

_SPOT_CHECK_BOOL_INDEX = 2  # "marked"
_SPOT_CHECK_DATE_INDEX = 3  # "capture_date"

_DEFAULT_PG_PORT = 5432
_LOOPBACK_HOSTNAME_ALLOWLIST = frozenset({"localhost"})

_TOKEN_HEADER_BY_KIND: dict[str, str] = {
    "account": "Authorization",
    "project": "Project-Access-Token",
}

_RAILWAY_GRAPHQL_QUERY = (
    "query($projectId: String!, $environmentId: String!, $serviceId: String) {\n"
    "  variables(projectId: $projectId, environmentId: $environmentId, "
    "serviceId: $serviceId)\n"
    "}"
)

_PROD_MAJOR_VERSION_RE = re.compile(r"^\d+$")

_REQUIRED_ENV_VARS: tuple[str, ...] = ("RAILWAY_API_TOKEN", "DB_DUMP_PASSPHRASE")


# ---------------------------------------------------------------------------
# DSN handling
# ---------------------------------------------------------------------------
def redact_dsn(dsn: str) -> str:
    """Return `scheme://<redacted>@host:port/path` -- never credentials.

    Raises ValueError (never AttributeError/IndexError) on unparseable
    input. The error message is a static string; it never echoes the raw
    `dsn` argument, so a malformed credential-shaped input can't leak
    through the exception either.

    >>> redact_dsn("postgresql://alice:s3cret@dbhost:5432/webibex")
    'postgresql://<redacted>@dbhost:5432/webibex'
    """
    try:
        parts = urlsplit(dsn)
        port = parts.port
    except ValueError as exc:
        raise ValueError("cannot parse DSN: malformed connection string") from exc

    hostname = parts.hostname
    if not parts.scheme or not hostname:
        raise ValueError("cannot parse DSN: missing scheme or host")

    host_display = f"[{hostname}]" if ":" in hostname else hostname
    netloc = host_display if port is None else f"{host_display}:{port}"
    userinfo = "<redacted>@" if (parts.username or parts.password) else ""
    return f"{parts.scheme}://{userinfo}{netloc}{parts.path}"


def _extract_query_param(query: str, name: str) -> str | None:
    for key, value in parse_qsl(query):
        if key == name:
            return value
    return None


def libpq_env(dsn: str) -> dict[str, str]:
    """Translate a Postgres DSN into libpq `PG*` environment variables.

    Missing DSN components are OMITTED from the returned dict (never an
    empty-string placeholder), so pg_dump/psycopg2 fall back to their own
    defaults for anything not explicitly present in the DSN. Returns a
    fresh dict on every call -- no shared mutable state between calls.

    `PGSSLMODE` defaults to `"require"` when the DSN has no `sslmode`
    query parameter. An explicit `sslmode=disable` (any case) is rejected
    outright -- this tool refuses to connect to a database over an
    unencrypted channel, even if asked to.
    """
    try:
        parts = urlsplit(dsn)
        port = parts.port
    except ValueError as exc:
        raise ValueError("cannot parse DSN: malformed connection string") from exc

    if not parts.scheme or not parts.hostname:
        raise ValueError("cannot parse DSN: missing scheme or host")

    result: dict[str, str] = {"PGHOST": parts.hostname}
    if port is not None:
        result["PGPORT"] = str(port)
    if parts.username:
        result["PGUSER"] = unquote(parts.username)
    if parts.password:
        result["PGPASSWORD"] = unquote(parts.password)
    dbname = parts.path.lstrip("/")
    if dbname:
        result["PGDATABASE"] = dbname

    sslmode = _extract_query_param(parts.query, "sslmode")
    if sslmode is None:
        result["PGSSLMODE"] = "require"
    elif sslmode.lower() == "disable":
        raise ValueError(
            "sslmode=disable is rejected -- refusing to connect without TLS"
        )
    else:
        result["PGSSLMODE"] = sslmode
    return result


@dataclass(frozen=True)
class _NormalizedTarget:
    hostname: str
    port: int
    dbname: str


def _parse_target_dsn(dsn: str) -> _NormalizedTarget:
    try:
        parts = urlsplit(dsn)
        port = parts.port
    except ValueError as exc:
        raise ValueError("cannot parse DSN: malformed connection string") from exc
    if not parts.hostname:
        raise ValueError("cannot parse DSN: missing host")
    return _NormalizedTarget(
        hostname=parts.hostname,
        port=port if port is not None else _DEFAULT_PG_PORT,
        dbname=parts.path.lstrip("/"),
    )


def _is_loopback_host(hostname: str) -> bool:
    if hostname in _LOOPBACK_HOSTNAME_ALLOWLIST:
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        # Not a valid IP literal at all (e.g. "127.0.0.1.evil.com",
        # "2130706433", "0x7f000001") -- ipaddress.ip_address() only
        # accepts strict decimal-dotted-quad / colon-hex forms, so these
        # bypass-shaped hostnames are correctly rejected here, not
        # accidentally treated as loopback.
        return False


def _assert_local_target(dsn: str, source_dsn: str | None = None) -> None:
    """Refuse to restore into anything but a loopback-only host, and
    (when `source_dsn` is known) refuse to restore into the same
    database as the source.

    `source_dsn` is optional: some callers (e.g. `restore_and_verify`
    with no source context available) can only enforce the loopback
    guard, which is the primary anti-prod-write protection. Whenever a
    caller does have both DSNs (the real `main()` orchestration path),
    the self-target guard is also enforced.
    """
    target = _parse_target_dsn(dsn)
    if not _is_loopback_host(target.hostname):
        raise ValueError(
            f"restore target must be a loopback host, got: {redact_dsn(dsn)}"
        )

    if source_dsn is not None:
        source = _parse_target_dsn(source_dsn)
        if (target.hostname, target.port, target.dbname) == (
            source.hostname,
            source.port,
            source.dbname,
        ):
            raise ValueError(
                "restore target must not equal the source database -- "
                "refusing to overwrite the source"
            )


# ---------------------------------------------------------------------------
# Railway GraphQL
# ---------------------------------------------------------------------------
def fetch_database_url(
    project_id: str,
    environment_id: str,
    service_id: str | None,
    *,
    token_kind: str,
    variable_name: str,
    endpoint: str = "backboard.railway.com",
) -> str:
    """Fetch a single Railway service-variable value (typically
    `DATABASE_URL`) via Railway's GraphQL API.

    Uses the module-level `requests.post` (not a `Session`, not
    `requests.request`) so this repo's `no_network` pytest fixture can
    intercept it.
    """
    if token_kind not in _TOKEN_HEADER_BY_KIND:
        raise ValueError(
            f"unsupported token_kind: {token_kind!r} -- expected 'account' or 'project'"
        )

    token = env("RAILWAY_API_TOKEN")
    header_name = _TOKEN_HEADER_BY_KIND[token_kind]
    header_value = (
        f"Bearer {token}" if token_kind == "account" else token  # noqa: S105 -- "account" is a token_kind mode literal, not a credential
    )
    headers = {header_name: header_value, "Content-Type": "application/json"}

    payload = {
        "query": _RAILWAY_GRAPHQL_QUERY,
        "variables": {
            "projectId": project_id,
            "environmentId": environment_id,
            "serviceId": service_id,
        },
    }

    url = f"https://{endpoint}/graphql/v2"
    response = requests.post(url, json=payload, headers=headers, timeout=15)

    if not (200 <= response.status_code < 300):
        raise RuntimeError(
            f"Railway GraphQL request failed: HTTP {response.status_code}"
        )

    try:
        body: dict[str, Any] = response.json()
    except ValueError as exc:
        raise RuntimeError("Railway GraphQL response was not valid JSON") from exc

    errors = body.get("errors")
    if errors:
        messages = [
            e.get("message", "unknown error") for e in errors if isinstance(e, dict)
        ]
        summary = "; ".join(messages) or "unknown error"
        logger.error("Railway GraphQL request returned errors: %s", summary)
        raise RuntimeError(f"Railway GraphQL returned errors: {summary}")

    data = body.get("data")
    if not isinstance(data, dict):
        raise RuntimeError("Railway GraphQL response missing 'data'")
    variables = data.get("variables")
    if not isinstance(variables, dict):
        raise RuntimeError("Railway GraphQL response 'data.variables' is not a mapping")

    if variable_name not in variables:
        available = sorted(variables.keys())
        raise RuntimeError(
            f"variable {variable_name!r} not found; available variables: {available}"
        )

    value = variables[variable_name]
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"variable {variable_name!r} has no usable string value")
    return value


# ---------------------------------------------------------------------------
# Source preflight
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ServerInfo:
    server_major_version: int
    tables_present: tuple[str, ...]


def _pg_dump_major_version(pg_dump_path: str) -> int:
    result = subprocess.run(  # noqa: S603 -- pg_dump_path resolved via shutil.which, fixed argv, shell=False
        [pg_dump_path, "--version"],
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )
    match = re.search(r"(\d+)(?:\.\d+)*", result.stdout)
    if not match:
        raise RuntimeError(
            f"could not parse pg_dump --version output: {result.stdout!r}"
        )
    return int(match.group(1))


def _connect_readonly(dsn: str, timeout: int = 10) -> PGConnection:
    """Connect read-only, with the same TLS enforcement as the pg_dump/
    pg_restore subprocess paths.

    `psycopg2.connect(dsn, ...)` alone is NOT sufficient here: libpq's own
    default `sslmode` (when the DSN doesn't specify one) is `"prefer"`,
    which silently downgrades to plaintext if TLS negotiation fails --
    contradicting this tool's documented "refuses to connect... without
    TLS" guarantee (see `libpq_env()`'s docstring). Reusing `libpq_env()`
    for the sslmode value (not for a full DSN rebuild) gets the same
    require-by-default / reject-disable normalization for this in-process
    connection that subprocess env vars already get.
    """
    sslmode = libpq_env(dsn).get("PGSSLMODE", "require")
    try:
        return psycopg2.connect(
            dsn,
            sslmode=sslmode,
            options="-c default_transaction_read_only=on",
            connect_timeout=timeout,
        )
    except psycopg2.Error as exc:
        raise RuntimeError(
            f"cannot connect to source database: {redact_dsn(dsn)}"
        ) from exc


def preflight_source(dsn: str) -> ServerInfo:
    """Validate the source is reachable read-only, has all
    `EXPECTED_TABLES`, and the local `pg_dump` can actually dump this
    server's major version.
    """
    pg_dump_path = shutil.which("pg_dump")
    if pg_dump_path is None:
        raise RuntimeError(
            "pg_dump not found on PATH -- install postgresql-client "
            "(e.g. `apt-get install postgresql-client` / `brew install libpq`)"
        )
    local_major = _pg_dump_major_version(pg_dump_path)

    conn = _connect_readonly(dsn)
    try:
        with conn.cursor() as cur:
            cur.execute("SHOW server_version_num")
            version_row = cur.fetchone()
            if version_row is None:
                raise RuntimeError("SHOW server_version_num returned no row")
            (version_num,) = version_row
            server_major = int(version_num) // 10000

            cur.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'public'"
            )
            present = {row[0] for row in cur.fetchall()}
    except psycopg2.Error as exc:
        raise RuntimeError(
            f"error querying source database: {redact_dsn(dsn)}"
        ) from exc
    finally:
        conn.close()

    missing = [t for t in EXPECTED_TABLES if t not in present]
    if missing:
        raise RuntimeError(f"source database is missing expected tables: {missing}")

    if local_major < server_major:
        raise RuntimeError(
            f"local pg_dump major version ({local_major}) is older than the "
            f"source server ({server_major}) -- upgrade postgresql-client"
        )

    return ServerInfo(
        server_major_version=server_major, tables_present=tuple(sorted(present))
    )


# ---------------------------------------------------------------------------
# Expected-state collection
# ---------------------------------------------------------------------------
_SPOT_CHECK_QUERY_DEFAULT = (
    "SELECT id_code, name, marked, capture_date, cohort, sex "
    "FROM core_animal WHERE id_code IS NOT NULL ORDER BY id LIMIT 1"
)
_SPOT_CHECK_QUERY_BY_ID_CODE = (
    "SELECT id_code, name, marked, capture_date, cohort, sex "
    "FROM core_animal WHERE id_code = %s LIMIT 1"
)


@dataclass(frozen=True)
class ExpectedState:
    counts: dict[str, int] = field(default_factory=dict)
    spot_row: tuple[Any, ...] = ()


def collect_expected(
    conn: PGConnection, spot_id_code: str | None = None
) -> ExpectedState:
    """Collect per-table row counts (in `EXPECTED_TABLES` order) plus a
    single `Animal` spot-check row, from an already-open connection.

    Table names are never string-built from external input -- they only
    ever come from the fixed `EXPECTED_TABLES` constant, interpolated via
    `psycopg2.sql.Identifier`. `spot_id_code`, if given, is always a bound
    parameter, never interpolated into the query text.
    """
    counts: dict[str, int] = {}
    with conn.cursor() as cur:
        for table in EXPECTED_TABLES:
            cur.execute(
                sql.SQL("SELECT COUNT(*) FROM {}").format(sql.Identifier(table))
            )
            count_row = cur.fetchone()
            if count_row is None:
                raise RuntimeError(f"COUNT(*) query for {table!r} returned no row")
            (count,) = count_row
            counts[table] = count

        if spot_id_code is None:
            cur.execute(_SPOT_CHECK_QUERY_DEFAULT)
        else:
            cur.execute(_SPOT_CHECK_QUERY_BY_ID_CODE, (spot_id_code,))
        row = cur.fetchone()

    if row is None:
        domain = (
            "default (id_code IS NOT NULL)"
            if spot_id_code is None
            else f"id_code={spot_id_code!r}"
        )
        raise RuntimeError(f"no eligible Animal row found for spot-check ({domain})")

    return ExpectedState(counts=counts, spot_row=tuple(row))


# ---------------------------------------------------------------------------
# Encrypted dump
# ---------------------------------------------------------------------------
def dump_encrypted(
    dsn: str,
    out_path: Path,
    passphrase_env_var: str = "DB_DUMP_PASSPHRASE",  # noqa: S107 -- this is an env *var name*, not a credential value
) -> None:
    """Stream `pg_dump -Fc` straight into `openssl enc` -- the plaintext
    dump never touches disk, only the encrypted artifact does.

    Credentials travel via child-process env only, never argv: `pg_dump`
    gets `PG*` vars, `openssl` gets the passphrase in a *separate* env
    dict with no `PG*` vars in it. The artifact is created with mode
    0o600 via `os.open(..., O_CREAT | O_EXCL | O_WRONLY, 0o600)` (not
    write-then-chmod -- avoids the TOCTOU window) and refuses to clobber
    an existing path (`O_EXCL` raises `FileExistsError`, left to
    propagate as an actionable error). On any pipeline failure the
    artifact is unlinked.
    """
    passphrase = os.environ.get(passphrase_env_var, "")
    if not passphrase.strip():
        raise RuntimeError(
            f"{passphrase_env_var} is not set -- refusing to dump without "
            "an encryption passphrase"
        )

    pg_dump_path = shutil.which("pg_dump")
    openssl_path = shutil.which("openssl")
    if pg_dump_path is None:
        raise RuntimeError("pg_dump not found on PATH -- install postgresql-client")
    if openssl_path is None:
        raise RuntimeError("openssl not found on PATH")

    # Atomic create-with-mode-0600, refuses to clobber an existing artifact
    # (O_EXCL). openssl below opens this same, already-existing, already-
    # 0600 path via -out; opening an existing file with O_CREAT does not
    # change its mode, so no write-then-chmod TOCTOU window ever opens.
    fd = os.open(str(out_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(fd)

    pg_env: dict[str, str] = dict(libpq_env(dsn))
    ssl_env: dict[str, str] = {passphrase_env_var: passphrase}

    p1: subprocess.Popen[bytes] | None = None
    p2: subprocess.Popen[bytes] | None = None
    try:
        p1 = subprocess.Popen(  # noqa: S603 -- pg_dump_path from shutil.which (trusted binary), fixed argv, shell=False
            [pg_dump_path, "-Fc", "-w", "--no-owner", "--no-privileges"],
            stdout=subprocess.PIPE,
            env=pg_env,
        )
        p2 = subprocess.Popen(  # noqa: S603 -- openssl_path from shutil.which (trusted binary), fixed argv, shell=False
            [
                openssl_path,
                "enc",
                "-aes-256-cbc",
                "-pbkdf2",
                "-iter",
                "600000",
                "-salt",
                "-pass",
                f"env:{passphrase_env_var}",
                "-out",
                str(out_path),
            ],
            stdin=p1.stdout,
            env=ssl_env,
        )
        if p1.stdout is not None:
            p1.stdout.close()  # let p2 see SIGPIPE if it exits first

        rc1 = p1.wait(timeout=1800)
        rc2 = p2.wait(timeout=1800)

        if rc1 != 0 or rc2 != 0:
            raise RuntimeError(
                f"dump pipeline failed: pg_dump rc={rc1}, openssl rc={rc2}"
            )
    except BaseException:
        out_path.unlink(missing_ok=True)
        raise
    finally:
        for proc in (p1, p2):
            if proc is not None and proc.poll() is None:
                proc.kill()


# ---------------------------------------------------------------------------
# Restore + verify
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class VerifyResult:
    passed: bool
    count_mismatches: dict[str, tuple[int, int]]
    spot_check_ok: bool
    spot_check_expected: tuple[Any, ...]
    spot_check_actual: tuple[Any, ...] | None


def _normalize_bool(value: bool | str | int | None) -> bool | str | int | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("t", "true", "1")
    if isinstance(value, int):
        return bool(value)
    return value


def _normalize_date(
    value: str | datetime.date | datetime.datetime | None,
) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime.datetime):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, datetime.date):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, str):
        return value[:10]
    return value


def _normalize_spot_row(row: tuple[Any, ...]) -> tuple[Any, ...]:
    normalized = list(row)
    if len(normalized) > _SPOT_CHECK_BOOL_INDEX:
        normalized[_SPOT_CHECK_BOOL_INDEX] = _normalize_bool(
            normalized[_SPOT_CHECK_BOOL_INDEX]
        )
    if len(normalized) > _SPOT_CHECK_DATE_INDEX:
        normalized[_SPOT_CHECK_DATE_INDEX] = _normalize_date(
            normalized[_SPOT_CHECK_DATE_INDEX]
        )
    return tuple(normalized)


def _compare_results(
    expected: ExpectedState,
    actual_counts: dict[str, int],
    actual_spot_row: tuple[Any, ...] | None,
) -> VerifyResult:
    mismatches: dict[str, tuple[int, int]] = {}
    for table, expected_count in expected.counts.items():
        actual_count = actual_counts.get(table)
        if actual_count != expected_count:
            mismatches[table] = (
                expected_count,
                actual_count if actual_count is not None else -1,
            )

    spot_ok = False
    if actual_spot_row is not None:
        spot_ok = _normalize_spot_row(expected.spot_row) == _normalize_spot_row(
            actual_spot_row
        )

    return VerifyResult(
        passed=not mismatches and spot_ok,
        count_mismatches=mismatches,
        spot_check_ok=spot_ok,
        spot_check_expected=expected.spot_row,
        spot_check_actual=actual_spot_row,
    )


def _normalize_container_dsn(dsn: str) -> str:
    """testcontainers' `PostgresContainer.get_connection_url()` returns a
    SQLAlchemy-style URL (e.g. `postgresql+psycopg2://...`) -- strip the
    `+driver` suffix so `urlsplit`/psycopg2 see a plain libpq scheme.
    """
    scheme, sep, rest = dsn.partition("://")
    if sep and "+" in scheme:
        scheme = scheme.split("+", 1)[0]
    return f"{scheme}{sep}{rest}" if sep else dsn


def restore_and_verify(
    enc_path: Path,
    expected: ExpectedState,
    prod_major_version: int | str,
    source_dsn: str | None = None,
) -> VerifyResult:
    """Spin up a scratch local Postgres container, decrypt+restore the
    dump into it, and compare row counts + the `Animal` spot-check row
    against `expected`.

    Requires `testcontainers[postgres]` (dev-only, NOT in
    `requirements.txt`) -- imported lazily so importing this module, and
    running the P0 test suite, never requires it to be installed.
    """
    try:
        from testcontainers.postgres import PostgresContainer
    except ImportError as exc:
        raise RuntimeError(
            "testcontainers is not installed -- install it with "
            "`pip install testcontainers[postgres]` (already listed in "
            "requirements-dev.txt) to run the real restore drill"
        ) from exc

    if not _PROD_MAJOR_VERSION_RE.match(str(prod_major_version)):
        raise ValueError(f"invalid prod_major_version: {prod_major_version!r}")

    passphrase_env_var = "DB_DUMP_PASSPHRASE"  # noqa: S105 -- an env *var name*, not a credential value
    passphrase = os.environ.get(passphrase_env_var, "")
    if not passphrase.strip():
        raise RuntimeError(
            f"{passphrase_env_var} is not set -- cannot decrypt the dump artifact"
        )

    openssl_path = shutil.which("openssl")
    pg_restore_path = shutil.which("pg_restore")
    if openssl_path is None:
        raise RuntimeError("openssl not found on PATH")
    if pg_restore_path is None:
        raise RuntimeError("pg_restore not found on PATH -- install postgresql-client")

    image = f"postgres:{prod_major_version}-alpine"
    with PostgresContainer(image) as container:
        target_dsn = _normalize_container_dsn(container.get_connection_url())
        _assert_local_target(target_dsn, source_dsn)

        target_env = dict(libpq_env(target_dsn))
        ssl_env = {passphrase_env_var: passphrase}

        p1: subprocess.Popen[bytes] | None = None
        p2: subprocess.Popen[bytes] | None = None
        try:
            p1 = subprocess.Popen(  # noqa: S603 -- openssl_path from shutil.which, fixed argv, shell=False
                [
                    openssl_path,
                    "enc",
                    "-d",
                    "-aes-256-cbc",
                    "-pbkdf2",
                    "-iter",
                    "600000",
                    "-pass",
                    f"env:{passphrase_env_var}",
                    "-in",
                    str(enc_path),
                ],
                stdout=subprocess.PIPE,
                env=ssl_env,
            )
            p2 = subprocess.Popen(  # noqa: S603 -- pg_restore_path from shutil.which, fixed argv, shell=False
                [
                    pg_restore_path,
                    "--no-owner",
                    "--no-privileges",
                    "--exit-on-error",
                    "--single-transaction",
                    "-d",
                    target_env.get("PGDATABASE", "postgres"),
                ],
                stdin=p1.stdout,
                env=target_env,
            )
            if p1.stdout is not None:
                p1.stdout.close()

            rc1 = p1.wait(timeout=1800)
            rc2 = p2.wait(timeout=1800)
            if rc1 != 0 or rc2 != 0:
                raise RuntimeError(
                    f"restore pipeline failed: openssl rc={rc1}, pg_restore rc={rc2}"
                )
        finally:
            for proc in (p1, p2):
                if proc is not None and proc.poll() is None:
                    proc.kill()

        conn = psycopg2.connect(target_dsn, connect_timeout=10)
        try:
            actual = collect_expected(conn, spot_id_code=expected.spot_row[0])
        finally:
            conn.close()

    return _compare_results(expected, actual.counts, actual.spot_row)


# ---------------------------------------------------------------------------
# CLI orchestration
# ---------------------------------------------------------------------------
def _check_required_env_vars(names: tuple[str, ...]) -> list[str]:
    return [name for name in names if not os.environ.get(name, "").strip()]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n", 1)[0])
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--environment-id", required=True)
    parser.add_argument("--service-id", default=None)
    parser.add_argument("--token-kind", choices=("account", "project"), required=True)
    parser.add_argument("--variable-name", default="DATABASE_URL")
    parser.add_argument("--endpoint", default="backboard.railway.com")
    parser.add_argument("--spot-id-code", default=None)
    parser.add_argument(
        "--out-path", type=Path, default=Path("webibex_restore_drill.dump.enc")
    )
    return parser.parse_args(argv)


def _emit(message: str, *, err: bool = False) -> None:
    """CLI entry-point output meant for the user (PASS/FAIL report, refusal
    messages) -- the documented print() exception in python.md, mirroring
    scripts/run_local_e2e_server.py's own print-based user-facing output.
    Suppression scoped to this single helper rather than one noqa per call.
    """
    print(message, file=sys.stderr if err else sys.stdout, flush=True)


def _print_report(expected: ExpectedState, result: VerifyResult) -> None:
    _emit("=== restore drill report ===")
    for table, expected_count in expected.counts.items():
        if table in result.count_mismatches:
            exp, act = result.count_mismatches[table]
            _emit(f"FAIL {table}: expected={exp} actual={act}")
        else:
            _emit(f"PASS {table}: count={expected_count}")
    if result.spot_check_ok:
        _emit("PASS spot-check: Animal row matches")
    else:
        _emit(
            f"FAIL spot-check: expected={result.spot_check_expected!r} "
            f"actual={result.spot_check_actual!r}"
        )
    _emit(f"=== overall: {'PASS' if result.passed else 'FAIL'} ===")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    missing = _check_required_env_vars(_REQUIRED_ENV_VARS)
    if missing:
        _emit(
            "refusing to run: required env var(s) not set or empty: "
            f"{', '.join(missing)}",
            err=True,
        )
        return 1

    try:
        source_dsn = fetch_database_url(
            args.project_id,
            args.environment_id,
            args.service_id,
            token_kind=args.token_kind,
            variable_name=args.variable_name,
            endpoint=args.endpoint,
        )
        info = preflight_source(source_dsn)

        conn = _connect_readonly(source_dsn)
        try:
            expected = collect_expected(conn, spot_id_code=args.spot_id_code)
        finally:
            conn.close()

        dump_encrypted(source_dsn, args.out_path)

        result = restore_and_verify(
            args.out_path,
            expected,
            info.server_major_version,
            source_dsn=source_dsn,
        )
    except Exception as exc:
        _emit(f"restore drill failed: {exc}", err=True)
        args.out_path.unlink(missing_ok=True)
        return 1

    _print_report(expected, result)
    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
