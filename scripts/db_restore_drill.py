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

`pg_dump`/`pg_restore` run inside one-shot `docker run --rm` containers
(`--pg-client-image`, default `dhi.io/postgres:16-alpine-dev`) -- this
tool requires a working Docker installation on PATH. No native
`postgresql-client` install is needed or used.

Scope: this script only dumps + restores + verifies. It does NOT schedule
anything, does NOT upload to B2, and does NOT resume the paused
`backup_db` management command -- see the paused-backup TODO directly
above the GATE section in the same doc. Running this once, successfully,
satisfies the GATE for the current migration; it is not meant to run on
a schedule.

Required environment variables (secrets are NEVER accepted on argv):
    RAILWAY_API_TOKEN      Railway account or project API token, used to
                            fetch the source DATABASE_URL via Railway's
                            GraphQL API. Never printed or logged. Not
                            required when SOURCE_DSN (below) is set.
    DB_DUMP_PASSPHRASE     Passphrase used to encrypt/decrypt the pg_dump
                            artifact (openssl -aes-256-cbc -pbkdf2
                            -iter 600000 -salt). Never printed or logged.

Optional environment variable:
    SOURCE_DSN             Bypasses the Railway GraphQL fetch and dumps
                            straight from this DSN instead -- for local
                            dry-runs against a throwaway/migrated database
                            when Railway isn't reachable. An env var, not
                            a CLI flag, consistent with "secrets are NEVER
                            accepted on argv" above (a DSN can carry a
                            password). --project-id/--environment-id/
                            --token-kind become unnecessary in this mode.
                            Every other guard (preflight_source,
                            collect_expected, the loopback-only restore-
                            target check) still runs normally against
                            whatever DSN is given -- this only bypasses
                            the credential *source*, not any safety
                            check. Never point this at production outside
                            the intended Railway-fetch path.

Example invocation:
    RAILWAY_API_TOKEN=... DB_DUMP_PASSPHRASE=... \\
        scripts/db_restore_drill.py \\
            --project-id <railway-project-id> \\
            --environment-id <railway-environment-id> \\
            --token-kind account

Local dry-run without Railway:
    SOURCE_DSN=postgresql://user:pw@localhost:5432/webibex_dryrun \\
        DB_DUMP_PASSPHRASE=... scripts/db_restore_drill.py

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
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qsl, unquote, urlsplit

import psycopg2
import requests
from environ import Env

if TYPE_CHECKING:
    # testcontainers is a dev-only dependency, imported lazily at call time
    # in restore_and_verify() -- this import is type-checking-only (never
    # executes) so importing this module never requires the package to be
    # installed.
    from testcontainers.postgres import PostgresContainer
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

# pg_dump/pg_restore/psql now run inside `docker run --rm` against this
# image -- not a secret, safe on argv (see --pg-client-image below).
DEFAULT_PG_CLIENT_IMAGE = "dhi.io/postgres:16-alpine-dev"

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

# `\A...\Z` (not `^...$`) so a trailing newline can't sneak past the end
# anchor, and an explicit ASCII `[0-9]` (not `\d`, which is Unicode-aware
# and accepts fullwidth/Arabic-Indic digits) so this can never accept
# anything but a plain decimal integer -- this value is interpolated
# directly into a Docker image tag (`postgres:{prod_major_version}-alpine`).
_PROD_MAJOR_VERSION_RE = re.compile(r"\A[0-9]+\Z")

_REQUIRED_ENV_VARS: tuple[str, ...] = ("DB_DUMP_PASSPHRASE",)  # RAILWAY_API_TOKEN
# is added conditionally in main() -- unnecessary for a SOURCE_DSN dry-run.


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
# Docker execution wrapper -- pg_dump/pg_restore run inside `docker run
# --rm`, never as host binaries. All docker knowledge (argv shape, image/
# network-id validation, env allowlisting, preflight, rc classification)
# lives in this section; callers below never build a `docker` argv by hand.
# ---------------------------------------------------------------------------
_DOCKER_CONTEXT_ENV_ALLOWLIST: tuple[str, ...] = (
    "HOME",
    "PATH",
    "DOCKER_HOST",
    "DOCKER_CONTEXT",
    "DOCKER_CONFIG",
    "DOCKER_CERT_PATH",
    "DOCKER_TLS_VERIFY",
    "XDG_RUNTIME_DIR",
)

# Dropped from the docker child env by case-sensitive EXACT NAME MATCH
# only -- no lowercase-variant filtering. This tool's own secret env vars
# are always uppercase by its own documented convention (see the module
# docstring), so exact match is the correct, non-surprising behaviour: it
# never silently drops an unrelated variable that merely happens to
# lowercase-collide with one of these names.
_DOCKER_SECRET_ENV_VAR_NAMES: frozenset[str] = frozenset(
    {"DB_DUMP_PASSPHRASE", "RAILWAY_API_TOKEN", "SOURCE_DSN"}
)

# `docker run` has no `--` separator before IMAGE, so an image ref
# beginning with `-` becomes a docker FLAG, not an argument -- that's the
# real injection surface here (shell metachars are already covered by
# `shell=False` throughout). Reject leading `-`, any whitespace/control/
# non-ASCII character, and cap total length well under any real reference
# ever needed by this tool.
_IMAGE_REF_RE = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9._:/@-]{0,127}\Z")

# testcontainers' `Container.id` (via `get_wrapped_container().id`) is the
# full 64-char lowercase hex Docker container ID; Docker also accepts any
# unambiguous prefix, so accept the realistic short-id floor of 12 as well.
# ASCII-only, `fullmatch`-equivalent anchors -- same regex-safety class as
# `_PROD_MAJOR_VERSION_RE` above.
_CONTAINER_ID_RE = re.compile(r"\A[0-9a-f]{12,64}\Z")

_ENV_NAME_RE = re.compile(r"\A[A-Za-z_][A-Za-z0-9_]*\Z")

# Bare binary name only (no `/`) -- same regex-safety class as
# `_IMAGE_REF_RE`/`_CONTAINER_ID_RE`/`_PROD_MAJOR_VERSION_RE` (`\A...\Z`
# anchors, ASCII-only classes).
#
# Design rationale: `docker run` stops parsing its own flags at IMAGE, so
# `command` (positional, strictly AFTER the image) can never become a
# docker flag and is deliberately never validated here. `entrypoint`'s
# value, by contrast, sits in the *pre-image flag region* -- the exact
# region `_IMAGE_REF_RE` defends -- so it gets validated too. Rule for
# this module: everything before IMAGE is validated, positionals after
# IMAGE are trusted.
_ENTRYPOINT_RE = re.compile(r"\A[A-Za-z_][A-Za-z0-9_.-]{0,63}\Z")

_DOCKER_LEVEL_RETURNCODES: tuple[int, ...] = (125, 126, 127)

_DOCKER_PREFLIGHT_TIMEOUT = 15  # seconds


def _docker_child_env(pg_env: dict[str, str]) -> dict[str, str]:
    """Build the env dict passed to a `docker run` child process.

    Union of the 8 docker-context vars (read from `os.environ`) and
    `pg_env`, minus the 3 secret env-var NAMES (see
    `_DOCKER_SECRET_ENV_VAR_NAMES`) -- dropped even if somehow present in
    `pg_env`. On a name collision between the allowlist and `pg_env`, the
    allowlist wins for those 8 specific names; `pg_env` wins for
    everything else (in practice, the `PG*` keys, which never appear in
    the allowlist at all). A docker-context var absent from `os.environ`
    is simply omitted from the result -- never an empty-string placeholder.
    Fresh dict on every call -- no shared mutable state between calls.
    """
    result: dict[str, str] = {
        name: value
        for name, value in pg_env.items()
        if name not in _DOCKER_SECRET_ENV_VAR_NAMES
    }
    for name in _DOCKER_CONTEXT_ENV_ALLOWLIST:
        value = os.environ.get(name)
        if value is not None and name not in _DOCKER_SECRET_ENV_VAR_NAMES:
            result[name] = value
    return result


def _docker_run_argv(
    docker_path: str,
    image: str,
    env_names: list[str],
    command: list[str],
    *,
    network: str | None = None,
    interactive: bool = False,
    entrypoint: str | None = None,
    no_network: bool = False,
) -> list[str]:
    """Build a `docker run --rm` argv list. Never emits `-t`/`--tty`, even
    when `interactive=True` (`-it` together is never emitted) -- this tool
    never needs a pseudo-TTY, and allocating one would let docker-CLI
    chatter corrupt a binary stdout/stdin stream.

    `entrypoint`, when given, bypasses the image's own ENTRYPOINT/CMD
    dispatch via `--entrypoint <value>` (emitted in the pre-image flag
    region, after the sorted `-e` block, immediately before IMAGE) --
    `command` then becomes ARGUMENTS ONLY (e.g. `["--version"]`, never
    `["pg_dump", "--version"]`). This is the fix for a real bug: some
    hardened images (`dhi.io/postgres:16-alpine-dev`) don't replicate the
    official `postgres` image's smart entrypoint dispatch of a positional
    command -- see docs/changes/2026-08-09-db-restore-drill.md's
    2026-08-11-bis addendum. `entrypoint=None` (the default) produces
    byte-identical argv to before this fix -- no `--entrypoint` token at
    all.

    `no_network` emits the single token `"--network=none"` (mirrors the
    existing `"--pull=never"` single-token style, not the two-token
    `["--network", "none"]` form). Mutually exclusive with
    `network=<container-id>` -- both set raises `ValueError` before any
    argv is built. Kept as a separate bool rather than a `network="none"`
    sentinel because `"none"` is already validated (and rejected) as an
    invalid container id by `_CONTAINER_ID_RE`.

    Raises `ValueError` if `image`, `network`, `entrypoint`, or any of
    `env_names` fails validation, or if `network` and `no_network` are
    both given -- never builds an argv containing an unvalidated value or
    a self-contradictory network configuration.
    """
    if not _IMAGE_REF_RE.match(image):
        raise ValueError(f"invalid docker image ref: {image!r}")
    if network is not None and not _CONTAINER_ID_RE.match(network):
        raise ValueError(f"invalid docker network/container id: {network!r}")
    if network is not None and no_network:
        raise ValueError("network and no_network are mutually exclusive -- got both")
    for name in env_names:
        if not _ENV_NAME_RE.match(name):
            raise ValueError(f"invalid docker env var name: {name!r}")
    if entrypoint is not None and not _ENTRYPOINT_RE.match(entrypoint):
        raise ValueError(f"invalid docker entrypoint: {entrypoint!r}")

    argv = [
        docker_path,
        "run",
        "--rm",
        "--pull=never",
        "--security-opt",
        "no-new-privileges",
    ]
    if interactive:
        argv.append("-i")
    if no_network:
        argv.append("--network=none")
    if network is not None:
        argv.extend(["--network", f"container:{network}"])
    for name in sorted(env_names):
        argv.extend(["-e", name])
    if entrypoint is not None:
        argv.extend(["--entrypoint", entrypoint])
    argv.append(image)
    argv.extend(command)
    return argv


def _classify_docker_rc(rc: int) -> str:
    """Classify a subprocess return code as a docker-level failure
    (daemon/image/exec problem -- rc in `_DOCKER_LEVEL_RETURNCODES`) or a
    `pg_dump`/`pg_restore` failure (`"pg"`, everything else, including the
    program's own non-zero exit codes).
    """
    return "docker" if rc in _DOCKER_LEVEL_RETURNCODES else "pg"


def _docker_path() -> str:
    path = shutil.which("docker")
    if path is None:
        raise RuntimeError(
            "docker not found on PATH -- install Docker Desktop / Docker "
            "Engine. pg_dump/pg_restore now run inside `docker run --rm` "
            "containers, not as host binaries."
        )
    return path


def _docker_preflight(docker_path: str, image: str) -> None:
    """Confirm the Docker daemon is reachable, then confirm `image` is
    already present locally -- exactly 2 `subprocess.run` calls, in that
    order, never a 3rd (no implicit pull; `docker run` itself always
    passes `--pull=never`).

    Order is load-bearing: a dead daemon and a missing image both surface
    as a non-zero `docker image inspect` return code too, so checking the
    image first would misdiagnose a dead daemon as merely "image missing".
    """
    if not _IMAGE_REF_RE.match(image):
        raise ValueError(f"invalid docker image ref: {image!r}")

    try:
        version_result = subprocess.run(  # noqa: S603 -- docker_path from shutil.which, fixed argv, shell=False
            [docker_path, "version", "--format", "{{.Server.Version}}"],
            capture_output=True,
            text=True,
            timeout=_DOCKER_PREFLIGHT_TIMEOUT,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            "docker version timed out -- is the Docker daemon running?"
        ) from exc
    except (PermissionError, FileNotFoundError) as exc:
        raise RuntimeError(f"cannot execute docker at {docker_path!r}: {exc}") from exc

    if version_result.returncode != 0:
        raise RuntimeError(
            "docker daemon is not reachable (`docker version` failed) -- "
            "is Docker running?"
        )

    try:
        inspect_result = subprocess.run(  # noqa: S603 -- docker_path from shutil.which, image validated by _IMAGE_REF_RE above, shell=False
            [docker_path, "image", "inspect", image],
            capture_output=True,
            text=True,
            timeout=_DOCKER_PREFLIGHT_TIMEOUT,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"docker image inspect {image!r} timed out") from exc
    except (PermissionError, FileNotFoundError) as exc:
        raise RuntimeError(f"cannot execute docker at {docker_path!r}: {exc}") from exc

    if inspect_result.returncode != 0:
        raise RuntimeError(
            f"docker image {image!r} is not present locally -- this tool "
            f"never pulls implicitly. Pull it first: docker pull {image}"
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


def _pg_dump_major_version(docker_path: str, image: str) -> int:
    # `--entrypoint pg_dump` bypasses the image's own entrypoint dispatch
    # (some hardened images, e.g. dhi.io/postgres:16-alpine-dev, don't
    # replicate the official postgres image's smart dispatch of a
    # positional command -- see the module's `_docker_run_argv`
    # docstring). A `--version` probe has zero legitimate network need,
    # so it also always runs with `--network=none`.
    argv = _docker_run_argv(
        docker_path,
        image,
        [],
        ["--version"],
        entrypoint="pg_dump",
        no_network=True,
    )
    result = subprocess.run(  # noqa: S603 -- argv built by _docker_run_argv (image validated), shell=False
        argv,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if result.returncode != 0:
        kind = _classify_docker_rc(result.returncode)
        raise RuntimeError(
            f"pg_dump --version failed inside docker (rc={result.returncode}, "
            f"{kind}-level failure): {result.stderr.strip()}"
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


def preflight_source(dsn: str, *, image: str = DEFAULT_PG_CLIENT_IMAGE) -> ServerInfo:
    """Validate the source is reachable read-only, has all
    `EXPECTED_TABLES`, and the dockerized `pg_dump` can actually dump this
    server's major version.

    Docker preflight (`_docker_preflight`) runs before the first
    `psycopg2.connect` -- fail fast on a docker-level problem (daemon down,
    image missing) before ever touching the database.
    """
    docker_path = _docker_path()
    _docker_preflight(docker_path, image)
    local_major = _pg_dump_major_version(docker_path, image)

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
    *,
    image: str = DEFAULT_PG_CLIENT_IMAGE,
) -> None:
    """Stream a dockerized `pg_dump -Fc` straight into `openssl enc` -- the
    plaintext dump never touches disk, only the encrypted artifact does.

    Credentials travel via child-process env only, never argv: `pg_dump`
    (inside `docker run --rm`) gets an allowlisted docker-context env plus
    `PG*` vars (`_docker_child_env`), `openssl` gets the passphrase in a
    *separate* env dict with no `PG*` vars in it. `--pull=never` and a
    prior image preflight (see `preflight_source`) mean this never pulls
    an image implicitly, and no docker-CLI chatter can reach the binary
    stdout pipe (no `-t`/`--tty` ever emitted). The artifact is created
    with mode 0o600 via `os.open(..., O_CREAT | O_EXCL | O_WRONLY, 0o600)`
    (not write-then-chmod -- avoids the TOCTOU window) and refuses to
    clobber an existing path (`O_EXCL` raises `FileExistsError`, left to
    propagate as an actionable error). On any pipeline failure the
    artifact is unlinked.
    """
    passphrase = os.environ.get(passphrase_env_var, "")
    if not passphrase.strip():
        raise RuntimeError(
            f"{passphrase_env_var} is not set -- refusing to dump without "
            "an encryption passphrase"
        )

    docker_path = _docker_path()
    openssl_path = shutil.which("openssl")
    if openssl_path is None:
        raise RuntimeError("openssl not found on PATH")

    pg_env: dict[str, str] = dict(libpq_env(dsn))
    docker_env = _docker_child_env(pg_env)
    # Build + validate the docker argv (can raise ValueError on a bad
    # `image`) BEFORE the artifact file is created below. If validation
    # happened after os.open(), a ValueError here would leave an orphaned
    # 0600 file behind that the *next* run's O_EXCL would then refuse to
    # clobber.
    docker_argv = _docker_run_argv(
        docker_path,
        image,
        sorted(docker_env),
        ["-Fc", "-w", "--no-owner", "--no-privileges"],
        entrypoint="pg_dump",
    )

    # Atomic create-with-mode-0600, refuses to clobber an existing artifact
    # (O_EXCL). openssl below opens this same, already-existing, already-
    # 0600 path via -out; opening an existing file with O_CREAT does not
    # change its mode, so no write-then-chmod TOCTOU window ever opens.
    fd = os.open(str(out_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(fd)

    ssl_env: dict[str, str] = {passphrase_env_var: passphrase}

    p1: subprocess.Popen[bytes] | None = None
    p2: subprocess.Popen[bytes] | None = None
    try:
        p1 = subprocess.Popen(  # noqa: S603 -- argv built by _docker_run_argv (image/env-names validated), shell=False
            docker_argv,
            stdout=subprocess.PIPE,
            env=docker_env,
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
            kind = _classify_docker_rc(rc1)
            raise RuntimeError(
                f"dump pipeline failed: pg_dump (docker, {kind}-level) "
                f"rc={rc1}, openssl rc={rc2}"
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


def _testcontainers_container_id(container: PostgresContainer) -> str:
    """Read the full 64-char lowercase hex Docker container ID from a
    `testcontainers.postgres.PostgresContainer` instance, via
    `get_wrapped_container().id` (docker-py's `Container.id` property --
    confirmed by live inspection to read `.attrs["Id"]`, the full ID).

    Any failure reading or validating the id (missing accessor, a
    testcontainers-internal start failure, or a malformed id) is
    normalized to an actionable `RuntimeError` -- never lets a raw,
    library-specific exception escape.
    """
    # Broad `except Exception` is intentional here: normalizes any
    # testcontainers/docker-py failure mode (missing accessor, an
    # internal ContainerStartException, etc.) into one actionable message
    # rather than letting a library-specific exception escape.
    try:
        container_id = container.get_wrapped_container().id
    except Exception as exc:
        raise RuntimeError(
            f"could not read the testcontainers container id "
            f"({type(exc).__name__}: {exc})"
        ) from exc

    if not isinstance(container_id, str) or not _CONTAINER_ID_RE.match(container_id):
        raise RuntimeError(
            f"testcontainers container id has an unexpected shape: {container_id!r}"
        )
    return container_id


def _restore_target_env(target_dsn: str) -> dict[str, str]:
    """Rewrite `PGHOST`/`PGPORT` to the container-internal loopback
    address and disable TLS for the *restore* leg only.

    `PGSSLMODE=disable` is legal here ONLY because `_assert_local_target`
    has already proven `target_dsn` is a genuine loopback host before this
    function is ever called (restore_and_verify calls the guard first,
    unconditionally -- see its ordering). The dockerized `pg_restore`
    reaches this Postgres via `--network container:<id>`, sharing the
    testcontainers container's network namespace, so traffic never leaves
    the local machine even with TLS disabled. This is NOT a general TLS
    downgrade: the source-DSN leg (`libpq_env`/`_connect_readonly`) still
    refuses `sslmode=disable` unconditionally, and this function must only
    ever be called strictly AFTER `_assert_local_target` has passed --
    calling it first would let this function manufacture the very value
    the guard is supposed to validate, making the guard vacuous.
    """
    env = dict(libpq_env(target_dsn))
    env["PGHOST"] = "127.0.0.1"
    env["PGPORT"] = str(_DEFAULT_PG_PORT)
    env["PGSSLMODE"] = "disable"
    return env


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
    *,
    image: str = DEFAULT_PG_CLIENT_IMAGE,
) -> VerifyResult:
    """Spin up a scratch local Postgres container, decrypt+restore the
    dump into it, and compare row counts + the `Animal` spot-check row
    against `expected`.

    Requires `testcontainers[postgres]` (dev-only, NOT in
    `requirements.txt`) -- imported lazily so importing this module, and
    running the P0 test suite, never requires it to be installed.

    Ordering (load-bearing, in this exact sequence):
      1. `_assert_local_target` -- the anti-prod-write guard -- runs
         before any `Popen`.
      2. `_testcontainers_container_id` / `_restore_target_env` run
         strictly AFTER (1) has passed. `_restore_target_env` rewrites
         `PGHOST`/`PGPORT`/`PGSSLMODE`; running it before the guard would
         let it manufacture the very value the guard validates.
      3. The post-restore verification `psycopg2.connect` reuses the
         container's *published* DSN (`target_dsn`, from
         `get_connection_url()`), never the rewritten container-internal
         `restore_env` -- reusing `restore_env` here would dial
         `127.0.0.1:5432` on the HOST, which can silently hit an
         unrelated real local Postgres instance on some machines.
    """
    try:
        # testcontainers.postgres is deprecated as of 4.15.0 in favor of
        # testcontainers.community.postgres (still functional here, no
        # behavior change) -- tracked in docs/security-remediation-plan.md's
        # "migrate testcontainers.postgres" TODO, deliberately not fixed in
        # this CR (out of scope, unrelated to the docker-run wiring).
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

    docker_path = _docker_path()
    openssl_path = shutil.which("openssl")
    if openssl_path is None:
        raise RuntimeError("openssl not found on PATH")

    pg_server_image = f"postgres:{prod_major_version}-alpine"
    with PostgresContainer(pg_server_image) as container:
        target_dsn = _normalize_container_dsn(container.get_connection_url())
        _assert_local_target(target_dsn, source_dsn)

        container_id = _testcontainers_container_id(container)
        restore_env = _restore_target_env(target_dsn)
        docker_env = _docker_child_env(restore_env)
        docker_argv = _docker_run_argv(
            docker_path,
            image,
            sorted(docker_env),
            [
                "--no-owner",
                "--no-privileges",
                "--exit-on-error",
                "--single-transaction",
                "-d",
                restore_env.get("PGDATABASE", "postgres"),
            ],
            network=container_id,
            interactive=True,
            entrypoint="pg_restore",
        )

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
            p2 = subprocess.Popen(  # noqa: S603 -- argv built by _docker_run_argv (image/network/env-names validated), shell=False
                docker_argv,
                stdin=p1.stdout,
                env=docker_env,
            )
            if p1.stdout is not None:
                p1.stdout.close()

            rc1 = p1.wait(timeout=1800)
            rc2 = p2.wait(timeout=1800)
            if rc1 != 0 or rc2 != 0:
                kind = _classify_docker_rc(rc2)
                raise RuntimeError(
                    f"restore pipeline failed: openssl rc={rc1}, "
                    f"pg_restore (docker, {kind}-level) rc={rc2}"
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
    parser.add_argument("--project-id")
    parser.add_argument("--environment-id")
    parser.add_argument("--service-id", default=None)
    parser.add_argument("--token-kind", choices=("account", "project"))
    parser.add_argument("--variable-name", default="DATABASE_URL")
    parser.add_argument("--endpoint", default="backboard.railway.com")
    parser.add_argument("--spot-id-code", default=None)
    parser.add_argument(
        "--out-path", type=Path, default=Path("webibex_restore_drill.dump.enc")
    )
    parser.add_argument(
        "--pg-client-image",
        default=DEFAULT_PG_CLIENT_IMAGE,
        help=(
            "docker image used to run pg_dump/pg_restore inside `docker "
            "run --rm` (default: %(default)s). This is NOT a secret -- "
            "an image reference is fine on argv."
        ),
    )
    args = parser.parse_args(argv)
    if not os.environ.get("SOURCE_DSN", "").strip():
        missing = [
            name
            for name, value in (
                ("--project-id", args.project_id),
                ("--environment-id", args.environment_id),
                ("--token-kind", args.token_kind),
            )
            if value is None
        ]
        if missing:
            parser.error(
                "the following arguments are required unless the SOURCE_DSN "
                f"env var is set: {', '.join(missing)}"
            )
    return args


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

    source_dsn_override = os.environ.get("SOURCE_DSN", "").strip() or None
    # RAILWAY_API_TOKEN is only needed when actually calling Railway's API;
    # a SOURCE_DSN dry-run needs no Railway account/token at all.
    required_env_vars = _REQUIRED_ENV_VARS
    if source_dsn_override is None:
        required_env_vars = (*required_env_vars, "RAILWAY_API_TOKEN")

    missing = _check_required_env_vars(required_env_vars)
    if missing:
        _emit(
            "refusing to run: required env var(s) not set or empty: "
            f"{', '.join(missing)}",
            err=True,
        )
        return 1

    try:
        if source_dsn_override is not None:
            source_dsn = source_dsn_override
        else:
            source_dsn = fetch_database_url(
                args.project_id,
                args.environment_id,
                args.service_id,
                token_kind=args.token_kind,
                variable_name=args.variable_name,
                endpoint=args.endpoint,
            )
        info = preflight_source(source_dsn, image=args.pg_client_image)

        conn = _connect_readonly(source_dsn)
        try:
            expected = collect_expected(conn, spot_id_code=args.spot_id_code)
        finally:
            conn.close()

        dump_encrypted(source_dsn, args.out_path, image=args.pg_client_image)

        result = restore_and_verify(
            args.out_path,
            expected,
            info.server_major_version,
            source_dsn=source_dsn,
            image=args.pg_client_image,
        )
    except Exception as exc:
        _emit(f"restore drill failed: {exc}", err=True)
        args.out_path.unlink(missing_ok=True)
        return 1

    _print_report(expected, result)
    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
