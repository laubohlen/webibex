"""P0 tests for scripts/db_restore_drill.py: the docker-run wrapper --
`_docker_run_argv`, `_IMAGE_REF_RE`, `_CONTAINER_ID_RE`, `_classify_docker_rc`,
`_docker_preflight`, `_docker_child_env`.

No real Docker daemon, no real Docker binary invocation -- everything
mocked at the `subprocess.run`/`subprocess.Popen` boundary (`fake_run`,
`fake_popen_cls` from conftest.py).
"""

from __future__ import annotations

import subprocess

import pytest
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st

import scripts.db_restore_drill as mod
from tests.scripts.conftest import FakeCompletedProcess

pytestmark = pytest.mark.spec(
    ref="docs/security-remediation-plan.md#gate-restore-drill-required"
)


# ---------------------------------------------------------------------------
# _docker_run_argv -- exact shape
# ---------------------------------------------------------------------------
def test_docker_run_argv_dump_leg_exact_shape():
    argv = mod._docker_run_argv(
        "/usr/bin/docker",
        "dhi.io/postgres:16-alpine-dev",
        ["PGHOST", "PGPORT", "PGUSER"],
        ["-Fc", "-w", "--no-owner", "--no-privileges"],
        entrypoint="pg_dump",
    )
    assert argv == [
        "/usr/bin/docker",
        "run",
        "--rm",
        "--pull=never",
        "--security-opt",
        "no-new-privileges",
        "-e",
        "PGHOST",
        "-e",
        "PGPORT",
        "-e",
        "PGUSER",
        "--entrypoint",
        "pg_dump",
        "dhi.io/postgres:16-alpine-dev",
        "-Fc",
        "-w",
        "--no-owner",
        "--no-privileges",
    ]


def test_docker_run_argv_restore_leg_exact_shape_with_network_and_interactive():
    container_id = "a" * 12
    argv = mod._docker_run_argv(
        "/usr/bin/docker",
        "dhi.io/postgres:16-alpine-dev",
        ["PGDATABASE", "PGHOST", "PGPORT", "PGSSLMODE"],
        [
            "--no-owner",
            "--no-privileges",
            "--exit-on-error",
            "--single-transaction",
            "-d",
            "postgres",
        ],
        network=container_id,
        interactive=True,
        entrypoint="pg_restore",
    )
    assert argv == [
        "/usr/bin/docker",
        "run",
        "--rm",
        "--pull=never",
        "--security-opt",
        "no-new-privileges",
        "-i",
        "--network",
        f"container:{container_id}",
        "-e",
        "PGDATABASE",
        "-e",
        "PGHOST",
        "-e",
        "PGPORT",
        "-e",
        "PGSSLMODE",
        "--entrypoint",
        "pg_restore",
        "dhi.io/postgres:16-alpine-dev",
        "--no-owner",
        "--no-privileges",
        "--exit-on-error",
        "--single-transaction",
        "-d",
        "postgres",
    ]


def test_docker_run_argv_entrypoint_none_default_matches_pre_fix_shape():
    """Back-compat guard: `entrypoint=None` (the default) produces
    byte-identical argv to before this fix -- no `--entrypoint` token at
    all, `command` still carries its own leading binary name.
    """
    argv = mod._docker_run_argv(
        "/usr/bin/docker",
        "postgres:16-alpine",
        ["PGHOST"],
        ["pg_dump", "--version"],
    )
    assert argv == [
        "/usr/bin/docker",
        "run",
        "--rm",
        "--pull=never",
        "--security-opt",
        "no-new-privileges",
        "-e",
        "PGHOST",
        "postgres:16-alpine",
        "pg_dump",
        "--version",
    ]
    assert "--entrypoint" not in argv


def test_docker_run_argv_env_names_sorted_regardless_of_input_order():
    argv = mod._docker_run_argv(
        "/usr/bin/docker",
        "postgres:16-alpine",
        ["PGUSER", "PGHOST", "PGPORT"],
        ["true"],
    )
    e_indices = [i for i, tok in enumerate(argv) if tok == "-e"]
    names_in_order = [argv[i + 1] for i in e_indices]
    assert names_in_order == ["PGHOST", "PGPORT", "PGUSER"]


@pytest.mark.parametrize("entrypoint", [None, "pg_dump"])
def test_docker_run_argv_each_e_flag_is_name_only_no_equals(entrypoint):
    argv = mod._docker_run_argv(
        "/usr/bin/docker",
        "postgres:16-alpine",
        ["PGHOST", "PGPASSWORD"],
        ["true"],
        entrypoint=entrypoint,
    )
    e_indices = [i for i, tok in enumerate(argv) if tok == "-e"]
    for i in e_indices:
        name = argv[i + 1]
        assert "=" not in name
        assert re_matches_env_name(name)
        # the entrypoint token/value must never land inside the -e loop
        # or be mistaken for a name emitted by it.
        assert name != "--entrypoint"
        if entrypoint is not None:
            assert name != entrypoint


def re_matches_env_name(name: str) -> bool:
    import re

    return bool(re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name))


@pytest.mark.parametrize("interactive", [True, False])
@pytest.mark.parametrize("entrypoint", [None, "pg_dump"])
def test_docker_run_argv_never_emits_tty_flags(interactive, entrypoint):
    argv = mod._docker_run_argv(
        "/usr/bin/docker",
        "postgres:16-alpine",
        ["PGHOST"],
        ["true"],
        interactive=interactive,
        entrypoint=entrypoint,
    )
    assert "-t" not in argv
    assert "--tty" not in argv
    assert "-it" not in argv


def test_docker_run_argv_pull_never_rm_and_no_new_privileges_present():
    argv = mod._docker_run_argv(
        "/usr/bin/docker", "postgres:16-alpine", [], ["true"]
    )
    assert "--rm" in argv
    assert "--pull=never" in argv
    assert "--security-opt" in argv
    assert "no-new-privileges" in argv


def test_docker_run_argv_image_index_precedes_command_index():
    """`--entrypoint pg_dump` sits in the PRE-image flag region;
    `command` (args-only, `["--version"]`) sits strictly AFTER the image.
    A prior version of this test did `argv.index("pg_dump")`, which would
    silently resolve to the entrypoint value's own (pre-image) position
    once the fix landed -- inverting the assertion into a false pass.
    Rewritten to check the entrypoint flag and the trailing command arg
    on opposite sides of the image index instead.
    """
    argv = mod._docker_run_argv(
        "/usr/bin/docker",
        "postgres:16-alpine",
        ["PGHOST"],
        ["--version"],
        entrypoint="pg_dump",
    )
    entrypoint_index = argv.index("--entrypoint")
    image_index = argv.index("postgres:16-alpine")
    command_index = argv.index("--version")
    assert entrypoint_index < image_index < command_index


# ---------------------------------------------------------------------------
# Image-ref validation
# ---------------------------------------------------------------------------
_BAD_IMAGE_REFS = [
    "-v/:/host",
    "--privileged",
    "-v /:/host",
    "; rm -rf /",
    "postgres:16 --privileged",
    "",
    "latest ",
    "\tpostgres:16",
    "postgres:16\n",
    "postgres:16\n--privileged",
    "postgres:16\x00",
    "–privileged",  # noqa: RUF001 -- U+2013 en-dash, not ASCII '-' (intentional confusable)
    "pöstgres:16",  # ö
    "postgres:１６",  # noqa: RUF001 -- fullwidth digits (intentional confusable)
    "postgres:16​",  # zero-width space
    "a" * 250,  # 200+-char ref
]

_GOOD_IMAGE_REFS = [
    "dhi.io/postgres:16-alpine-dev",
    "postgres:16-alpine",
    "registry.example.com:5000/ns/postgres:16",
    f"postgres@sha256:{'0' * 64}",
]


@pytest.mark.parametrize("bad_ref", _BAD_IMAGE_REFS)
def test_image_ref_re_rejects_unsafe_refs(bad_ref):
    assert mod._IMAGE_REF_RE.match(bad_ref) is None


@pytest.mark.parametrize("good_ref", _GOOD_IMAGE_REFS)
def test_image_ref_re_accepts_legitimate_refs(good_ref):
    assert mod._IMAGE_REF_RE.match(good_ref) is not None


@pytest.mark.parametrize("bad_ref", _BAD_IMAGE_REFS)
def test_docker_run_argv_rejects_unsafe_image_ref(bad_ref):
    with pytest.raises(ValueError):
        mod._docker_run_argv("/usr/bin/docker", bad_ref, [], ["true"])


_ASCII_ALNUM_CHARS = st.characters(
    whitelist_categories=("Lu", "Ll", "Nd"), max_codepoint=127
)


@given(
    prefix=st.text(alphabet=_ASCII_ALNUM_CHARS, min_size=0, max_size=15),
    suffix=st.text(alphabet=_ASCII_ALNUM_CHARS, min_size=0, max_size=15),
)
@example(prefix="", suffix="")
@settings(deadline=None, max_examples=100)
def test_image_ref_re_property_leading_dash_always_rejected(prefix, suffix):
    candidate = "-" + prefix + suffix
    assert mod._IMAGE_REF_RE.match(candidate) is None


@given(
    body=st.text(alphabet=_ASCII_ALNUM_CHARS, min_size=1, max_size=15),
    bad_char=st.sampled_from(
        [" ", "\t", "\n", "\r", "\x00", "–", "é", "​"]  # noqa: RUF001 -- intentional confusables (en-dash, ZWSP)
    ),
)
@settings(deadline=None, max_examples=100)
def test_image_ref_re_property_embedded_bad_char_always_rejected(body, bad_char):
    candidate = body[: len(body) // 2] + bad_char + body[len(body) // 2 :]
    assert mod._IMAGE_REF_RE.match(candidate) is None


# ---------------------------------------------------------------------------
# Container-id validation
# ---------------------------------------------------------------------------
_BAD_CONTAINER_IDS = [
    "",
    "host",
    "none",
    "../../etc",
    "a" * 11,
    "a" * 65,
    "A" * 12,
    "deadbeefcafe ",
    "deadbeefcafe\n",
    "deadbeefcaf;",
]

_GOOD_CONTAINER_IDS = ["a" * 12, "0" * 64]


@pytest.mark.parametrize("bad_id", _BAD_CONTAINER_IDS)
def test_container_id_re_rejects_unsafe_ids(bad_id):
    assert mod._CONTAINER_ID_RE.match(bad_id) is None


@pytest.mark.parametrize("good_id", _GOOD_CONTAINER_IDS)
def test_container_id_re_accepts_legitimate_ids(good_id):
    assert mod._CONTAINER_ID_RE.match(good_id) is not None


@pytest.mark.parametrize(
    "length,should_match",
    [(11, False), (12, True), (64, True), (65, False)],
)
def test_container_id_re_boundary_pairs(length, should_match):
    candidate = "a" * length
    matched = mod._CONTAINER_ID_RE.match(candidate) is not None
    assert matched is should_match


@pytest.mark.parametrize("bad_id", _BAD_CONTAINER_IDS)
def test_docker_run_argv_rejects_unsafe_network_id(bad_id):
    with pytest.raises(ValueError):
        mod._docker_run_argv(
            "/usr/bin/docker", "postgres:16-alpine", [], ["true"], network=bad_id
        )


# ---------------------------------------------------------------------------
# env_names validation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "bad_names",
    [
        ["PGHOST=evil"],
        ["-v"],
        [""],
        ["PG HOST"],
        ["PGHOST\n"],
    ],
)
def test_docker_run_argv_rejects_invalid_env_names(bad_names):
    with pytest.raises(ValueError):
        mod._docker_run_argv(
            "/usr/bin/docker", "postgres:16-alpine", bad_names, ["true"]
        )


# ---------------------------------------------------------------------------
# entrypoint validation (_ENTRYPOINT_RE) -- the fix for the live
# entrypoint-dispatch bug: `dhi.io/postgres:16-alpine-dev`'s hardened
# entrypoint doesn't dispatch `pg_dump ...` as a positional command the
# way the official `postgres` image does. `--entrypoint <binary>` +
# args-only `command` is the confirmed fix.
# ---------------------------------------------------------------------------
_BAD_ENTRYPOINTS = [
    "-v",
    "",
    "pg_dump; rm -rf /",
    "/bin/sh",
    "pg dump",
    "pg_dump\n",
    "pg_dump\x00",
    "pg_dümp",
    "a" * 65,
    "pg_dump=evil",
    ".pg_dump",  # leading dot
    "1pg_dump",  # leading digit
]

_GOOD_ENTRYPOINTS = ["pg_dump", "pg_restore", "psql", "_x", "pg-dump.v2", "a" * 64]


@pytest.mark.parametrize("bad_entrypoint", _BAD_ENTRYPOINTS)
def test_entrypoint_re_rejects_unsafe_entrypoints(bad_entrypoint):
    assert mod._ENTRYPOINT_RE.match(bad_entrypoint) is None


@pytest.mark.parametrize("good_entrypoint", _GOOD_ENTRYPOINTS)
def test_entrypoint_re_accepts_legitimate_entrypoints(good_entrypoint):
    assert mod._ENTRYPOINT_RE.match(good_entrypoint) is not None


@pytest.mark.parametrize(
    "length,should_match",
    [(63, True), (64, True), (65, False)],
)
def test_entrypoint_re_boundary_lengths(length, should_match):
    candidate = "a" * length
    matched = mod._ENTRYPOINT_RE.match(candidate) is not None
    assert matched is should_match


@pytest.mark.parametrize("bad_entrypoint", _BAD_ENTRYPOINTS)
def test_docker_run_argv_rejects_unsafe_entrypoint(bad_entrypoint):
    with pytest.raises(ValueError):
        mod._docker_run_argv(
            "/usr/bin/docker",
            "postgres:16-alpine",
            [],
            ["true"],
            entrypoint=bad_entrypoint,
        )


@given(
    prefix=st.text(alphabet=_ASCII_ALNUM_CHARS, min_size=0, max_size=15),
    suffix=st.text(alphabet=_ASCII_ALNUM_CHARS, min_size=0, max_size=15),
)
@example(prefix="", suffix="")
@settings(deadline=None, max_examples=100)
def test_entrypoint_re_property_leading_dash_always_rejected(prefix, suffix):
    candidate = "-" + prefix + suffix
    assert mod._ENTRYPOINT_RE.match(candidate) is None


@given(
    body=st.text(alphabet=_ASCII_ALNUM_CHARS, min_size=1, max_size=15),
    bad_char=st.sampled_from(
        [" ", "\t", "\n", "\r", "\x00", "/", ";", "=", "–", "é"]  # noqa: RUF001 -- intentional confusables (en-dash, non-ASCII)
    ),
)
@settings(deadline=None, max_examples=100)
def test_entrypoint_re_property_embedded_bad_char_always_rejected(body, bad_char):
    candidate = body[: len(body) // 2] + bad_char + body[len(body) // 2 :]
    assert mod._ENTRYPOINT_RE.match(candidate) is None


# ---------------------------------------------------------------------------
# no_network -- "--network=none" single-token form, mutually exclusive
# with network=<container-id>
# ---------------------------------------------------------------------------
def test_docker_run_argv_network_and_no_network_conjunction_raises():
    with pytest.raises(ValueError):
        mod._docker_run_argv(
            "/usr/bin/docker",
            "postgres:16-alpine",
            [],
            ["true"],
            network="a" * 12,
            no_network=True,
        )


def test_docker_run_argv_network_alone_does_not_raise():
    argv = mod._docker_run_argv(
        "/usr/bin/docker",
        "postgres:16-alpine",
        [],
        ["true"],
        network="a" * 12,
    )
    assert "--network" in argv


def test_docker_run_argv_no_network_alone_does_not_raise():
    argv = mod._docker_run_argv(
        "/usr/bin/docker",
        "postgres:16-alpine",
        [],
        ["true"],
        no_network=True,
    )
    assert "--network=none" in argv


def test_docker_run_argv_no_network_emits_single_token_pre_image():
    argv = mod._docker_run_argv(
        "/usr/bin/docker", "postgres:16-alpine", [], ["true"], no_network=True
    )
    assert argv.count("--network=none") == 1
    # never the two-token `["--network", "none"]` form
    assert "--network" not in argv
    image_index = argv.index("postgres:16-alpine")
    assert argv.index("--network=none") < image_index


def test_docker_run_argv_no_network_false_default_omits_both_tokens():
    argv = mod._docker_run_argv("/usr/bin/docker", "postgres:16-alpine", [], ["true"])
    assert "--network" not in argv
    assert "--network=none" not in argv


# ---------------------------------------------------------------------------
# 12-case hand-rolled pairwise sweep (no `allpairspy` -- new dependency,
# forbidden) over network_mode x interactive x entrypoint. Structural
# invariants only -- exact-shape coverage lives in the dedicated tests
# above.
# ---------------------------------------------------------------------------
_PAIRWISE_CASES = [
    ("unset", False, None),
    ("unset", False, "pg_dump"),
    ("unset", True, None),
    ("unset", True, "pg_dump"),
    ("no_network", False, None),
    ("no_network", False, "pg_dump"),
    ("no_network", True, None),
    ("no_network", True, "pg_dump"),
    ("network", False, None),
    ("network", False, "pg_dump"),
    ("network", True, None),
    ("network", True, "pg_dump"),
]


@pytest.mark.parametrize("network_mode,interactive,entrypoint", _PAIRWISE_CASES)
def test_docker_run_argv_pairwise_structural_invariants(
    network_mode, interactive, entrypoint
):
    kwargs: dict = {"interactive": interactive, "entrypoint": entrypoint}
    if network_mode == "no_network":
        kwargs["no_network"] = True
    elif network_mode == "network":
        kwargs["network"] = "a" * 12

    argv = mod._docker_run_argv(
        "/usr/bin/docker",
        "postgres:16-alpine",
        [],
        ["--sentinel-cmd"],
        **kwargs,
    )

    image_index = argv.index("postgres:16-alpine")
    # no command token leaks into the pre-image region
    assert "--sentinel-cmd" not in argv[:image_index]
    assert argv[image_index + 1 :] == ["--sentinel-cmd"]

    assert argv.count("--entrypoint") == (1 if entrypoint else 0)

    has_network_flag = "--network" in argv
    has_no_network_flag = "--network=none" in argv
    assert not (has_network_flag and has_no_network_flag)
    if network_mode == "network":
        assert has_network_flag
        assert not has_no_network_flag
    elif network_mode == "no_network":
        assert has_no_network_flag
        assert not has_network_flag
    else:
        assert not has_network_flag
        assert not has_no_network_flag

    assert ("-i" in argv) is interactive
    assert "-t" not in argv
    assert "--tty" not in argv
    assert "-it" not in argv


# ---------------------------------------------------------------------------
# _classify_docker_rc
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "rc,expected",
    [
        (0, "pg"),
        (1, "pg"),
        (2, "pg"),
        (124, "pg"),
        (125, "docker"),
        (126, "docker"),
        (127, "docker"),
        (128, "pg"),
        (255, "pg"),
        (-9, "pg"),
    ],
)
def test_classify_docker_rc_exact_literal_per_case(rc, expected):
    assert mod._classify_docker_rc(rc) == expected


# ---------------------------------------------------------------------------
# _docker_preflight
# ---------------------------------------------------------------------------
_PREFLIGHT_IMAGE = "dhi.io/postgres:16-alpine-dev"


def test_docker_preflight_happy_path_exact_calls_and_order(monkeypatch, fake_run):
    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    fake_run.queue_result(FakeCompletedProcess(stdout="24.0.5\n", returncode=0))
    fake_run.queue_result(FakeCompletedProcess(stdout="[{}]", returncode=0))

    mod._docker_preflight("/usr/bin/docker", _PREFLIGHT_IMAGE)

    assert len(fake_run.calls) == 2
    (args0, kwargs0), (args1, kwargs1) = fake_run.calls
    assert args0[0] == ["/usr/bin/docker", "version", "--format", "{{.Server.Version}}"]
    assert args1[0] == ["/usr/bin/docker", "image", "inspect", _PREFLIGHT_IMAGE]
    assert kwargs0.get("timeout") is not None
    assert kwargs1.get("timeout") is not None


def test_docker_preflight_daemon_unreachable_raises_before_inspect(
    monkeypatch, fake_run
):
    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    fake_run.queue_result(FakeCompletedProcess(stdout="", returncode=1))

    with pytest.raises(RuntimeError, match="daemon"):
        mod._docker_preflight("/usr/bin/docker", _PREFLIGHT_IMAGE)

    assert len(fake_run.calls) == 1


def test_docker_preflight_image_missing_names_docker_pull_command(
    monkeypatch, fake_run
):
    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    fake_run.queue_result(FakeCompletedProcess(stdout="24.0.5\n", returncode=0))
    fake_run.queue_result(FakeCompletedProcess(stdout="", returncode=1))

    with pytest.raises(RuntimeError) as exc_info:
        mod._docker_preflight("/usr/bin/docker", _PREFLIGHT_IMAGE)

    assert f"docker pull {_PREFLIGHT_IMAGE}" in str(exc_info.value)
    assert len(fake_run.calls) == 2


@pytest.mark.parametrize(
    "exc",
    [
        subprocess.TimeoutExpired(cmd=["docker"], timeout=15),
        PermissionError("permission denied"),
        FileNotFoundError("no such file or directory"),
    ],
)
def test_docker_preflight_binary_level_failures_raise_actionable_runtime_error(
    monkeypatch, fake_run, exc
):
    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    fake_run.queue_exception(exc)

    with pytest.raises(RuntimeError):
        mod._docker_preflight("/usr/bin/docker", _PREFLIGHT_IMAGE)


# ---------------------------------------------------------------------------
# _docker_child_env
# ---------------------------------------------------------------------------
def test_docker_child_env_full_dict_equality_happy_path(docker_env):
    pg_env = {"PGHOST": "127.0.0.1", "PGPORT": "5432", "PGUSER": "alice"}
    result = mod._docker_child_env(pg_env)
    assert result == {
        "HOME": "/home/tester",
        "PATH": "/usr/bin:/bin",
        "DOCKER_HOST": "unix:///var/run/docker.sock",
        "PGHOST": "127.0.0.1",
        "PGPORT": "5432",
        "PGUSER": "alice",
    }


@pytest.mark.parametrize(
    "secret_name", ["DB_DUMP_PASSPHRASE", "RAILWAY_API_TOKEN", "SOURCE_DSN"]
)
def test_docker_child_env_drops_secret_names_even_when_in_pg_env(
    docker_env, secret_name
):
    pg_env = {"PGHOST": "127.0.0.1", secret_name: "should-not-appear"}
    result = mod._docker_child_env(pg_env)
    assert secret_name not in result


def test_docker_child_env_allowlist_var_absent_from_os_environ_omitted(monkeypatch):
    monkeypatch.delenv("DOCKER_CONTEXT", raising=False)
    result = mod._docker_child_env({})
    assert "DOCKER_CONTEXT" not in result


def test_docker_child_env_fresh_dict_per_call_no_shared_state(docker_env):
    pg_env = {"PGHOST": "127.0.0.1"}
    first = mod._docker_child_env(pg_env)
    first["PGHOST"] = "mutated"
    first["NEW_KEY"] = "leaked"
    second = mod._docker_child_env(pg_env)
    assert second["PGHOST"] == "127.0.0.1"
    assert "NEW_KEY" not in second


def test_docker_child_env_allowlist_wins_on_path_and_docker_host_collision(docker_env):
    pg_env = {"PATH": "attacker-controlled", "DOCKER_HOST": "attacker-controlled"}
    result = mod._docker_child_env(pg_env)
    assert result["PATH"] == "/usr/bin:/bin"
    assert result["DOCKER_HOST"] == "unix:///var/run/docker.sock"


def test_docker_child_env_pg_env_wins_on_pg_star_keys(docker_env):
    pg_env = {"PGHOST": "127.0.0.1", "PGPASSWORD": "s3cret"}
    result = mod._docker_child_env(pg_env)
    assert result["PGHOST"] == "127.0.0.1"
    assert result["PGPASSWORD"] == "s3cret"


@given(
    pg_env=st.dictionaries(
        st.sampled_from(
            [
                "PGHOST",
                "PGPORT",
                "PGUSER",
                "PGPASSWORD",
                "PGDATABASE",
                "PGSSLMODE",
                "DB_DUMP_PASSPHRASE",
                "RAILWAY_API_TOKEN",
                "SOURCE_DSN",
            ]
        ),
        st.text(min_size=1, max_size=10),
        max_size=9,
    )
)
@settings(
    deadline=None,
    max_examples=50,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_docker_child_env_property_subset_and_no_secrets(docker_env, pg_env):
    result = mod._docker_child_env(pg_env)
    allowed = set(mod._DOCKER_CONTEXT_ENV_ALLOWLIST) | set(pg_env)
    assert set(result) <= allowed
    for secret in mod._DOCKER_SECRET_ENV_VAR_NAMES:
        assert secret not in result


# ---------------------------------------------------------------------------
# _pg_dump_major_version -- T13, the non-negotiable scenario: this
# function had ZERO direct test coverage anywhere before this fix (only
# ever monkeypatched away in other tests), and is the exact function that
# failed on the real GATE-evidence run (entrypoint-dispatch bug -- see
# docs/changes/2026-08-09-db-restore-drill.md's 2026-08-11-bis addendum).
# This test's argv assertion must fail against the pre-fix shape
# (`["pg_dump", "--version"]` with no `--entrypoint`) and pass against the
# fixed shape.
# ---------------------------------------------------------------------------
def test_pg_dump_major_version_full_argv_equality_against_subprocess_run(
    monkeypatch, fake_run
):
    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    fake_run.queue_result(
        FakeCompletedProcess(stdout="pg_dump (PostgreSQL) 16.14\n", returncode=0)
    )

    result = mod._pg_dump_major_version(
        "/usr/bin/docker", "dhi.io/postgres:16-alpine-dev"
    )

    assert result == 16
    assert len(fake_run.calls) == 1
    (args, _kwargs) = fake_run.calls[0]
    assert args[0] == [
        "/usr/bin/docker",
        "run",
        "--rm",
        "--pull=never",
        "--security-opt",
        "no-new-privileges",
        "--network=none",
        "--entrypoint",
        "pg_dump",
        "dhi.io/postgres:16-alpine-dev",
        "--version",
    ]
    assert "-e" not in args[0]


@pytest.mark.parametrize(
    "stdout,expected_major",
    [
        ("pg_dump (PostgreSQL) 16.14\n", 16),
        ("pg_dump (PostgreSQL) 17.2\n", 17),
        ("pg_dump (PostgreSQL) 16\n", 16),
    ],
)
def test_pg_dump_major_version_parses_version_string(
    monkeypatch, fake_run, stdout, expected_major
):
    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    fake_run.queue_result(FakeCompletedProcess(stdout=stdout, returncode=0))

    result = mod._pg_dump_major_version(
        "/usr/bin/docker", "dhi.io/postgres:16-alpine-dev"
    )
    assert result == expected_major


@pytest.mark.parametrize(
    "rc,match",
    [
        (125, "docker"),
        (126, "docker"),
        (127, "docker"),
        (1, "pg"),
        (124, "pg"),
        (128, "pg"),
    ],
)
def test_pg_dump_major_version_nonzero_rc_classifies_via_classify_docker_rc(
    monkeypatch, fake_run, rc, match
):
    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    fake_run.queue_result(FakeCompletedProcess(stdout="", returncode=rc))

    with pytest.raises(RuntimeError, match=match):
        mod._pg_dump_major_version("/usr/bin/docker", "dhi.io/postgres:16-alpine-dev")


def test_pg_dump_major_version_unparseable_stdout_raises_with_actionable_message(
    monkeypatch, fake_run
):
    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    fake_run.queue_result(
        FakeCompletedProcess(stdout="not a version string", returncode=0)
    )

    with pytest.raises(RuntimeError, match="could not parse"):
        mod._pg_dump_major_version("/usr/bin/docker", "dhi.io/postgres:16-alpine-dev")
