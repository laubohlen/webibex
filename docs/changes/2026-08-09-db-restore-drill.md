# CR: local DB dump + restore-drill tool, satisfying the id_code-migration GATE

**What changed:**
- New: `scripts/db_restore_drill.py` (880 lines at the initial commit, 929 at
  the final commit after the TLS fix and `SOURCE_DSN` addition) — a one-off
  local operator
  tool. Fetches prod `DATABASE_URL` via Railway's GraphQL API (`fetch_database_url`),
  connects read-only (`preflight_source`/`_connect_readonly`, TLS-enforced —
  see the security fix below), streams `pg_dump -Fc` straight into
  `openssl enc -aes-256-cbc -pbkdf2 -iter 600000 -salt` so no plaintext dump
  ever touches disk (`dump_encrypted`), restores into an ephemeral
  `testcontainers` Postgres and verifies row counts + an `Animal` spot-check
  across `Animal`/`Region`/`Location`/`IbexImage`/`IbexChip`/`Embedding`
  (`restore_and_verify`, `collect_expected`). All credentials travel via
  subprocess env only, never argv; every DSN that could reach a log line or
  exception is routed through `redact_dsn()`. Optional `SOURCE_DSN` env var
  (not a CLI flag) bypasses the Railway fetch entirely for local dry-runs
  against a throwaway/migrated database, making `RAILWAY_API_TOKEN`
  unnecessary in that mode.
- New: `tests/scripts/` — 112 tests total (109 real P0, 3 skip-gated integration
  stubs under a new `live_pg_restore` pytest marker registered in
  `pytest.ini`), covering DSN redaction, libpq env translation, the
  loopback-only restore-target guard (including bypass-shaped hostnames:
  `127.0.0.1.evil.com`, decimal/hex-encoded loopback IPs), Railway GraphQL
  fetch (both token kinds, error/malformed-response handling, never logging
  the full `data` payload), table-name parity against the live Django model
  registry, dump/restore subprocess argv+env shape, and CLI orchestration
  including the `SOURCE_DSN` bypass.
- New: `pyrightconfig.json` (basic mode, excludes migrations/`.venv`/static
  dirs) — this repo had no pyright config at all before; added specifically
  because a bare `pyproject.toml` (the initially-considered alternative) was
  tested in an isolated scratch dir and found to make `uv run` silently
  create a shadow `.venv`, which would have broken every `uv run pytest`/
  `ruff`/`pyright` invocation against this repo's existing
  `requirements.txt`-based venv.
- Modified: `requirements-dev.txt` (+`hypothesis==6.165.2`,
  +`testcontainers[postgres]==4.15.0`, both dev-only). `requirements.txt`
  itself confirmed byte-for-byte untouched throughout.
- Security fix included in the same commit as the initial implementation
  (caught by this session's own `/security-review` pass, independently
  adversarially re-verified by a second sub-agent at confidence 8/10 both
  times): `_connect_readonly()`'s two production connections
  (`preflight_source`, `main()`'s direct fetch) were calling
  `psycopg2.connect(dsn, ...)` with the raw DSN, falling through to libpq's
  own `sslmode=prefer` default — silently allowing a TLS downgrade despite
  the script's documented "refuses to connect... without TLS" guarantee
  (which was only actually enforced on the `pg_dump`/`pg_restore` subprocess
  paths via `libpq_env()`). Fixed by deriving `sslmode` through
  `libpq_env(dsn)` before connecting; 2 new tests + 1 extended assertion
  lock this in.

**Follow-up action:** none required to land locally — 3 commits sit on
branch `db-restore-drill` (off `main`), not yet pushed. User has a private
GitLab mirror (`origin_gitlab` remote) and intends to push themselves. The
actual GATE-evidence run (real Railway fetch + real Docker + real
`pg_dump`/`pg_restore`) still needs to happen on the user's own machine —
see Trigger below.

**Do NOT:**
- Treat this as resuming the separately-paused `backup_db` → B2
  recurring-automation TODO in `docs/security-remediation-plan.md` (~line
  940). That decision is still "revisit when opening to other users," per
  the professor's 2026-08-08 answer — unrelated and unaffected by this CR.
  This script only dumps + restores + verifies, once, manually; it does not
  schedule anything or touch B2.
- Add a `pyproject.toml` to this repo for pyright/ruff config without first
  verifying it won't make `uv run` shadow-`.venv` the existing setup (tested
  and confirmed a real risk this session — see `pyrightconfig.json` above).
  `ruff.toml` already covers ruff; only pyright needed a home.
- Reintroduce `--source-dsn` as a CLI flag. It was built that way first,
  self-caught in review as inconsistent with the script's own "secrets are
  NEVER accepted on argv" invariant (a DSN can carry a password, visible in
  `ps`/shell history), and reverted to the `SOURCE_DSN` env var before
  committing.
- Assume Postgres 17 is prod's confirmed major version. The user has a local
  `postgres:17.9` Docker container with client tools, unrelated to prod's
  actual version — explicitly not to be assumed, must be confirmed via the
  real Railway fetch.
- Route `pg_dump`/`pg_restore` calls through `docker exec` into a
  persistent, already-running container. Considered and explicitly rejected
  this session: `docker exec` doesn't forward the subprocess `env=`
  credentials automatically, and `PGHOST=localhost` would resolve to the
  container itself, not the host — a real networking trap, not just extra
  plumbing. **Superseded in part by the 2026-08-10 addendum below**: the
  "native host install" alternative this bullet originally assumed was
  itself dropped (macOS host can't run any container's Linux binaries
  natively); one-shot `docker run` (not `docker exec`) is now the chosen
  approach — the prohibition here applies specifically to `docker exec`
  into a persistent container, not to `docker run`.

**Trigger:** ~~the user's Railway account login (blocked as of 2026-08-09,
verification email not arriving)~~ — resolved 2026-08-10, login now works,
and prod's Postgres major version is confirmed 16.13 (see addendum below).
The actual remaining blocker is the docker-run wiring under "Still pending"
below. Once that's implemented: run the gitignored
`tmp/db_restore_drill_preflight.py` diagnostic script first (resolves the
GraphQL endpoint domain, token kind, and whether Public Networking/
`DATABASE_PUBLIC_URL` is enabled, without printing any secret values), then
run `scripts/db_restore_drill.py` itself for the real GATE-evidence run —
needs Docker (per the 2026-08-10 addendum's `docker run` wrapper, not a
native `postgresql-client` install) + the real `RAILWAY_API_TOKEN`/project/
environment IDs.

**Why:** the GATE (`docs/security-remediation-plan.md` ~line 1015) blocks
the `Animal.id_code` `max_length` schema migration from shipping until a
backup mechanism has been proven to actually *restore*, not just run and
upload successfully — "a backup that writes a file but has never been
restored is unverified." This CR builds the tool that produces that
evidence; it does not itself constitute the evidence (the live run is still
pending on Railway access).

**Verify:** `uv run pytest tests/scripts/ -v` → 112 collected, 109 passed + 3
skipped (integration stubs). `uv run pytest -q` (full suite) → 388 passed, 4
skipped, 1 xfailed — no regressions against the pre-session baseline (279
passed at session start, confirmed by checking out `3aa7187` and re-running:
279 passed, 1 skipped, 1 xfailed; 279+109=388, 1+3=4, arithmetic reconciles).
`uv run ruff check scripts/db_restore_drill.py
tests/scripts/` → clean. `git diff requirements.txt` → empty. `/sonar fetch`
(no scan re-run — Docker unavailable in the coding sandbox — read existing
findings instead): 376 pre-existing CRITICAL/MAJOR project-wide, 0 touching
this diff. `/insecure-defaults`: 0 findings. `/security-review`: 1 confirmed
MEDIUM finding (TLS downgrade, above), fixed and re-verified green, no
others. `/request-adherence --impl`: PASS, 5/5 requirements COVERED.

**Known flaky test (found 2026-08-10, not yet fixed):**
`test_db_restore_drill_property_password_never_survives` in
`tests/scripts/test_db_restore_drill_dsn.py` filters generated passwords
against `_DSN_STATIC_TOKENS` by exact match only, but the leak-check
assertion is a substring check against the full redacted DSN. A password
that's a substring of a static DSN part (e.g. `pw="me"`, substring of
`dbname`) fails the assertion even though redaction is correct — a false
positive, not a real leak. Reproduced directly with `pw="me"`/`"db"`/`"st"`/
`"na"`/`"post"`. The documented green run above is real but seed-dependent;
Hypothesis can regenerate a falsifying example on a future run. Fix: filter
substrings of the static tokens, not just exact matches, or assert on the
extracted password field instead of the whole redacted string.

**Rollback:** the branch isn't merged into `main` and hasn't been pushed —
delete the local `db-restore-drill` branch (and `pyright-config` if also
unwanted) to fully undo. No migration, no schema change, no production code
path touched (`requirements.txt` untouched, `scripts/` isn't part of the
Django app or the `--cov` gate) — safe to drop cleanly at any point before
push.

**Addendum (2026-08-10):** Railway login now works. Prod Postgres confirmed
16.13 (16.14 available) — resolves the "don't assume 17" open item above.

Client-tooling decision revisited: ruled out native `apt`/`brew` install and
ruled out extracting binaries from a Docker image (Linux ELF binaries don't
run on the user's macOS host regardless of which image they come from).
Settled on wrapping `pg_dump`/`pg_restore` calls in `docker run --rm -i`
against `dhi.io/postgres:16-alpine-dev` (hardened/minimal image; user-provided
package manifest `tmp/postgresql-client16-alpine.1.txt` confirms the
`postgresql-client` package ships `pg_dump`/`pg_restore`/`psql` — that file
lists command names only, no version string; the 16.14-r0 figure came from
the user's message, not independently verified against the manifest).
Design sketch, not yet implemented:
- Env vars passed as `docker run -e VARNAME` (no `=value`) so Docker reads
  the value from the calling process's own env — keeps the existing
  "credentials never touch argv" invariant intact.
- Restore leg reaches the ephemeral testcontainers Postgres via
  `--network container:<testcontainers_id>` (shares its network namespace —
  no external network route needed for that leg at all).
- One-shot `docker run` per call, not a persistent container +
  `docker exec` (avoids the env-passthrough/`PGHOST=localhost` pitfalls
  flagged in the "Do NOT" section above, which applied to `docker exec`).

**New TODO (2026-08-10, unrelated to this CR — do not conflate with the
above):** study upgrading prod Postgres from 16.13 to 17.9, to match
`tmgame`'s version. Not scoped or started this session.

**Addendum (2026-08-11): docker-run wiring implemented.** Routed through
code-planner (Opus) → code-analyst (Opus) → code-executioner this
session, red-first per the planning-TDD pipeline. `pg_dump`/`pg_restore`
now run inside one-shot `docker run --rm` (never `docker exec` into a
persistent container); zero `shutil.which("pg_dump")`/`("pg_restore")`
remains in the production path.

Confirmed decisions this session (all implemented as specified):
- Container-id accessor: `container.get_wrapped_container().id` (docker-py's
  `Container.id`, reads `.attrs["Id"]`, full 64-char hex) — confirmed by
  live inspection, no fallback branch needed.
- Restore leg uses `PGSSLMODE=disable` against the ephemeral testcontainers
  Postgres — legal only because `_assert_local_target` has already proven
  a loopback target; documented inline in `_restore_target_env`'s
  docstring, not a silent downgrade. The source-DSN leg still refuses
  `sslmode=disable` unconditionally (unchanged).
- Docker child env: 8-var allowlist (`HOME`, `PATH`, `DOCKER_HOST`,
  `DOCKER_CONTEXT`, `DOCKER_CONFIG`, `DOCKER_CERT_PATH`, `DOCKER_TLS_VERIFY`,
  `XDG_RUNTIME_DIR`) wins on collision with `pg_env`; `pg_env` wins on
  everything else (the `PG*` keys). 3 secret names
  (`DB_DUMP_PASSPHRASE`/`RAILWAY_API_TOKEN`/`SOURCE_DSN`) dropped by
  case-sensitive exact match only.
- `_PROD_MAJOR_VERSION_RE` regex-safety bug fixed (`^\d+$` + `.match()` →
  `\A[0-9]+\Z`): the old pattern accepted `"16\n"` (via `$` matching
  before a trailing newline) and `"１６"`/fullwidth-or-Arabic-Indic digits
  (via `\d` being Unicode-aware) — both verified live, both now rejected.
  New `_IMAGE_REF_RE`/`_CONTAINER_ID_RE` follow the same regex-safety
  class (`fullmatch`-equivalent anchors, ASCII-only classes).
- Pull policy: `docker version` → `docker image inspect` (exactly 2 calls,
  in that order — order is load-bearing, both return the same rc when the
  daemon is down, so inspect-first would misdiagnose a dead daemon as a
  missing image) then `--pull=never` always on every `docker run` — never
  an implicit pull.

New production surface in `scripts/db_restore_drill.py`: `DEFAULT_PG_CLIENT_IMAGE`,
`_docker_child_env`, `_docker_run_argv`, `_IMAGE_REF_RE`, `_CONTAINER_ID_RE`,
`_ENV_NAME_RE`, `_classify_docker_rc`, `_docker_path`, `_docker_preflight`,
`_testcontainers_container_id`, `_restore_target_env`; `_pg_dump_major_version`
signature changed from 1-arg (`pg_dump_path`) to 2-arg (`docker_path, image`);
`preflight_source`/`dump_encrypted`/`restore_and_verify` gained a keyword-only
`image` parameter (default `DEFAULT_PG_CLIENT_IMAGE`); new `--pg-client-image`
CLI flag threads the same value into all three (not a secret — safe on argv).

New test file `tests/scripts/test_db_restore_drill_docker.py` (102 tests):
argv shape (exact full-list equality, not membership), image-ref/container-id/
env-name validation (parametrize + Hypothesis properties), `_classify_docker_rc`
boundary table, `_docker_preflight` (happy path, daemon-down, image-missing,
binary-level failures), `_docker_child_env` (allowlist/secret/collision
semantics + a subset-and-no-secrets Hypothesis property). Existing test files
updated for the new docker-wrapped argv/env shapes (`test_db_restore_drill_dump.py`,
`test_db_restore_drill_preflight.py`, `test_db_restore_drill_restore.py`,
`test_db_restore_drill_main.py`) and the daemon-reachability skip gate
(`test_db_restore_drill_live_integration.py` — `shutil.which("docker") is
None` alone is `False` in this sandbox, since the `docker` binary IS
installed even though the daemon isn't reachable; the gate now actually
probes `docker version`).

Also fixed as part of this session (separate commit, before the docker-run
work): the flaky `test_redact_dsn_property_password_never_survives` noted
below (R9) — root cause was exact-membership skip-list vs. substring
leak-check, not a redaction bug. `derandomize=True` added so the fix is
deterministic, not seed-lucky.

**Verify (2026-08-11):** `uv run pytest tests/scripts/ -v` → 223 passed,
3 skipped (up from 112 collected / 109 passed + 3 skipped at the prior
checkpoint — net +111 tests). `uv run pytest -q` (full repo) → 502 passed,
4 skipped, 1 xfailed, 0 failed (up from 388 passed/4 skipped/1 xfailed;
note the prior checkpoint's numbers already included a pre-existing,
unrelated env-drift failure this session also fixed along the way —
`test_restore_and_verify_without_testcontainers_gives_actionable_message`
assumed `testcontainers` was genuinely absent from the sandbox venv, which
stopped being true after this session's earlier `RAILWAY_API_TOKEN`
addendum work; fixed by poisoning `sys.modules` with `None` instead of
relying on real absence). `uv run ruff check scripts/db_restore_drill.py
tests/scripts/` → clean. `uv run pyright scripts/db_restore_drill.py`
(project's own basic-mode config) → 0 errors/warnings. `git diff
requirements.txt requirements-dev.txt` → empty, confirmed byte-for-byte
untouched.

**Deviations from this session's plan, flagged explicitly:**
- **Commit-split conflict, resolved in favor of the explicit two-commit
  breakdown.** The plan's decision-3c text said the `_PROD_MAJOR_VERSION_RE`
  regex fix should land "in the same commit as the new `_IMAGE_REF_RE`/
  `_CONTAINER_ID_RE` guards" (i.e. the docker-wiring commit), but the plan's
  own "Explicitly out of scope" section separately enumerated commit 1 as
  "R9 flaky-test fix + the `_PROD_MAJOR_VERSION_RE` regex-safety fix" —
  directly contradicting the first instruction. Resolved in favor of the
  explicit two-commit breakdown (more authoritative — phrased as the final
  commit-boundary directive) — `_PROD_MAJOR_VERSION_RE` landed in commit 1
  with R9, not commit 2 with the new regexes.
- **`fake_run` fixture did not pre-exist**, contrary to the plan's claim
  ("that fixture already exists and is currently unused, confirm/reuse
  it"). Only the underlying `FakeCompletedProcess` duck-type class existed
  in `tests/scripts/conftest.py`; no fixture wrapped it. Built the
  `fake_run` fixture (records calls, replays a result/exception queue) on
  top of the existing class rather than assuming it needed no new code.

**Rollback:** the branch isn't merged into `main` and hasn't been pushed —
delete the local `db-restore-drill` branch to fully undo. No migration, no
schema change, no production code path touched (`requirements.txt`
untouched, `scripts/` isn't part of the Django app or the `--cov` gate) —
safe to drop cleanly at any point before push.

**Still pending:** the actual GATE-evidence run itself (real Railway fetch
+ real Docker + real `pg_dump`/`pg_restore`) — needs to happen on the
user's own machine where Docker Desktop and real Railway credentials are
available; none reachable in this sandbox (`docker` binary present, daemon
unreachable — confirmed live this session).

**Addendum (2026-08-11 bis): entrypoint-dispatch bug found on the real
GATE-evidence run attempt.**

Live run (user's machine, real `dhi.io/postgres:16-alpine-dev`, real
Docker) failed at `_pg_dump_major_version`:
```
restore drill failed: pg_dump --version failed inside docker (rc=1, pg-level failure): ERROR: Database is uninitialized and the superuser password is not specified.
       POSTGRES_PASSWORD or POSTGRES_PASSWORD_FILE must be set to a non-empty value.
```

**Root cause:** the docker-run wiring invokes `docker run ... <image>
pg_dump --version` — positional command after the image name, relying on
the image's own ENTRYPOINT/CMD to dispatch. The official `postgres`
Docker image's entrypoint has smart dispatch (first arg `pg_dump`, not
`postgres`/a flag → exec directly, no init). `dhi.io/postgres:16-alpine-dev`'s
hardened entrypoint does not replicate that dispatch — it routes into
server-init regardless of the command given after the image name.

**Confirmed fix, live-tested by the user:**
```bash
docker run --rm --network=none --entrypoint pg_dump dhi.io/postgres:16-alpine-dev --version
# -> pg_dump (PostgreSQL) 16.14
```
Bypassing the image's default entrypoint via `docker run --entrypoint
<binary>` and supplying only arguments (not the binary name) as `command`
works.

**Plan (code-planner, Opus)**, routed through the planning-TDD pipeline
again given this touches the already-reviewed subprocess/docker boundary:
- `_docker_run_argv` gains `entrypoint: str | None = None` (emits
  `--entrypoint <value>` in the pre-image flag region; when set, `command`
  becomes args-only, e.g. `["--version"]` instead of `["pg_dump",
  "--version"]`) and `no_network: bool = False` (emits `--network=none`,
  mutually exclusive with `network=<container-id>` — both set raises
  `ValueError`).
- New `_ENTRYPOINT_RE = re.compile(r"\A[A-Za-z_][A-Za-z0-9_.-]{0,63}\Z")` —
  bare binary name only, no `/`, same regex-safety class as
  `_IMAGE_REF_RE`/`_CONTAINER_ID_RE`/the fixed `_PROD_MAJOR_VERSION_RE`.
  Design rationale: the argv region *before* IMAGE is validated (docker
  parses its own flags there — `_IMAGE_REF_RE` exists precisely because an
  image ref starting with `-` becomes a flag, not an image); positionals
  *after* IMAGE (`command`) are trusted by construction and were never
  validated. `entrypoint` sits in the validated region, so it gets
  validated too — keeps the module's posture stateable in one rule rather
  than case-by-case.
- Call sites: `_pg_dump_major_version` → `entrypoint="pg_dump",
  no_network=True` (a `--version` probe has zero legitimate network need —
  the user's own live-tested addition). `dump_encrypted`'s dump leg →
  `entrypoint="pg_dump"`, no `no_network` (needs real network to reach
  Railway). `restore_and_verify`'s restore leg → `entrypoint="pg_restore"`,
  keeps `network=<container-id>`, no `no_network` (mutually exclusive with
  the container-namespace join it needs).
- `--network=none` deliberately NOT folded into the existing `network`
  parameter as a sentinel value — `network="none"` would conflict with the
  existing test asserting `"none"` must be *rejected* as a container-id.
  Kept as a separate `no_network: bool`.

**Corrections the planner found by reading the actual current test files**
(not assumed from the bug report): `test_db_restore_drill_preflight.py`
and `test_db_restore_drill_main.py` need **zero edits** — neither has any
docker-argv assertion. `_pg_dump_major_version` had **no direct test
coverage anywhere** — the exact function that failed live — closed by
this fix. One existing test (`image_index precedes command_index`) does
`argv.index("pg_dump")`, which inverts once `--entrypoint pg_dump` is
emitted before the image — rewritten, not just re-literal'd. One
pre-existing test (`test_db_restore_drill_restore.py`'s post-image-argv
check) was membership-style (`in`), not full-equality, due to a
non-deterministic `-e` name set — upgraded to full-equality as part of
this fix, repaying a convention deviation flagged during planning.

**Unresolved at plan time:** the restore-leg fix (`--entrypoint
pg_restore`) is inferred from the same root cause, not independently
live-verified — the drill failed earlier, at the version probe, before
ever reaching `pg_restore`.

**Test spec (code-analyst, Opus)**: 23 scenarios (P0: 7, P1: 12, P2: 4).
Independently re-verified all 5 of the planner's file-level claims by
reading the actual test files (not trusting the plan), plus found one
more: `_BAD_CONTAINER_IDS` (`test_db_restore_drill_docker.py`) already
requires `"none"` to be *rejected* as a container-id — empirical proof
the `no_network: bool` design (vs. a `network="none"` sentinel) was
correct, since the sentinel would have forced deleting that existing
guard.

Five **exact diffs to existing tests** (not new tests) identified, each
tied to a specific line range in the current test files:
- Diff A/B: `test_db_restore_drill_docker.py`'s dump-leg and restore-leg
  exact-shape tests — insert `--entrypoint <binary>` before the image,
  remove the positional binary name from `command`.
- Diff C: `test_docker_run_argv_image_index_precedes_command_index` —
  full rewrite, not a re-literal. Its current `argv.index("pg_dump")`
  would silently resolve to the *entrypoint value's* position once the
  fix lands, inverting the assertion into a false pass.
- Diff D: `test_db_restore_drill_dump.py`'s `p1.argv` literal, same
  entrypoint-token insertion.
- Diff E: `test_db_restore_drill_restore.py` — add the `docker_env`
  fixture (makes the `-e` name set deterministic) and convert the
  post-image check from membership-style to full-list equality, repaying
  a pre-existing convention deviation.

**Non-negotiable scenario**: T13 — direct full-argv-equality coverage for
`_pg_dump_major_version`, which had **zero tests anywhere** before this
fix (grep-confirmed: 5 monkeypatch sites across the suite, 0 assertions).
That's the exact function that failed on the live run — its total lack of
coverage is the documented root cause of this bug shipping in the first
place. Every other scenario guards a surface that already had *some*
test.

`_docker_run_argv` crossed the 8-parameter complexity-triage threshold
(4 positional + `network`/`interactive`/`entrypoint`/`no_network`
keyword-only). Resolved via `targeted_counter_inputs` + a 12-case
hand-rolled pairwise sweep (T23) over the two genuinely-interacting
params (`network`×`no_network` mutual exclusion, `entrypoint`×`command`
semantic coupling) — no new dependency (`allpairspy` explicitly rejected,
would violate R7's "no new dependency" constraint for a 12-case matrix).

Mutation-testing follow-up flagged in advance: `cosmic-ray` scoped to
this file post-implementation, with one likely-equivalent mutant
pre-identified (`if entrypoint:` vs. `if entrypoint is not None:` — both
correct given `""` is already rejected by `_ENTRYPOINT_RE` upstream).

**Implementation (code-executioner)**: complete, routed through the
planning-TDD pipeline (red-first). All requirements (R1-R7) implemented
as specified; no deviations.

`_docker_run_argv` gained two new keyword-only params:
`entrypoint: str | None = None` (emits `--entrypoint <value>` in the
pre-image flag region, after the sorted `-e` block, immediately before
IMAGE; `command` becomes args-only when set) and `no_network: bool =
False` (emits the single token `"--network=none"`, mirroring the
existing `"--pull=never"` style -- never the two-token `["--network",
"none"]` form). Both default to inert, so `entrypoint=None` produces
byte-identical argv to before this fix. New `_ENTRYPOINT_RE =
re.compile(r"\A[A-Za-z_][A-Za-z0-9_.-]{0,63}\Z")`, validated
fail-before-build alongside `image`/`network`/`env_names`; a design-
rationale comment in the source states the module's posture in one rule:
everything before IMAGE is validated, positionals after IMAGE are
trusted. `network` and `no_network` together raise `ValueError` before
any argv is built.

Call sites: `_pg_dump_major_version` -> `entrypoint="pg_dump",
no_network=True` (a `--version` probe has zero legitimate network need,
one-line comment added). `dump_encrypted`'s pg_dump leg ->
`entrypoint="pg_dump"`, no `no_network` (needs real network to reach the
Railway-hosted Postgres). `restore_and_verify`'s pg_restore leg ->
`entrypoint="pg_restore"`, keeps `network=<container-id>`/
`interactive=True`, no `no_network` (mutually exclusive with the
container-namespace join it needs).

Test changes: the 5 exact diffs to existing tests identified during
planning were applied as specified (Diffs A-E across
`test_db_restore_drill_docker.py`, `test_db_restore_drill_dump.py`,
`test_db_restore_drill_restore.py`), including Diff C's full rewrite
(not a re-literal) of `test_docker_run_argv_image_index_precedes_command_index`,
and Diff E's restore-leg happy-path test gaining the `docker_env`
fixture and converting its post-image argv check from membership to
full-list equality. `test_dump_encrypted_env_separation` confirmed to
need zero edits, as expected (verified via `git diff`, not just
assumption) -- the credential-boundary invariant held. All 5
zero-edit files claimed by the plan (`test_db_restore_drill_preflight.py`,
`test_db_restore_drill_main.py`, `test_db_restore_drill_live_integration.py`,
`test_db_restore_drill_dsn.py`, `test_db_restore_drill_railway.py`) were
independently re-confirmed to need zero edits.

New coverage added (all in `test_db_restore_drill_docker.py` except
where noted): the T13 non-negotiable scenario -- a direct,
standalone `_pg_dump_major_version` full-argv-equality test, mocking
`subprocess.run` via the existing `fake_run` fixture, asserting the
exact argv that produced the live bug now includes `--network=none
--entrypoint pg_dump`; version-string parsing (16.14/17.2/16);
rc-classification failure branches (docker-level 125/126/127, pg-level
1/124/128, unparseable-stdout); `_ENTRYPOINT_RE` reject/accept
parametrize sets plus a length-boundary triple (63/64/65 ->
True/True/False) and two Hypothesis properties (leading-dash,
embedded-bad-char) mirroring the existing `_IMAGE_REF_RE` properties;
`entrypoint=None` back-compat byte-identical-argv guard;
`network`/`no_network` conjunction + counter-cases; `no_network`
single-token-form and pre-image-position checks; the existing
`-t`/`--tty`/`-e`-name-only tests extended to a 2x2 sweep over
`interactive` x `entrypoint` presence; and a 12-case hand-rolled
(no `allpairspy`) pairwise sweep over `network_mode` x `interactive` x
`entrypoint`, asserting structural invariants (no command-token leakage
pre-image, `--entrypoint` count, mutual exclusion of
`{--network, --network=none}`, `-i` iff `interactive`, tty flags never
present).

One test-infra gap found and fixed, not anticipated by the plan:
`FakeCompletedProcess` (`tests/scripts/conftest.py`) had no `stderr`
attribute -- harmless until now because no existing test exercised
`_pg_dump_major_version`'s failure branch (`result.stderr.strip()`),
which is exactly the coverage hole this fix closes. Added an optional
`stderr=""` constructor param; a 2-line, test-infra-only change.

Mutation-testing follow-up resolved, not deferred: the pre-identified
likely-equivalent mutant (`if entrypoint:` vs `if entrypoint is not
None:` in the argv-construction branch) was confirmed equivalent by
direct check -- `_ENTRYPOINT_RE.match("")` is `None`, so an empty string
never survives validation to reach that branch either way. No `cosmic-ray`
run performed this session (Docker unavailable in the sandbox is
unrelated to this being a pure-Python check); flagged here for the
user's own judgment on whether a full scoped run is still wanted.

**Verify (2026-08-11 ter):** `uv run pytest tests/scripts/ -v` -> 290
passed, 3 skipped (up from 223 passed/3 skipped at the prior checkpoint
-- net +67 tests). `uv run pytest -q` (full repo) -> 569 passed, 4
skipped, 1 xfailed, 0 failed (up from 502 passed; 502+67=569, arithmetic
reconciles). `uv run ruff check scripts/db_restore_drill.py
tests/scripts/` -> clean (2 findings caught and fixed during this
session: an unused `noqa: RUF001` and one `E501` line-length violation,
both in new test code). `uv run pyright scripts/db_restore_drill.py`
(project's own basic-mode config) -> 0 errors/warnings.

**Staging decision:** this fix corrects the exact same uncommitted,
already-staged docker-run-wiring surface introduced by the 2026-08-11
addendum above (same functions: `_docker_run_argv`, same call sites:
`_pg_dump_major_version`/`dump_encrypted`/`restore_and_verify`) -- since
nothing on this branch has been committed for that work yet, there is no
prior commit to "amend" in the git sense. Extended the existing staged
snapshot directly: `git add`'d the 5 touched files
(`scripts/db_restore_drill.py`, `tests/scripts/conftest.py`,
`tests/scripts/test_db_restore_drill_docker.py`,
`tests/scripts/test_db_restore_drill_dump.py`,
`tests/scripts/test_db_restore_drill_restore.py`) into the same staged
state as the rest of the docker-run-wiring change, rather than treating
it as a second logical commit. Not committed -- per this project's
established pattern this session, the user reviews and commits
personally; a draft commit message was written to
`tmp/db_restore_drill_entrypoint_fix_commit_msg.txt` (gitignored).

**Addendum (2026-08-11 ter): real GATE-evidence run, PASS.**

Ran on the user's own machine (real Docker Desktop, real Railway
credentials, real prod database) — the actual evidence run this whole CR
exists to produce.

One more fix needed first, unrelated to the entrypoint-dispatch bug: the
default `--variable-name DATABASE_URL` resolves to
`postgres.railway.internal`, Railway's *private*-network hostname —
unreachable from outside Railway's own network. Fixed by passing
`--variable-name DATABASE_PUBLIC_URL` instead (the flag already existed;
just wasn't being used by the wrapper script). No code change to
`scripts/db_restore_drill.py` itself, only to `tmp/run_db_restore_drill.sh`.

Result:
```
=== restore drill report ===
PASS core_animal: count=110
PASS core_region: count=3
PASS core_location: count=143
PASS core_ibeximage: count=143
PASS core_ibexchip: count=142
PASS core_embedding: count=142
PASS spot-check: Animal row matches
=== overall: PASS ===
```

This also independently confirms the restore-leg's `--entrypoint
pg_restore` fix, which had only been inferred (not live-verified) at
implementation time — `pg_restore` ran and restored correctly against
`dhi.io/postgres:16-alpine-dev`, same as the dump leg.

Benign, non-fatal `WARNING: database "railway" has a collation version
mismatch` lines appeared 3 times during the run (client/server glibc
collation library version drift — 2.36 vs 2.41; stderr-only, never
touches the piped binary dump stream, doesn't affect pg_dump/pg_restore's
exit code). Not fixed, not blocking — the hinted fix
(`ALTER DATABASE railway REFRESH COLLATION VERSION`) is a
production-mutating operation, deliberately out of scope for a read-only
verification drill.

**GATE satisfied**: `docs/security-remediation-plan.md`'s "restore drill
required before the id_code max_length migration ships" gate — see that
doc for the full record. The migration itself is unblocked but not yet
scoped or deployed; that's a separate future step.

**Still pending:** none for this CR's original scope. Remaining follow-ups
are tracked as separate TODOs in `docs/security-remediation-plan.md`:
containerizing the drill tool itself, hash-pinning `requirements.txt`,
the `testcontainers.postgres` deprecation, prod Postgres 16→17.9 upgrade,
and stage-level progress logging.

**Deferred to next session (2026-08-11, user decision):**
- Run `/post-production` on this CR's diff — skipped this session given
  the size of the change and that the real GATE-evidence run already
  independently confirmed the tool works end to end; deferred, not
  skipped outright.
- Promote `tmp/run_db_restore_drill.sh` and
  `tmp/run_railway_token_smoke_test.sh` out of the gitignored `tmp/` dir
  into `scripts/` (as `scripts/run_db_restore_drill.sh` and
  `scripts/run_railway_token_smoke_test.sh`) so they become committable
  — both are non-secret operator convenience wrappers (Keychain *service
  names* only, never secret values) proven working this session.
