# Session notes — 2026-07-26 — TF removal (done) + moto S3-mock test tier (done)

Supersedes the plan-only version of this doc from earlier the same date — both
tracks it described are now executed and committed.

## Done this session (committed)

1. **Auth-hardening test coverage gaps** (commit `978f785`) — see prior session
   notes for detail, carried over unchanged.
2. **TensorFlow removal** (commit `ee1839f`) — see prior session notes for
   detail, carried over unchanged.
3. **moto-based S3-mock test tier for `core/b2_utils.py`** (commit `2cbfb84`) —
   new this session, full detail below.

## moto S3-mock test tier — execution summary

Ran the full planning-TDD pipeline against the plan already produced in the
earlier session (code-planner output, reproduced in that plan-only doc
version): code-analyst (test spec matrix, 15 scenarios T01-T15) →
code-executioner (wrote `tests/core/test_b2_utils_moto.py` + supporting
conftest/pytest.ini/requirements-dev.txt changes) → `/post-production` (tier 3,
Opus review + two-stage Opus-authors/Fable5-executes adversarial pass) →
commit.

**`code-analyst` is a Skill, not a registered Agent type** in this
environment — `Agent(subagent_type:"code-analyst")` errored outright; invoked
via `Skill(skill:"code-analyst", ...)` instead, matching the same finding
already noted in the 2026-07-25 auth-hardening session notes. Saved as
project feedback memory this session (`feedback_code_analyst_is_skill.md`).

**Smoke-test gate (mandatory first step per the plan)**: initially BLOCKED —
this sandbox's squid egress proxy returned HTTP 403 for `pypi.org`,
`pip install --dry-run moto==4.2.14` failed with "from versions: none". Wrote
the test file correct-by-inspection first (module docstring flagged
unverified-by-local-execution), matching the existing `hypothesis` precedent
in `tests/core/test_utils_pure.py`.

**User then whitelisted `pypi.org` + `files.pythonhosted.org`** (asked
specifically which domains — same two needed for any `pip install`, no custom
`index-url` in this repo's `pip.conf`). `moto==4.2.14` installed successfully,
pulling in `cryptography`, `responses`, `xmltodict`, `werkzeug`, `jinja2`,
`markupsafe`, `cffi`, `pycparser`, `pyyaml` as new transitive deps (all
already-satisfied deps — `boto3`, `botocore`, `requests`, `python-dateutil` —
needed no changes). Saved as project feedback memory
(`feedback_pypi_proxy_allowlist.md`) — same allowlist pattern already used
once before for `pyright`/`ruff` in the 07-25 session.

**Real execution results**: all 15 new tests pass. Full suite: 184 passed, 1
skipped, 1 xfailed, 0 failed, 0 errors (was 172 passed + 12
`ModuleNotFoundError` collection errors before the allowlist widened).
`core/b2_utils.py` coverage: 36% → 100%.

**Two design-phase open questions, resolved by the real run**:
- `AWS_DEFAULT_REGION` (added defensively to root `conftest.py` to guard
  against a possible `NoRegionError`) turned out unnecessary — resource
  construction and `create_bucket` both succeed under moto with no region env
  var set at all. Left in place anyway (harmless).
- `delete_files([])` (T06): moto raises `ClientError` with code `MalformedXML`
  for an empty `Delete.Objects` list, matching real S3's documented behavior
  — confirmed caught correctly by `delete_files`'s `except ClientError`,
  returns `None`, no crash. Not the speculated `ParamValidationError` (which
  would NOT have been caught).

Both resolutions folded back into the test file's docstring/comments and into
`docs/security-remediation-plan.md`'s boto3/botocore/urllib3 landmine section
(dated note: moto confirms compatibility with the exact pinned triangle, but
doesn't unblock that landmine itself — B2 test bucket still needed since moto
simulates S3, not Backblaze B2's actual behavior).

## Post-production (tier 3)

Deterministic tools: ruff (0 new findings, 4 pre-existing `RUF100` untouched),
pyright (7 diagnostics, all confirmed stub-gap false positives — no
`boto3-stubs`/`botocore-stubs`/`django-stubs` installed, same class as an
already-documented gap), complexity (0 findings, no function crosses
threshold 15), full test suite (184 passed). sonar was unusable as signal —
fetch-mode only (no Docker daemon in this devcontainer), and the `webibex`
project has zero analyses ever run in SonarQube (not a clean-pass signal,
just no data). insecure-defaults: 0 findings, confirmed via grep that the two
new env vars never leak into any production code path. trufflehog/pip-licenses
user-declined. diff-cover attempted but blocked (not on the scoped pypi
allowlist, only `moto` was whitelisted this session).

Opus phase-5A review (checks [1]-[7] + request-adherence): 0
CRITICAL/MAJOR/MINOR, 9 INFO (confirmations + pre-existing/out-of-scope
context). All 6 plan requirements (R1-R6) independently verified COVERED
with cited evidence.

**Two-stage adversarial review** (Opus authors a targeted prompt, Fable5
executes the actual hunt against live code — per the CLAUDE.md pattern,
confirmed effective again this session): targeted the riskiest change, the
`no_network` autouse guard's new `moto_s3`-marker-gated bypass. Fable5
empirically verified 8 candidates with 13 throwaway live pytest reproductions.
**Verdict: GO-WITH-ADVISORY.** One real advisory confirmed: the `moto_s3`
marker alone (without the `moto_b2` fixture) disables the `boto3.resource`
guard with no misuse detection — today every use in the suite is legitimate
(grepped, confirmed), but nothing would stop a future test from misusing the
marker to silence the guard with a real, unmocked `boto3.resource()` call
underneath. Amplified by root `conftest.py`'s `os.environ.setdefault` on AWS
credentials, which would preserve real exported creds if a dev/CI shell
already has them set.

**Fixed same session**: added a `pytest_collection_modifyitems` hook to root
`conftest.py` that fails collection if a `moto_s3`-marked test doesn't also
request the `moto_b2` fixture (one allowlist exception for the guard test
that deliberately doesn't touch S3, `test_moto_s3_marker_does_not_bypass_requests_post_guard`).
Verified: the hook correctly rejects a throwaway misuse test, 184/184 still
pass after the fix, ruff/pyright findings unchanged (same 4 pre-existing, 0
new). Second advisory (pre-existing `boto3.client()` was never guarded by
`no_network` at all, only `.resource`) logged as out-of-scope context, nil
exposure today since production code only uses `.resource`.

## Commit

`2cbfb84` — "test: add moto-based S3-mock test tier for core/b2_utils.py".
6 files changed (`conftest.py`, `docs/security-remediation-plan.md`,
`pytest.ini`, `requirements-dev.txt`, `tests/conftest.py`,
`tests/core/test_b2_utils_moto.py` new), 389 insertions, 5 deletions.
`requirements.txt`/`core/b2_utils.py` confirmed byte-for-byte untouched (R6).

## Housekeeping left for the user

Three empty, untracked, harmless scratch files from this session's testing
need manual deletion (the `rm` hook correctly blocked both my own and
Fable5's attempts to clean them up): `tests/test_tmp_probe_guard_a.py`,
`tests/test_tmp_probe_guard_b.py`, `tests/test_zz_hook_selftest.py`.

## Environment note

User flagged mid-session: `~/.claude/` (rules, skills, project memory, the
feedback JSONL log) does not survive this container's reboot — `./docs/` in
each project repo is the durable reference going forward. Feedback memory was
still written this session per the usual process, but its persistence depends
on whatever sync mechanism (an `agents_writer` config-mirror job was seen
logged elsewhere) actually runs before a reboot — not confirmed either way
this session.

## Still open from before this session (untouched)

- boto3/botocore/urllib3 triangle — still blocked on a dedicated B2 test
  bucket (moto tier confirms compatibility with the pin but doesn't unblock
  the actual bump — see above).
- CI scaffold, region-detail cross-owner exposure, IDOR fix, ruff baseline
  config, logging pass, documentation gaps — all as previously tracked in
  `docs/security-remediation-plan.md`, untouched this session.
- `live_b2` marker registered as scaffold only (per plan R5), zero tests use
  it — the future real-B2 integration tier itself is not started, still
  blocked on the same B2 test bucket as the triangle bump above.
