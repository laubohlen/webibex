# Session notes — 2026-07-23 — TF2210 e2e verification, landmark CSS fix, dep-bump sweep

Continuation session. Committed `9015b46` on top of `cd7b5b0`
(TF1→TF2 triplet-reid consolidation) and `480607b` (security CVE bumps).

## Docker image audit (start of session)

Reviewed a 10-image, ~29GB docker inventory. Verdict: ~20GB safe to
reclaim (`triplet-reid-verify:*`, `tf218-nets-deps:local`, both
`tensorflow/tensorflow` base images, `r-base`/`rocker` — the latter two
belong to a different, unrelated project's Dockerfile comment, not
webibex). `laubohlen/ibex_embedding_cpu:v0.01` (production) and
`dhi.io/tensorflow-serving:2` (open serving-hardening candidate,
documented in `docs/tf1-to-tf2-migration-plan.md`) kept.

## TF2210 default-version decision + full e2e verification

**Decision: TF 2.21.0 (`tf2210`) is now the default** for
`training/triplet-reid/dockerfiles/` (committed this session). Rationale:
production runs TF1, so 2.18.1-vs-2.21.0 is immaterial to production risk
either way — pick the version with more maintenance headroom.

E2E test methodology (all steps run manually, not automated):
1. `tmp/inference/host_runbook/e2e_tf2210_manual_test.sh` (gitignored
   scratch, written this session) — exports a persisted (not `mktemp`)
   SavedModel via the `tf2210` image, builds a throwaway RunPod-inference
   image from it, runs the real `handler.py` in RunPod SDK local-test
   mode. Confirmed working: 128-dim embedding, matches the byte-level
   Phase 4b/4c gate already documented as passing.
2. Ran the actual Django app locally (`ENVIRONMENT=e2e-test` — deliberately
   not `"development"`, to dodge a `debug_toolbar` crash — see
   `[[project_webibex_local_dev_gaps]]` memory), with `ENDPOINT_LOCALLY`
   temporarily flipped and later reverted, driving the real upload →
   landmark → crop → embed flow through a browser via `bin/dev-tunnel`.
3. Local RunPod container reachability from inside the devcontainer
   required `--network container:<devcontainer-id>` (network-namespace
   sharing) — `host.docker.internal`/Docker Desktop gateway IP were both
   unreachable despite the server confirmed listening. Full writeup in
   `[[project_devcontainer_network_sandbox]]` memory.
4. Repeated successfully across several real images — multiple
   `Embedding` rows created, all 128-dim, via the real HTTP
   `endpoint_inference()` code path (not the broken local-model branch).

All temp code changes (`webibex/settings.py`, `webibex/urls.py`,
`core/utils.py`'s `endpoint_inference()` URL override,
`core/embedding_model/` swap) were reverted before committing — verified
via `git diff --cached` showing none of those files present in the final
diff.

## Landmark image CSS scaling bug (found + fixed)

Found while debugging why cropped chips looked wrong during the e2e test.
`.imageToLandmark` (`static/css/tailwind.css`) had no width rule, so
uploaded images narrower than `settings.LANDMARK_IMAGE_WIDTH` (1600px)
rendered at natural size in the browser, but `core/utils.py:
scale_coordinate()` still divided by 1600 — landmarks shrunk toward the
top-left corner (one case landed entirely in blank sky). Confirmed via
annotated before/after screenshots (`cv2.circle` at the stored
coordinates). Fixed with `width: 100%; height: auto;`, rebuilt
`static/css/style(.min).css` (gitignored build outputs) and
`staticfiles/css/*` (tracked, actually served copies), verified via
`manage.py collectstatic` + WhiteNoise server restart (see
`[[project_webibex_local_dev_gaps]]` — WhiteNoise caches at startup when
`DEBUG=False`).

Pre-existing bug, confirmed unrelated to TF version (reproduces
identically against the old `laubohlen/ibex_embedding_cpu:v0.01` image —
the whole code path runs before any embedding call).

## .gitignore Dockerfile-anchoring bug (found + fixed)

`.gitignore` had a bare `Dockerfile` entry (Fly.io-era leftover, alongside
`.dockerignore`/`fly.toml`), which silently blocked `git add` from ever
tracking the three new `training/triplet-reid/dockerfiles/*/Dockerfile`
files — `verify_gate.sh` staged fine, the Dockerfiles it depends on didn't.
Caught by `/post-production` Phase 0. Fixed: `Dockerfile` → `/Dockerfile`
(root-anchored).

## svglib/reportlab dependency fix — verified this session

Egress opened (temp whitelist), `uv pip install -r requirements.txt`
resolved cleanly (41 packages, Django 5.2.15 at the time). Correction to
the original rationale: `svglib`/`reportlab` aren't actually removable —
`easy-thumbnails` depends on them non-optionally; unpinning let the
resolver pick working versions (`svglib==2.0.2`, `reportlab==5.0.0`)
instead of the broken originals (`1.5.1`/`4.3.1`, no mutual
wheel-compatible combination). `core/test_model.py` deliberately not run
as originally planned — confirmed not a real test (no assertions,
hardcoded path to the original developer's machine).

## /post-production run (tier 4) + Fable 5 review

Tier 4 (new Dockerfiles trigger the security-sensitive floor).
`insecure-defaults` and `/security-review`: zero findings.
`sonar`: skipped (project registered, zero prior analyses, scan step
needs Docker unavailable here). `markdownlint`: 241 findings, all
default-rule noise — no `.markdownlint` config exists in this repo, even
untouched `README.md` fails the same rules.

`/judges` has no `--model` override and hardcodes its inline judge to
Opus ("Judges stay Opus" per CLAUDE.md) — requested Fable 5 instead, ran
it as a standalone independent review agent outside `/judges`'
orchestration. Caught two real issues before commit:
- `h5py>=3.14.0` floor pin was only empirically verified on the `tf2210`
  base image, wrongly applied to `tf2180`/`tf2181` (2.18.0 base) too —
  reverted those two to unpinned (matching the original, already
  gate-passed state).
- CR doc self-contradiction: "Status: committed" next to "nothing has
  been committed yet" in the same file (written before the commit
  actually happened) — fixed to "staged, pending commit."

Committed as `9015b46`.

## Dependency bump sweep — stashed, not applied

OSV batch query across all 36 pinned packages in `requirements.txt`.
Found 6 safe patch/minor bumps (Django 5.2.15→5.2.16, idna 3.10→3.15,
pillow 12.2.0→12.3.0, pip 24.2→26.1.2, setuptools 78.1.1→83.0.0,
sqlparse 0.5.3→0.5.4) — all verified resolving cleanly via
`uv pip install -r requirements.txt`. Two flagged but not touched:
- `pillow_heif` (aka `pillow-heif`/`pi-heif` — same PyPI package,
  underscore/hyphen normalized): pinned `0.22.0`, latest `1.5.0`, HIGH
  severity integer-overflow fix at `1.3.0`. Major version jump, needs
  real changelog/breaking-change review before bumping — not done this
  session.
- `urllib3`: already documented as blocked
  (`docs/security-remediation-plan.md` — boto3/botocore/urllib3
  triangle, needs B2 test-bucket verification for
  `x-amz-checksum-algorithm` compatibility). Not re-litigated.

**Stashed** (`git stash@{0}`, message: "deps: Django 5.2.16, idna 3.15,
pillow 12.3.0, pip 26.1.2, setuptools 83.0.0, sqlparse 0.5.4
(OSV-verified, pending pytest coverage first)") rather than committed —
user paused to prioritize test coverage first, on the reasoning that
further bumps shouldn't rely on manual-smoke-test-only verification like
prior rounds did.

## Test coverage survey (quick pass, not implemented)

Confirmed genuinely zero pytest coverage for the Django app: `core/tests.py`
and `simple_landmarks/tests.py` are unmodified `startapp` boilerplate;
`db_management/test.py` is a one-off data-migration script misnamed
`test.py`, not a test. No `pytest`/`pytest-django` in `requirements.txt`,
no `pyproject.toml`/`pytest.ini`/`conftest.py` for the Django app (only
`training/triplet-reid/tests/` has real tests, separate tree).

Surface area: 2 local apps (`core`, `simple_landmarks`), `core/views.py`
895 lines / 28 view functions, `core/utils.py` 603 lines of mostly-pure
logic (best first-pass unit-test target — this is where the AVIF and
CSS-scaling bugs were both found this session), ~6 models across the two
apps, 33 URL patterns. Not scoped further or implemented — next-session
work.

## Untracked leftovers at session end

- `core/embedding_model/fingerprint.pb` — leftover from the tf2210
  export swap, confirmed not needed (inconsistent with the reverted
  original model files sitting next to it).
- `staticfiles/admin/css/unusable_password_field.css` +
  `staticfiles/admin/js/unusable_password_field.js` — real Django 5.2
  admin files pulled in by `collectstatic`, but orphaned without the rest
  of that sync (deliberately not done this session, tracked as a TODO in
  `docs/security-remediation-plan.md` — stale `staticfiles/admin|filer`
  vs. current Django version).
- `.claude/settings.local.json` — never resolved whether to `.gitignore`
  it; user is fine leaving it permanently untracked.

Update: `fingerprint.pb` and both `unusable_password_field.*` files have
since been deleted by the user. Only `.claude/settings.local.json`
remains untracked at actual session end.
