# webibex Security Remediation Plan

> Reconstructed 2026-07-08 from `agents_writer` session notes and project memory — the
> original file (written 2026-07-07) was lost before being committed to this repo (it was
> untracked, never staged). This reconstruction captures the findings and decisions at
> the level of detail preserved in session notes; granular per-package upgrade steps
> should be re-verified against current `requirements.txt` before execution.

## Context

Professor asked to continue work on "the railway app" = webibex, a Django app deployed
on Railway.app (`wibex.up.railway.app`). Last commit before this audit: 2025-10-29
(~9 months stale). This is the **live system** the professor uses today — distinct from
the `ibex_stambecchi` HF Space (separate Gradio/PyTorch codebase), which has its own
fully-designed migration plan (`~/workspace/ibex_stambecchi/docs/migration-plan.md`) to
eventually retire webibex's entire stack (Django/Postgres/B2-chips/RunPod/TF).

**Decision (user, explicit)**: secure webibex now; treat the migration as a separate,
later decision — ibex_stambecchi has its own unresolved dev issues first.

## Findings

### Supply chain (first-ever `/supply-chain` audit against this repo)

Confirmed via pip-audit/OSV against exact pinned versions in `requirements.txt`:

| Package | Pinned | Severity | CVEs |
|---|---|---|---|
| `urllib3` | 1.26.20 | **CRITICAL** | 6 open CVEs (fixed in 2.5.0–2.7.0 range) |
| `Django` | 5.0.14 | MAJOR | 4 CVEs |
| `pillow` | 11.2.1 | MAJOR | 6 CVEs (hits the image-upload path) |
| `django-allauth` | 65.7.0 | MAJOR | 3 CVEs (auth) |
| `django-filer` | 3.1.1 | MINOR | 1 CVE (upload path) |
| `requests` / `lxml` | — | MINOR | routine |

JS-side findings (`node/`, Tailwind/clean-css build tooling) are confined to
build-time-only tooling — lower priority. **No CI exists in this repo at all**
(no `.gitlab-ci.yml`, no `.github/`, no `Dockerfile`, no `Makefile`).

### The boto3/botocore/urllib3 landmine

`boto3==1.26.0` / `botocore==1.29.165` / `urllib3==1.26.20` are **deliberately pinned
old** — a code comment says a newer boto3 adds an `x-amz-checksum-algorithm` header that
Backblaze B2 (the object storage backend) doesn't support. This is why urllib3 can't just
be bumped independently. The CVE audit confirms this pin is now evidence-backed risk, not
just staleness. Bumping this triangle needs a **dedicated B2 test bucket** to verify
compatibility before rollout — status as of 2026-07-07: unresolved (user pointed at the
existing `AWS_S3_*` / `AWS_STORAGE_BUCKET_NAME` env vars, but a dedicated separate test
bucket was not confirmed to exist).

### TensorFlow removal

`core/embedding_model/saved_model.pb` is committed to the repo (not gitignored) and
embeds producer version string `1.15.02` — confirmed TensorFlow 1.15, long past EOL.
Production never executes this path: real inference calls a RunPod serverless HTTP
endpoint (`embed_new_chip()` in `core/utils.py`, using `RUNPOD_ENDPOINT_ID` /
`RUNPOD_API_KEY`, confirmed present in Railway's variable list). TensorFlow is only
imported in two local-dev-only branches, gated by a **hardcoded** (not env-driven)
`ENDPOINT_LOCALLY = True` in `webibex/settings.py:141`, plus a standalone dev script
`core/test_model.py` with Bohlen's hardcoded local path.

**Recommendation**: remove the local-dev TF branches + `core/test_model.py` outright —
not bump-and-verify, not fold into the larger ibex_stambecchi migration. Production
doesn't use it, and TF1→TF2 compatibility for this exact model is already proven working
elsewhere (91.27% mAP, per `ibex_stambecchi/docs/migration-plan.md`). Low-risk, low-cost.

### OpenStreetMap ToS exposure (CR-2)

webibex hits `tile.openstreetmap.org/{z}/{x}/{y}.png` directly via Leaflet in 7 templates
(region create/update/read/delete/naming-error, location create, multi-location create).
Attribution is correct, but OSMF's Tile Usage Policy is explicit that this server is for
light/dev/evaluation use only — OSMF can throttle/block without warning.

**Fix**: swap to **MapTiler** (same `L.tileLayer()` call shape, different URL + API key,
across all 7 templates). Verified against professor-confirmed scale (20-50 non-concurrent
users): MapTiler's free tier is 5,000 map-sessions/month (session-based — panning within a
session is free), no non-commercial restriction. Compared against Stadia Maps (200,000
credits/month but explicitly non-commercial-use-only — disqualifying) and Thunderforest
(150,000 tile-requests/month). Even a generous estimate (~25k tile-equivalents/month)
leaves 6-10x headroom on all three; MapTiler chosen for the session-based accounting plus
no non-commercial restriction to verify against the university's status.

### Auto-match-or-new-ID requirement (CR-1) — mostly already built

Unlike the ibex_stambecchi HF Space (which genuinely lacks search at upload time),
webibex's `default_chip_compare_view` / `project_chip_compare_view` /
`geographic_chip_compare_view` already run full nearest-neighbor gallery search and show
top-5 matches with distances before any human decision; `created_animal_view` already
exists as the "no match" path. A `threshold_distance = 9.3` constant is defined but only
displayed, not used to auto-decide. Gap is UX polish (auto-suggest/pre-highlight using the
existing threshold), not new architecture.

## Remediation plan — 6-9 person-days

| Task | Estimate | Notes |
|---|---|---|
| dcg profile fix | done | already merged to main |
| Routine CVE bumps (Django, pillow, django-allauth, django-filer, requests, lxml, setuptools) | **done (2026-07-09)** | commit `480607b` — final versions in the table below |
| boto3/botocore/urllib3 triangle | ~1.5-2.5d | **blocked on B2 test bucket** — root cause + fix path documented in `tmp/memo-questions-for-lauren.md` item #5, not yet applied |
| TensorFlow removal | ~0.5d | **deferred to 2026-07-17** — user talking to original developer (Lauren) first; see below |
| JS build-tooling cleanup | **done (2026-07-09)** | exact pins applied, commit `480607b`; `npm audit fix` scope was a no-op (already clean) |
| Minimal CI scaffold | ~1-1.5d | **deferred to 2026-07-17**, same conversation — repo currently has zero CI |

**Status**: plan approved by user 2026-07-07. Batch narrowed twice on 2026-07-08 (CI
scaffold deferred pending a Lauren conversation; TensorFlow removal already deferred) to:
Python CVE bumps (Task 2) + JS pin cleanup (Task 3) + `init_prod_requirements.txt` sync
(Task 4). **Executed and committed 2026-07-09** (`480607b`), verified **locally**
(OSV-clean for all 7 bumped packages, re-confirmed via socket.dev against the final
committed versions; `manage.py check`/`migrate`/`test` pass; staged Django 5.1→5.2
deprecation check completed with zero new deprecations; allauth and filer-upload flows
exercised end-to-end via Django's test client) instead of via a CI gate. See
`docs/session-notes-2026-07-09-webibex-security-remediation.md` for the full execution
trace. CI scaffold and TensorFlow removal remain deferred to 2026-07-17.

**Open blocker**: B2 test-bucket provisioning — needed before the boto3/botocore/urllib3
triangle task can start.

**TensorFlow removal deferred**: user will talk to the original developer (Lauren,
`laubohlen/webibex`) on 2026-07-17 about both this task's scope and the separate
RunPod-access request (needed for the TF2 embedding-model swap, see session notes
2026-07-08). TF-removal excluded from the current batch until after that conversation.

### Expanded CVE findings (2026-07-08 `/supply-chain` audit, OSV + socket.dev)

Cross-checked with real OSV batch API + socket.dev behavioral scan data (not simulated).
Found more than the 2026-07-07 snapshot — newly disclosed in the intervening day. **Final
column added 2026-07-09** — the actual version committed in `480607b`, live-verified
against PyPI at execution time (higher than some floors below, where the floor version
still left a CVE open):

| Package | Pinned | Fix target (2026-07-08 estimate) | Final (committed) | Severity | Notes |
|---|---|---|---|---|---|
| `Django` | 5.0.14 | 5.2 LTS (5.0.x has **no** fixed version) | **5.2.15** | **CRITICAL** (socket.dev reclass; OSV CVSS 7.5) | SQL injection via `_connector` kwarg (GHSA-frmv-pr5f-9mcr) + 3 more; 5.2.8 still had 39 open advisories, needed 5.2.15 |
| `pillow` | 11.2.1 | likely 12.x, confirm before bump | **12.2.0** | MAJOR | confirmed — 11.3.0 still had 7 open CVEs, 12.2.0 needed |
| `django-allauth` | 65.7.0 | ≥65.14.1 | **65.14.1** | MAJOR | 6 advisories, 2 open-redirect + 3 IdP identity issues |
| `lxml` | 5.3.2 | ≥6.1.0 | **6.1.0** | MAJOR | XXE via default `iterparse()` config (CVSS 7.5) |
| `requests` | 2.32.3 | ≥2.32.4 (one patch) | **2.33.0** | MAJOR | 2.32.4 still left CVE-2026-25645 (temp-file reuse) open, needed 2.33.0 |
| `setuptools` | 78.1.0 | ≥78.1.1 (one patch) | **78.1.1** | MAJOR | path traversal → arbitrary file write (CVSS 8.9) |
| `django-filer` | 3.1.1 | ≥3.3.0 | **3.3.0** | MAJOR | unrestricted dangerous-type file upload |
| `urllib3` | 1.26.20 | blocked — see boto3 triangle above | *(unchanged, out of scope)* | CRITICAL | 10 open advisories; fix path documented in `tmp/memo-questions-for-lauren.md` item #5 |
| `pip`, `idna`, `sqlparse` | — | 25.3+/3.15/0.5.4 | *(unchanged, transitive, out of scope)* | MINOR | build-time only |

`django-polymorphic==3.1.0` (not in the original table) was confirmed compatible with
Django 5.2 via a real `pip install` — no bump needed.

Full report in session transcript 2026-07-08; socket.dev confirmed **no malware/typosquat/
install-script alerts** across all 41 Python + 2 JS packages at both the pre-bump scan
(2026-07-08) and re-verified against the final committed versions (2026-07-09) —
behavioral scan clean throughout.

## Explicitly deferred (separate track)

Age/zone/muzzle recognition pipeline redesign for `ibex_stambecchi` was researched in
depth in the same session but explicitly deferred by the user ("first CR it's the
supply-chain webibex update"). See `agents_writer` project memory
(`project_webibex_rework.md`) and session notes
(`docs/session-notes-2026-07-07-webibex-ibex-security-plan.md` in `agents_writer`) for
the full research trail (MegaDescriptor/wildlife-tools backbone comparison, horn-tip wear
literature, muzzle-recognition precedent, near-duplicate augmentation risk).

## TODO — Railway deployment hardening (not started, 2026-07-23)

Railway currently builds this app via its own auto-detected Nixpacks builder — no
`Dockerfile`/`railway.json`/`nixpacks.toml` in this repo (confirmed
`docs/session-notes-2026-07-08-webibex-security-remediation.md:61-71`). That means there
is currently no way to apply the same hardened-base-image treatment already done for
`samgis-be` (pinned DHI/hardened base, VEX-exception trust matrix, Scout/Trivy-verified).

- Check Railway's docs for whether a custom-`Dockerfile` build path is supported
  (would be the prerequisite for adopting a hardened base image here, same pattern as
  the `dhi.io/tensorflow-serving:2` candidate already scoped for the RunPod serving side
  in `docs/tf1-to-tf2-migration-plan.md:133-168`).
- At minimum, bump the Python pin: `runtime.txt` is currently `python-3.12.5` — check
  current supported versions on both Railway and RunPod (the three
  `tmp/inference/*/builder/requirements.txt` trees are Python-pinned independently, see
  `docs/session-notes-2026-07-21-svglib-reportlab-dependency-separation.md:41-45`) and
  bring them in line.
- Trigger: next security-remediation batch, or whenever the `samgis-be` hardening
  pattern is revisited for this project.
- **Dev/prod dependency separation** (found 2026-07-23, adding pytest coverage):
  Railway's default Nixpacks builder only installs `requirements.txt` — there's no
  `railway.toml`/`nixpacks.toml` here to declare a build-only or optional dependency
  group, and plain `requirements.txt` has no extras syntax to mark packages
  dev-only. Test deps (`pytest`, `pytest-django`, `pytest-cov`) were moved to a
  new `requirements-dev.txt` instead, installed alongside for local dev
  (`pip install -r requirements.txt -r requirements-dev.txt`) but never shipped to
  Railway. Study whether a `railway.toml`/`nixpacks.toml` (or migrating to
  `pyproject.toml` with a `[dependency-groups]`/`[project.optional-dependencies]`
  split, `uv`-native) would let Railway itself skip a declared dev group instead of
  relying on file-split-by-convention.

## TODO — stale `staticfiles/` vs. current Django version (found 2026-07-23)

Re-running `manage.py collectstatic` locally (during the landmark-CSS-fix CR,
`docs/changes/2026-07-23-fix-landmark-image-scale-mismatch.md`) showed real content
drift in `staticfiles/admin/*` and `staticfiles/filer/*` — Django 5.2.15's own admin
CSS/JS (the currently-pinned version, per `480607b`) differs from whatever version
last generated the committed `staticfiles/` tree. That commit bumped `requirements.txt`
but apparently never re-ran `collectstatic`, so the tracked, served copy of Django's
own admin assets is stale relative to the app's actual dependency versions.

- Run a full `manage.py collectstatic` and review the diff (expect `staticfiles/admin/*`,
  `staticfiles/filer/*`, possibly others) as its own dedicated commit — do not bundle
  with an unrelated CR (already deliberately kept out of the 2026-07-23 CSS fix).
- Confirm no visual/functional regressions in `/webibex/` (Django admin) and
  `/filer/` before committing — Django 5.2's admin CSS has real UI changes vs. earlier
  versions (dark-mode variables, etc., seen in the diff).
- Trigger: next dependency bump that touches Django, `django-filer`, or
  `django-allauth`; or the next full security-remediation batch.
