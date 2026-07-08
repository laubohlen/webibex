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
| Routine CVE bumps (Django, pillow, django-allauth, django-filer, requests, lxml) | ~2.75d | standard version bumps + regression check |
| boto3/botocore/urllib3 triangle | ~1.5-2.5d | **blocked on B2 test bucket** — verify `x-amz-checksum-algorithm` compatibility before bump |
| TensorFlow removal | ~0.5d | delete local-dev branches + `core/test_model.py`; leave RunPod path untouched |
| JS build-tooling cleanup | ~0.25d | `npm audit fix` scope, build-time only |
| Minimal CI scaffold | ~1-1.5d | repo currently has zero CI |

**Status**: plan approved by user 2026-07-07, not yet executed.

**Open blocker**: B2 test-bucket provisioning — needed before the boto3/botocore/urllib3
triangle task can start.

## Explicitly deferred (separate track)

Age/zone/muzzle recognition pipeline redesign for `ibex_stambecchi` was researched in
depth in the same session but explicitly deferred by the user ("first CR it's the
supply-chain webibex update"). See `agents_writer` project memory
(`project_webibex_rework.md`) and session notes
(`docs/session-notes-2026-07-07-webibex-ibex-security-plan.md` in `agents_writer`) for
the full research trail (MegaDescriptor/wildlife-tools backbone comparison, horn-tip wear
literature, muzzle-recognition precedent, near-duplicate augmentation risk).
