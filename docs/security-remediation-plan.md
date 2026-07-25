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

## TODO — latent bugs found during the B1/B3/B4 bug-fix round review (found 2026-07-24)

Surfaced by the Opus + Fable 5 independent reviews of `core/utils.py`/`core/models.py`/
`core/middleware.py` while fixing B1/B3/B4/Region-twin/middleware bugs. All three are
pre-existing, dormant with the app's current data shapes, unchanged by that fix — not
regressions, not urgent, but worth tracking rather than losing.

- **`get_task_request_origin` is dead code** (`core/utils.py:530`) — confirmed zero
  production callers anywhere in the codebase (only referenced by its own unit tests).
  Either wire it up to whatever redirect flow it was originally meant for, or remove it
  (along with its 2 leftover `print()` calls and broad `except Exception`).
- **Middleware still has one uncaught exception path**: `RedirectToUserFolderMiddleware`
  now catches `Http404` (fixed this round), but `get_object_or_404` also re-raises
  `Folder.MultipleObjectsReturned` as-is if two `&lt;user&gt;_files` folders exist for the
  same user (e.g. under different parents, or duplicate NULL-parent rows on Postgres).
  Was equally uncaught before this fix — not new, just never fully closed.
- **`generate_animal_id_code` regex family of bugs** (`core/utils.py:200`, all
  pre-existing, dormant with current `PNGP24`-style 2-digit prefixes):
  - `re.findall(r"\d{3}", id_code)[0]` picks the *first* 3-digit run, not necessarily
    the sequence number — an id_code like `PNGP2024_007` would incorrectly extract `202`
    instead of `007` if a prefix ever grows to 3+ digits.
  - No prefix scoping: the max is computed across ALL `id_code`s containing `"_"`,
    regardless of prefix — an existing `ALPS23_099` would make the next `PNGP24` animal
    `PNGP24_100` instead of `PNGP24_001`.
  - `>999` rollover: `f"{1000:03}"` → `PNGP24_1000` (11 chars) exceeds the model's
    `id_code` `max_length=10` — would fail to save on Postgres once any prefix crosses
    999 generated animals.
- Trigger: next full test-coverage-expansion pass on `core/utils.py` (already planned),
  or if any of these actually reproduce in production data.

## TODO — add debug/observability logging at high-risk boundaries (found 2026-07-24)

`core/utils.py` uses `print()` pervasively (20+ call sites, including the 2 left
untouched in `get_task_request_origin` during the B1 fix, deliberately not converted
mid-bugfix since there's zero logging infrastructure anywhere in `core/` — see
`docs/session-notes-*` for that scoping decision) instead of the `logging` module
(`python.md` § Logging: avoid `print()` for diagnostics, use a logger). Before/alongside
the next coverage-expansion pass, introduce real logging — prioritize:

- **Local-dev-specific branches**: everywhere `ENVIRONMENT`/`DEBUG`/`POSTGRES_LOCALLY`/
  `ENDPOINT_LOCALLY`/`MODEL_IS_LOCAL`-style settings gate different code paths
  (`webibex/settings.py`, `core/utils.py`'s `embed_new_chip()` branch selection) — these
  are exactly where dev-vs-prod behavior silently diverges and where a wrong branch is
  hardest to notice without a log line stating which path was taken.
- **I/O operations**: image load/decode (`load_image()`, the AVIF/None-decode gap — B5,
  still deferred), file reads/writes (`process_horn_chip()`, chip file generation),
  `manage.py collectstatic`-adjacent static-asset drift (see the staticfiles TODO above).
- **User interactions**: image upload → landmark → crop → embed flow (the multi-step
  pipeline through `core/signals.py`, `core/views.py`'s upload handlers) — currently the
  highest-value untested/unlogged surface per the coverage survey (`core/views.py` at
  18% coverage, `core/signals.py` at 44%).
- **External API calls**: RunPod `endpoint_inference()` (already has some error handling
  but no structured logging of request/response shape on failure), B2/boto3 calls in
  `core/b2_utils.py` (currently only 36% covered, real S3 logic largely untested).
- Set up a proper logger (`logging.getLogger(__name__)` per `python.md` — no `structlog`
  dependency currently in this project, so stdlib `logging` is the right default per the
  Code Choice Hierarchy) once, then migrate `print()` call sites incrementally as each
  area gets touched — not a single big-bang rewrite of the whole file.
- Trigger: next time any of these areas gets touched for a bug fix or coverage
  expansion (natural opportunity to add logging alongside, rather than a standalone
  logging-only PR).

## TODO — region dropdown empty for users who didn't create the region (found 2026-07-24, RESOLVED 2026-07-24)

Found during a manual e2e walkthrough (staticfiles/admin+filer refresh CR verification,
unrelated to that CR — reproduced identically before and after it). Confirmed app-wide,
not page-specific: Dashboard > Identification > "location" column > "set", clicking an
existing location's region link, AND the equivalent "location" column in the Animal
Dashboard (`templates/core/animal_images.html:23`) all use the same
`{% post_task_redirect 'locate-image' oid=i.id %}` link, routing to the single shared
`create_loaction` view (`webibex/urls.py:70-73`, note the pre-existing typo in the view
name), which builds the region dropdown from:

```python
region_qs = Region.objects.filter(owner=request.user)  # core/views.py:665
```

Confirmed via direct DB query: both existing `Region` rows (`smoketest-region`,
`regione2`) have `owner_id=2` (`chiptestuser`). Logging in as a *different* user (e.g.
a freshly created superuser) legitimately gets an empty queryset — not a rendering bug,
not related to the staticfiles refresh, the filter is working exactly as written.

**Confirmed NOT a regression** (git blame, checked 2026-07-24): both `owner=` filters
(`core/views.py:665`, `core/utils.py:382`) date to `46a66a8f`/`a6724250`
(2025-02-17/2025-02-27, original developer) — over a year old, predates every commit
touched this session. The "it worked before" observation is explained by session
context, not a code change: a prior manual test almost certainly used the actual
region-owning account (`chiptestuser`), while this session's walkthrough used a
freshly created `e2e_admin` superuser with zero owned regions — same long-standing
filter, different logged-in user, different visible result.

**Decided: shared-by-design.** Regions are shared/global, not private to whoever
created them — this matches the two other read-path call sites already in the
codebase (`region_overview` at `core/views.py:613`, `save_image_location`'s region
lookup at `core/views.py:628`) which never scoped by `owner` in the first place. The
two outlier filters were the anomaly, not the rest of the app; aligning them removes
the inconsistency instead of introducing a new policy.

Fix applied (2026-07-24): both outlier read-path filters now match the rest of the
codebase —
- `create_loaction`'s `region_qs` (`core/views.py:665`): `Region.objects.filter(owner=request.user)` → `Region.objects.all()`.
- `multi_task_url()`'s `"locate"` branch `region_qs` (`core/utils.py:382`): `Region.objects.filter(owner=user)` → `Region.objects.all()`.

The EDIT-permission `owner=request.user` scoping (`save_region`'s update path at
`core/views.py:534`, `delete_region` at `core/views.py:593`, `update_region` at
`core/views.py:604-606`) is unrelated to region *visibility* and was deliberately left
unchanged — a region being visible to everyone does not imply everyone can edit it.
A regression-guard test (`test_region_edit_permission_unchanged_for_non_owner` in
`core/tests/test_views_smoke.py`) now proves this scoping still rejects a non-owner on
all three paths.

New tests added: `core/tests/test_utils_db.py`
(`test_multi_task_url_locate_branch_shows_region_owned_by_other_user`,
`test_multi_task_url_locate_branch_shows_all_regions_not_just_cross_owner`,
`test_multi_task_url_locate_branch_shows_orphaned_region_with_no_owner`) and
`core/tests/test_views_smoke.py`
(`test_create_loaction_view_shows_region_owned_by_other_user`,
`test_create_loaction_view_shows_all_regions_not_just_cross_owner`,
`test_create_loaction_view_shows_orphaned_region_with_no_owner`,
`test_create_loaction_region_visibility_same_for_existing_vs_new_location`,
`test_create_loaction_view_empty_region_list_returns_200`,
`test_region_edit_permission_unchanged_for_non_owner`) — each proves a cross-owner or
orphaned (`owner=None`) region is a genuine member of the returned queryset, not just
"non-empty".

- Note: `save_region`'s global (not per-owner) duplicate-name check vs. the model's
  per-owner `UniqueConstraint` remains a separate, already-flagged inconsistency —
  unrelated to this fix, needs its own future decision.

## TODO — "Delete" tool in the multi-image Tools menu is unimplemented and crashes (found 2026-07-24)

Found during the same manual e2e walkthrough (unrelated to the staticfiles refresh CR).
Reproducible: Identification (or Animal) dashboard > select row(s) > Tools menu >
Delete > `TypeError: cannot unpack non-iterable NoneType object` at `/unidentified/`
(500).

Root cause traced via the server traceback (`core/views.py:874`,
`template, task_context = utils.multi_task_url(task, image=image, user=request.user)`)
to `core/utils.py:372-399`, `multi_task_url()`:

```python
elif tool == "delete":
    print("Deleting images")
    # no deletion logic, no return -- falls through to implicit `return None`
else:
    print("No valid tool selected.")
    # same problem -- any unrecognized tool value hits this too
```

Every other branch (`view`, `locate`, `landmark`) correctly `return (template, context)`;
`delete` only prints a message and does nothing else. **This is not a working feature
with a crash bug — it's an unfinished stub with zero actual deletion logic** (no
`IbexImage`/`IbexChip`/`Embedding` row deletion, no file/storage cleanup, no
confirmation UX), which additionally crashes instead of silently no-op'ing.

**Open design question, not yet decided** (why not fixed this session — real behavior
needs a deliberate decision, not a guess): what should Delete actually do?
- Hard-delete the `IbexImage` row only, or cascade to its `IbexChip`/`Embedding` rows
  and the underlying stored file (B2/local media)?
- Any confirmation step before a destructive multi-select delete?
- Soft-delete/archive instead of hard delete?

Until that's decided, the crash itself is arguably the safer failure mode (loud 500,
no data touched) versus a rushed implementation that silently does the wrong kind of
delete.

- Trigger: next session the professor/user wants the Delete tool working, or when
  scoping a broader `core/views.py` coverage/cleanup pass (per the existing
  coverage-expansion TODO above — `core/views.py` is at 18% coverage).

## TODO — auth/session hardening settings missing (found 2026-07-24, RESOLVED 2026-07-25)

Raised during a session discussion on login-system security (not a deep audit — a
quick read of `webibex/settings.py` in full, already done this session for unrelated
reasons, surfaced this gap). None of the following are set anywhere in `settings.py`,
including inside the existing `if ENVIRONMENT == "production" or POSTGRES_LOCALLY ==
True:` conditional blocks (lines ~137, ~194, ~235) that already gate other
production-only config (DB, `STORAGES`, email backend):

- `SESSION_COOKIE_SECURE` — session cookie sent over plain HTTP today even in
  production (Railway serves HTTPS, so this is a real, closable gap, not theoretical).
- `CSRF_COOKIE_SECURE` — same exposure for the CSRF cookie.
- `SECURE_SSL_REDIRECT` — no server-side enforcement that requests upgrade to HTTPS.
- `SECURE_HSTS_SECONDS` (+ `SECURE_HSTS_INCLUDE_SUBDOMAINS`/`SECURE_HSTS_PRELOAD` if
  applicable) — no HSTS header, so a user's browser won't remember to prefer HTTPS.

**Must be gated the same way existing production-only settings already are** (the
`ENVIRONMENT == "production"` conditional) — setting `SESSION_COOKIE_SECURE`/
`CSRF_COOKIE_SECURE`/`SECURE_SSL_REDIRECT` unconditionally would break local dev over
plain `http://127.0.0.1:8000`, confirmed this session while running the app locally
for other CRs' manual walkthroughs.

Context for urgency: current threat model is a small (~20-50 non-concurrent users,
per this doc's own Context section), trusted, internal research-tool user base — not
a public-facing app. These are still cheap, standard hardening with no real downside
once correctly gated to production, closing a real (if lower-probability-given-scale)
exposure.

- Trigger: next full security-remediation batch, or before any planned increase in
  user base / exposure (e.g. if the app is ever opened beyond the current trusted
  research group).

Fix applied (2026-07-25): `webibex/settings.py` now sets all 4 requested settings
(plus `SECURE_PROXY_SSL_HEADER`, needed alongside `SECURE_SSL_REDIRECT` — Railway
terminates TLS at its edge and forwards plain HTTP, so without it the redirect would
loop) inside a new `if ENVIRONMENT == "production" or POSTGRES_LOCALLY == True:` block
(`webibex/settings.py:51-62`), placed right after `CSRF_TRUSTED_ORIGINS` and before
`AUTHENTICATION_BACKENDS` — the same gate condition already used by the DB/`STORAGES`/
email blocks (R2 rationale: excluding dev/test avoids the plain
`http://127.0.0.1:8000` breakage flagged above):

- `SESSION_COOKIE_SECURE = True`
- `CSRF_COOKIE_SECURE = True`
- `SECURE_SSL_REDIRECT = True`
- `SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")`
- `SECURE_HSTS_SECONDS = 3600` (conservative starting value — ratchet to `31536000`
  after confirming stability; `SECURE_HSTS_INCLUDE_SUBDOMAINS`/`SECURE_HSTS_PRELOAD`
  deliberately omitted, no subdomains in `ALLOWED_HOSTS` and preload submission isn't
  possible for a `railway.app` subdomain anyway).

New tests added: `core/tests/test_settings_security_hardening.py` —
`test_hardening_settings_present_with_correct_values_under_environment_production`
(T01, plus T04's `SECURE_PROXY_SSL_HEADER`/`SECURE_SSL_REDIRECT` coupling check),
`test_hardening_settings_absent_under_ambient_environment_test` (T02, the
no-regression keystone — asserts none of the 5 settings exist under the ambient
pytest `ENVIRONMENT=test`), `test_hardening_settings_absent_under_environment_development`
(T03, proves local dev over plain HTTP is unaffected). T05
(`POSTGRES_LOCALLY == True` alternate path) and T06 (`manage.py check --deploy`)
were not implemented — T05 is dead in practice (`POSTGRES_LOCALLY` is a hardcoded
dev-only toggle, never `True` in prod/CI) and T06 is a manual verification step, not
a scripted test.

Note: implementing this surfaced a real pre-existing ordering bug — `POSTGRES_LOCALLY
= False` was defined at (old) line 136, *after* the `CSRF_TRUSTED_ORIGINS`/
`AUTHENTICATION_BACKENDS` span where the new hardening block needed to reference it.
Moved to immediately after `ENVIRONMENT = env("ENVIRONMENT", default="production")`
(`webibex/settings.py:23`) as part of this fix — not part of the original TODO scope,
but a genuine `NameError`-on-import risk for any code inserted between its old
definition and its first use, closed while touching this area.

## TODO — evaluate `allauth.mfa` (to evaluate, not decided)

Raised in the same session discussion — `django-allauth` (already a dependency,
patched to 65.14.1 per this doc's CVE remediation section) ships an optional
`allauth.mfa` app for TOTP/WebAuthn multi-factor auth, not currently in
`INSTALLED_APPS`. **Explicitly not decided or scoped yet** — needs evaluation before
any implementation:

- Does the professor/user base actually want MFA, or is it disproportionate friction
  for a ~20-50 non-concurrent user trusted research group?
- If yes: TOTP (authenticator app) vs. WebAuthn (hardware key/platform biometric) —
  different UX and support burden.
- Interacts with the auth-hardening TODO above (both touch the login/session surface)
  but is a separate, larger decision — don't conflate scoping the two.

- Trigger: professor/user explicitly requests stronger auth, or a future
  security-remediation batch revisits the login system in depth.

## TODO — `endpoint_inference()` has dead/unused parameters (found 2026-07-24)

Found while reviewing the `INFERENCE_ENDPOINT_URL_OVERRIDE` addition to
`core/utils.py:endpoint_inference()`. The function signature accepts
`endpoint_id`/`endpoint_api_key` as parameters (def-time defaults via
`env("RUNPOD_ENDPOINT_ID")`/`env("RUNPOD_API_KEY")`), but the function body
never references either parameter — it re-reads
`env("RUNPOD_ENDPOINT_ID")`/`env("RUNPOD_API_KEY")` directly inline instead,
duplicating the same lookups. Pre-existing, predates this session (not
introduced by the override change, confirmed by the Opus post-production
review of that CR) — the only caller (`embed_new_chip()`) never passes these
args, so it hasn't manifested as an observed bug, but it's misleading dead
code: anyone calling `endpoint_inference(img, endpoint_id="x")` expecting
that to take effect would be silently ignored.

**Decided direction (user, 2026-07-24): remove the two parameters** — they
add no value, the body's direct `env()` re-reads already are the real
behavior. Keep the def-time-vs-call-time reasoning already documented inline
for `INFERENCE_ENDPOINT_URL_OVERRIDE` (that one's read at call time
deliberately; `RUNPOD_ENDPOINT_ID`/`RUNPOD_API_KEY` can stay as direct
`env()` calls in the body, matching what the code already actually does).

- Trigger: next time `core/utils.py`'s RunPod integration is touched, or a
  dedicated small cleanup pass.

## TODO — documentation gaps: docstrings/comments, user guide, developer docs (found 2026-07-24)

Raised in a session discussion, not from a specific bug — this codebase has
very little documentation beyond inline `print()`-style narration and this
security-remediation doc itself. Three distinct gaps, each needing its own
scoping:

1. **Docstrings/comments on existing code** — most functions in
   `core/utils.py`/`core/views.py` have no docstrings; the "why" behind
   non-obvious decisions is scattered across session notes and this doc
   rather than living next to the code (see `python.md`'s Decision
   Documentation rule, already applied ad hoc for the
   `INFERENCE_ENDPOINT_URL_OVERRIDE` addition this session — should become
   the norm, not the exception).
2. **End-user documentation** — a guided in-app tour for the actual users
   (professor/research team) using the identification/upload/landmark flow.
   User suggested evaluating **driver.js** (lightweight, no-dependency
   JS library for step-by-step UI tours/spotlights) as a candidate —
   not yet evaluated against alternatives or scoped.
3. **Developer documentation** — onboarding-level docs for a future
   maintainer (architecture overview, the Django apps' responsibilities,
   the RunPod/B2 integration points, local dev setup) distinct from this
   security-remediation tracking doc and the various dated session-notes/
   CR docs under `docs/`, which are historical records, not reference docs.

- Trigger: next dedicated documentation pass, or when onboarding a new
  developer/maintainer makes the gap acutely felt.

## TODO — evaluate reducing region detail exposed cross-owner (found 2026-07-24)

Follow-up to the region-visibility fix above (region dropdown/picker now shows
`Region.objects.all()` instead of only the current user's own regions — see the
"region dropdown empty" TODO earlier in this doc). A `/post-production` security
review (tier 4, Opus) flagged that this widening exposes more than region
*names*: `templates/core/location_create.html` and `multi_location_create.html`
render `origin_latitude`/`origin_longitude`/`radius` for **every** region shown,
not just name+radius (unlike `region_overview.html`, which only ever showed
name+radius). For an ibex conservation app, precise study-area coordinates carry
real poaching-sensitivity if this ever extends beyond the current small,
trusted, authenticated research team.

**User's proposed direction, not yet decided**: reduce the info shown per
region in the list/dropdown to name-only, revealing full coordinates only for
the specific region a user actually selects.

**Why this isn't a simple follow-up** — flagged during the same session as
"could become complicated quickly":
- The location-assignment UI is map-based (Leaflet/OSM), which likely needs
  each region's coordinates to plot it as a circle/marker for the user to
  visually place a location relative to — hiding coordinates for
  not-yet-selected regions may break that visual-placement UX entirely, not
  just hide harmless detail.
- A "reveal full detail only on selection" pattern would likely need a small
  AJAX endpoint (fetch one region's coordinates on demand) rather than a
  template-level field hide — a real feature change, not a queryset tweak.
- Whether this tradeoff (visual usability vs. coordinate exposure) is even
  worth solving depends on how researchers actually use the picker day to
  day — domain knowledge the professor has and this session doesn't.

**Explicitly deferred pending professor/domain-owner direction** — do not
implement a guess at this. The already-committed region-visibility fix itself
is not blocked by this; this is a distinct, deeper follow-up.

**Corollary if the direction instead comes back "private-by-design is
correct"** (i.e. the region-visibility fix above should be reverted, not
just this coordinate-detail question): `region_overview` (`core/views.py`,
the "Region" Dashboard list page) has been doing `Region.objects.all()`
**unfiltered since before this session** — pre-existing, untouched by the
region-visibility fix. Reverting just the two dropdown-filter call sites
would NOT make regions private end-to-end; `region_overview` would also need
to become owner-scoped, or the app would end up with a private
"assign region" step sitting next to a public "browse all regions" page —
an inconsistent, half-private state. Any decision to go private needs to
scope both places together, not just the two sites this session's fix
touched.

- Trigger: professor/domain-owner confirms whether coordinate-level exposure
  is acceptable as-is, or specifies which UX tradeoff they'd prefer; if the
  answer is "go private instead," re-scope to include `region_overview`.

## TODO — IDOR: `location-id`/`oid` unauthenticated-relative in `save_image_location`/`create_loaction` (found 2026-07-24)

Found by a Fable5 adversarial review run against the region-visibility fix CR
(confirmed pre-existing, NOT introduced or worsened by that fix — the review's
actual verdict on the fix itself was clean, zero bypasses).

- `save_image_location` (`core/views.py:617-634`, `@login_required` only):
  `get_object_or_404(Location, pk=location_id)` — `location_id` comes straight
  from the POST body with **no check that this Location belongs to the
  caller**. Any authenticated user can POST an arbitrary `location-id` and
  overwrite another user's image's `latitude`/`longitude`/`source`/`region`.
  `Location` has no direct `owner` field (ownership is only via the
  `IbexImage.location` OneToOne) — no existing check ties the two together
  here.
- `create_loaction` (`core/views.py:653`, `@login_required` only):
  `get_object_or_404(IbexImage, id=oid)` — same shape, no owner filter on the
  image `oid` — any authenticated user can load any other user's image's
  locate/landmark page via a guessable/enumerable integer id.

Both are pre-existing (predate this session), independent of the region
queryset widening (they read `location-id`/`image-id`/`oid` directly from the
request, never from `region_qs`). Given the app's current threat model (~20-50
trusted, authenticated researchers, not public-facing), the practical impact
is scoped to insiders — but it's a real cross-user write/read primitive worth
closing, and needs the same "shared vs. private" design input as the region
questions above (are images/locations meant to be editable by any
authenticated user, or only their owner/creator?).

- Trigger: next full security-remediation batch, or before this app's user
  base or threat model changes from the current small trusted group.

## TODO — behavioral gaps in auth-hardening test coverage (found 2026-07-25)

Surfaced by a `/request-adherence` check run against the auth/session
hardening CR (see the RESOLVED entry above) — the requested login/logout/
password-change coverage (`tests/webibex/test_manual_login_logout_check.py`)
is complete, but three related behaviors were never explicitly requested and
remain untested:

- **`SECURE_SSL_REDIRECT`'s actual redirect behavior**: all
  simulated-production tests use Django's test `Client(..., secure=True)`,
  which makes the request already look secure and bypasses the redirect
  path entirely. The 301/302-on-plain-HTTP behavior itself has no test.
  Lower risk since Railway's own edge independently enforces HTTPS
  (confirmed via Railway's public-networking docs during this CR) — this
  setting is defense-in-depth, not the primary guarantee.
- **`Strict-Transport-Security` response header presence**:
  `SECURE_HSTS_SECONDS=3600` is asserted as a settings *value*
  (`tests/webibex/test_settings_security_hardening.py`), but no test
  confirms Django's `SecurityMiddleware` actually emits the header on a
  live response.
- **CSRF failure case** (missing/invalid token) on an authenticated POST:
  only the happy path (valid CSRF) is tested for password-change; the
  reject-on-invalid-token path isn't. Likely already covered by Django/
  allauth's own test suite rather than anything this session's code
  touched.

- Trigger: next dedicated test-coverage pass on the auth/session surface,
  or if any of these three specifically becomes suspect during a future
  auth-related bug investigation.

## TODO — ruff baseline findings, first run on this codebase (found 2026-07-25)

`pyright`/`ruff` were added as dev dependencies this session
(`requirements-dev.txt`) as part of the `/post-production` tier-4 gate for
the auth-hardening CR. This is ruff's first-ever run on this codebase — no
`ruff.toml`/`[tool.ruff]` config exists yet, so it ran with bare defaults,
not even the `python.md`-recommended baseline config.

8 findings, all confirmed **pre-existing** (verified via `git diff` — none
on lines touched by this session's changes):

- `conftest.py`: 4x `RUF100` unused `# noqa: E402` directives (E402 isn't
  enabled in ruff's default rule set, so the pre-existing noqa comments are
  now flagged as unnecessary).
- `tests/core/test_models.py:6`: `F401` unused import (`Region`).
- `tests/core/test_models.py:61`: `SIM117` nested `with` statements
  (mergeable).
- `tests/core/test_utils_pure.py:17`: `I001` import block unsorted.
- `webibex/settings.py:13`: `I001` import block unsorted.

All 8 are marked fixable by `ruff --fix` (mechanical, low-risk). Left
untouched this session per minimal-blast-radius discipline — these files
were either only comment-edited (`conftest.py`) or pure `git mv` renames
with zero content change (the `tests/core/*.py` files), so "fixing" them
now would mean modifying files outside this session's actual scope.

- Trigger: next dedicated lint-cleanup pass, or set up a `ruff.toml`
  matching `python.md`'s recommended baseline config (which would enable
  E402 and make the noqa comments meaningful again) and run `ruff --fix`
  project-wide as its own CR.
