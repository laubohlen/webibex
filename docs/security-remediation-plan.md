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

**2026-07-26 update**: a `moto`-based in-process S3-mock test tier now exists for
`core/b2_utils.py` (`tests/core/test_b2_utils_moto.py`, pinned `moto==4.2.14`,
gated behind a `moto_s3` pytest marker) and has been verified with real local
execution — all 15 tests pass, `core/b2_utils.py` coverage is 100% under this
file. This confirms `moto==4.2.14` is compatible with the exact pinned
`boto3==1.26.0`/`botocore==1.29.165` triangle above, but it does **not** unblock
this landmine itself: moto simulates S3 semantics, not Backblaze B2's actual
behavior (B2 is only S3-*compatible*), so the real triangle bump still needs the
dedicated B2 test bucket to verify against. moto 5.x is deliberately deferred to
be bumped together with that future triangle bump, not independently (moto's
simulated-S3 fidelity is calibrated to the botocore version it mocks).

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

**Resolved (2026-07-26)**: the deferred Lauren conversation happened and RunPod access
is confirmed. User's call (common-sense, not a new investigation): real e2e manual
tests already exercise the actual RunPod inference container (via
`INFERENCE_ENDPOINT_URL_OVERRIDE` in `core/utils.py`'s `endpoint_inference()`), so the
old local-dev TF1 fallback path is no longer needed for anything. `get_tf()`/`_tf`/
`model_is_local`/`ENDPOINT_LOCALLY` removed from `core/utils.py`/`webibex/settings.py`;
the orphaned 100MB `core/embedding_model/` TF1 SavedModel binary tree deleted. Turned
out to be even lower-risk than scoped: `ENDPOINT_LOCALLY` was hardcoded `True`, so
`model_is_local` evaluated to `False` in every environment — the local-model branches
were already unreachable dead code, not just rarely used. Two extra stale references
found and cleaned up while verifying no dangling pointers remained: `.gitignore`'s dead
commented-out line and `.coveragerc`'s now-pointless `omit` entry, both referencing the
deleted directory. `core/test_model.py` (the standalone dev script from the original
finding above) was already removed in an earlier, unrelated commit (`2bde17e`) —
nothing to do there. Added 6 new regression tests to `tests/core/test_utils_io.py`
(branch-selection equivalence proving the `if` simplification is behavior-identical,
plus previously-uncovered error paths: B2 download returning `None`, undecodable/
corrupt image bytes, missing local file) alongside 2 structural regression guards
(`hasattr` checks confirming `get_tf`/`ENDPOINT_LOCALLY` are gone — both went red
against pre-deletion code, green after, proving they're real guards not tautologies).

### OpenStreetMap ToS exposure (CR-2) — status: on the radar, not blocking (2026-07-31)

webibex hits `tile.openstreetmap.org/{z}/{x}/{y}.png` directly via Leaflet in
**6 templates**, verified by grep 2026-07-31 (doc previously said 7 — corrected):
`templates/core/location_create.html`, `multi_location_create.html`,
`region_create_naming_error.html`, `region_delete.html`, `region_read.html`,
`region_update.html`. Each template has its own inline `<script>` block with a
duplicated `L.tileLayer(...)` call — there's no shared partial/include for the map
JS. `region_create.html` itself has no map (only its `_naming_error` variant does).
Attribution is correct, but OSMF's Tile Usage Policy is explicit that this server is
for light/dev/evaluation use only — OSMF can throttle/block without warning.

**Fix**: swap to **MapTiler** (same `L.tileLayer()` call shape, different URL + API
key). Verified against professor-confirmed scale (20-50 non-concurrent users):
MapTiler's free tier is 5,000 map-sessions/month (session-based — panning within a
session is free), no non-commercial restriction. Compared against Stadia Maps
(200,000 credits/month but explicitly non-commercial-use-only — disqualifying) and
Thunderforest (150,000 tile-requests/month). Even a generous estimate (~25k
tile-equivalents/month) leaves 6-10x headroom on all three; MapTiler chosen for the
session-based accounting plus no non-commercial restriction to verify against the
university's status.

**Implementation checklist — beyond just getting a MapTiler (or similar) account +
API key** (verified this session: Leaflet 1.9.4 loaded via CDN in `templates/base.html`,
no CSP/`django-csp` configured anywhere in `webibex/settings.py` — confirmed by
grep, so no CSP whitelist step needed):
1. Add the key as an env var (`env("MAPTILER_API_KEY")` in `webibex/settings.py`,
   following the exact pattern already used for `AWS_ACCESS_KEY_ID` etc.) — set it
   in Railway's env vars for prod, and in local `.env`/`.env.local` for dev.
2. Since the 6 templates each inline their own `<script>` block (no shared JS
   include), the key needs to reach all 6 — cleanest is a Django context processor
   injecting `MAPTILER_API_KEY` into every template context, rather than threading
   it through each view's render call individually.
3. Update the tile URL + attribution string in all 6 templates. **Attribution is
   not a straight swap** — MapTiler's ToS requires its own attribution alongside
   OSM's (not just the current OSM-only copyright line) — confirm the exact
   required wording against MapTiler's current ToS at implementation time, don't
   assume.
4. Restrict the API key by domain/referrer in the MapTiler dashboard (prod domain
   + local dev) — the key is inherently visible client-side (Leaflet tile requests
   are always browser-side), so domain restriction is the actual mitigation against
   quota theft, not secrecy.
5. No CSP update needed (confirmed above) — one less step than a typical
   third-party-embed swap.
6. No browser/E2E test suite currently exercises these map pages (confirmed no
   Playwright/Selenium in this repo) — nothing to mock for CI today, but flag it
   if such tests are added later so they don't hit live MapTiler quota.
7. Update `.env`/deployment docs to note the new required env var so a future
   redeploy doesn't silently break maps by omission.
8. Flip this section's status to done once shipped, matching the other resolved
   items in this doc.

- Trigger: not urgent at current scale (6-10x headroom), but should happen before
  OSMF actually throttles/blocks — no fixed deadline, just don't let it linger
  indefinitely.

### Auto-match-or-new-ID requirement (CR-1) — fully already built (corrected 2026-07-31)

Unlike the ibex_stambecchi HF Space (which genuinely lacks search at upload time),
webibex's `default_chip_compare_view` / `project_chip_compare_view` /
`geographic_chip_compare_view` already run full nearest-neighbor gallery search and show
top-5 matches with distances before any human decision; `created_animal_view` already
exists as the "no match" path.

**Correction**: the line above about the threshold being "only displayed, not used to
auto-decide" was wrong — verified against the actual templates this session.
`templates/core/result_default.html:94-98` and `result_refined.html:98-102` already
color-code every candidate: **green badge** if `distance <= threshold` (9.3, meaning a
reliable identification), **red badge** otherwise (unlikely to be the same animal). The
domain semantics (explained by the developer this session): below 9.3 is a reliable
match, above it is unlikely to be the same animal — exactly what the green/red badges
already encode. No gap here; nothing to build. Second finding this
session (after the `new-ibex` button) where this doc undersold what's already shipped —
worth a skim of the remaining "mostly already built" / "gap" claims in this doc for the
same staleness before trusting them at face value.

## Remediation plan — 6-9 person-days

| Task | Estimate | Notes |
|---|---|---|
| dcg profile fix | done | already merged to main |
| Routine CVE bumps (Django, pillow, django-allauth, django-filer, requests, lxml, setuptools) | **done (2026-07-09)** | commit `480607b` — final versions in the table below |
| boto3/botocore/urllib3 triangle | ~1.5-2.5d | **blocked on B2 test bucket** — root cause + fix path documented in `tmp/memo-questions-for-lauren.md` item #5, not yet applied |
| TensorFlow removal | ~0.5d | **done (2026-07-26)** — dead local-dev TF1 branches + orphaned `core/embedding_model/` removed; see Resolved note below |
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

**Professor confirmed (2026-08-08 email reply)**: not needed for now — happy to
leave it hidden (matches the already-committed `3d5168d` crash-guard/hide fix).
If/when it is implemented: **hard delete** (not soft-delete/archive), with an
explicit confirmation step before the destructive multi-select action. Still no
decision on cascade scope (`IbexChip`/`Embedding`/stored file) — ask
specifically when this is picked up, since it wasn't addressed directly.

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

**Professor partially answered (2026-08-08 email reply)**: shared/cross-owner
region *visibility* is confirmed intentional — her original design was for
any user to browse and compare against other users' regions, while only
being able to act on their own images (see the corresponding note on the
IDOR TODO below). So "go private instead" is **not** the direction; the
`region_overview`-consistency corollary above no longer applies. The
coordinate-detail-exposure UX tradeoff itself (name-only in dropdown vs. full
detail) is still open — she wants to discuss it further before deciding.

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

**Professor confirmed the blocking ownership semantics (2026-08-08 email reply)**:
users should be able to browse/use regions created by other users (reduces
cross-region comparison errors — an animal seen in Gran Paradiso is never the
same individual as one seen in Austria), but should only be able to **act**
(edit, delete) on their own images/locations. This matches `update_region`'s
existing `region.owner != request.user` → `HttpResponseForbidden` pattern —
that pattern is now the confirmed direction, not just an inferred precedent.
Unblocks implementing owner-only checks in `save_image_location` and
`create_loaction`. She also flagged wanting to talk through the deeper region
design further, so treat this as directional confirmation for the ownership
question specifically, not a sign-off on every region-related detail.

Also found while re-reading this view for the above: `save_image_location`
additionally does `get_object_or_404(Region, pk=region_id)` with no ownership
check on `region_id` either — same bug class, not previously listed here.
Given the confirmed "regions are shared/browsable" direction, this one may be
intentional (any user can select any region for their own image) rather than
a gap — worth a quick confirm, not necessarily a fix, when this TODO is
picked up.

## TODO — behavioral gaps in auth-hardening test coverage (found 2026-07-25, RESOLVED 2026-07-25)

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

Fix applied (2026-07-25): all three added to
`tests/webibex/test_manual_login_logout_check.py`, test-only changes (no
production code touched) —

- `test_plain_http_request_redirects_to_https_under_simulated_production_settings`:
  a plain (non-`secure=True`) GET under the full simulated-production
  `override_settings` block now asserts a 301 with an `https://`-prefixed
  `Location`, exercising the redirect path the existing `secure=True` tests
  structurally cannot reach.
- `test_strict_transport_security_header_present_on_secure_response`
  (needs `db` — rendering the login page hits the database, unlike the
  redirect test above which returns before the view runs): asserts
  `response.headers["Strict-Transport-Security"] == "max-age=3600"` on a
  live `secure=True` response.
- `test_password_change_rejects_post_with_invalid_csrf_token`: logs in with
  a normal `Client()`, copies the session cookie into a second
  `Client(enforce_csrf_checks=True)`, then POSTs the password-change form
  with a deliberately wrong `csrfmiddlewaretoken` and asserts 403.

All 10 tests in the two auth-hardening test files pass; full suite (159
passed, 1 skipped, 1 xfailed) and `ruff check` on the changed file are
clean.

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

**RESOLVED (2026-07-27):** superseded by the two CRs below. Commit `a20220b`
fixed all 8 findings listed above (mechanical, on the 4 named files) as a
standalone cleanup, ruff still running under bare defaults. Commit (this
session, see `docs/changes/2026-07-27-ruff-baseline-config.md`) then added
the actual `ruff.toml` this TODO called for, with the `python.md`-recommended
curated ruleset (`E,F,UP,B,S,SIM,PIE,C4,T20,ANN,RUF`) — but enforced only on
files at 100% measured test coverage (plus `tests/**` + root `conftest.py`,
self-verifying). Below-100%/unmeasured production files are deferred via
`per-file-ignores`, tracked in the new TODO immediately below. E402 is back
to being meaningful (the `# noqa: E402` comments in `conftest.py` are live
again under the curated ruleset).

## TODO — ruff-baseline deferred files, re-enable as coverage improves (added 2026-07-27)

`ruff.toml` (added this session, see
`docs/changes/2026-07-27-ruff-baseline-config.md`) enforces the curated
ruleset only on files at 100% measured line coverage (`--cov=core
--cov=simple_landmarks --cov=webibex`, per `pytest.ini`) plus `tests/**` +
root `conftest.py` (self-verifying). All other `.py` files are deferred via
individual `per-file-ignores` entries in `ruff.toml` — full ruleset
suppressed, not just a subset — so re-enabling one is a one-line removal,
not a glob edit.

Deferred files (coverage % as of 2026-07-27's full `pytest --cov-report=
term-missing` run; "unmeasured" = outside the three `--cov` target packages,
or excluded via `.coveragerc`'s `omit =` list):

| File | Coverage | Reason deferred |
|---|---|---|
| `core/admin.py` | 72% | below 100% |
| `core/models.py` | 98% | below 100% |
| `core/signals.py` | 46% | below 100% |
| `core/templatetags/custom_template_tags.py` | 93% | below 100% |
| `core/utils.py` | 71% | below 100% |
| `core/views.py` | 23% | below 100% |
| `simple_landmarks/views.py` | 0% | below 100% |
| `webibex/urls.py` | 50% | below 100% |
| `manage.py` | unmeasured | `.coveragerc` omit |
| `webibex/asgi.py` | unmeasured | `.coveragerc` omit |
| `webibex/wsgi.py` | unmeasured | `.coveragerc` omit |
| `db_management/__init__.py` | unmeasured | `.coveragerc` omit (`db_management/*`) |
| `db_management/populate_created_at_field.py` | unmeasured | `.coveragerc` omit (`db_management/*`) |
| `scripts/run_local_e2e_server.py` | unmeasured | outside the three `--cov` target packages |

- Trigger: re-enable per-file once that file reaches 100% measured coverage
  (or gains coverage measurement for the first time) — remove its
  `per-file-ignores` entry in `ruff.toml`, run `ruff check <file>` to see
  what actually fires, triage as its own small CR.
- `core/admin.py` was not called out explicitly in this session's original
  scoping conversation (only 7 of these 8 production files were) but is
  deferred here anyway per the general coverage-gating rule (72% < 100%) —
  flagged for awareness, not a deviation.
- `db_management/test.py` was on this table's first version, deferred as
  "unmeasured". Found 2026-07-27 (same session) to be a byte-for-byte
  duplicate of `db_management/populate_created_at_field.py` — deleted rather
  than kept deferred; its `ruff.toml` entry, `conftest.py`'s `collect_ignore`
  reference, and `tests/webibex/test_infra.py`'s T03 test were all
  updated/removed accordingly. See
  `docs/changes/2026-07-27-ruff-baseline-config.md` for the original entry;
  this is a same-day correction, not a separate CR.

**UPDATE (2026-07-27, same day, follow-up session):** `core/models.py` (98%)
and `core/templatetags/custom_template_tags.py` (93%) re-enabled as a
one-off, explicit user-approved exception — the documented gate policy
stays 100% coverage, this was a case-by-case call, not a threshold change.
Both files were already ruff-clean under the full curated ruleset once
enabled: fixed 3 `RUF012` (mutable Django `Meta`/choices class attributes,
now `ClassVar`-annotated), 1 `E501`, 1 `ANN204`, and on
`custom_template_tags.py` 8 `ANN001`/`ANN201`/`ANN002`/`ANN003` (added type
annotations to `dict_get`/`post_task_redirect`, narrowed from `Any` to avoid
`ANN401`). Verified net-zero new `pyright` errors (6 pre-existing
django-stubs-gap errors, unchanged) and full test suite green (184 passed,
1 skipped, 1 xfailed, coverage unchanged). See
`docs/changes/2026-07-27-ruff-coverage-gate-expansion.md`.

**STALE — flagged 2026-07-31 (docs-accuracy pass), not fixed here (config
change, out of scope for a docs check):** live `pytest --cov` run this
session shows `core/admin.py` and `core/utils.py` both at **100%** measured
coverage right now (251 passed, 1 skipped, 1 xfailed) — same graduation
condition that already got `core/models.py`/`custom_template_tags.py`
re-enabled above. But `ruff.toml`'s `per-file-ignores` **still lists both of
them as deferred** (full ruleset suppressed), unlike `core/models.py`/
`custom_template_tags.py` which were correctly removed from that list. Per
this doc's own stated policy ("remove its `per-file-ignores` entry... one
line at a time"), these two entries should have been removed already. Actual
current coverage snapshot, for reference (same run): `core/signals.py` 98%
(3 missing, matches the already-documented dead branches below — correctly
still deferred), `custom_template_tags.py` 94% (correctly not deferred),
`core/views.py` 23%, `webibex/urls.py` 50%, `simple_landmarks/views.py` 0%
(all three correctly still deferred, unchanged from the original table) —
`core/admin.py`/`core/utils.py` are the only two entries actually out of
sync with reality.

- Trigger: remove `"core/admin.py"` and `"core/utils.py"` from `ruff.toml`'s
  `per-file-ignores`, run `ruff check` on both to see what fires, triage as
  its own small CR — same mechanical process already used for
  `core/models.py`/`custom_template_tags.py` above.

## TODO — SonarQube first-ever scan findings, webibex (found 2026-07-27)

`webibex`'s first-ever SonarQube analysis ran this session (host-side,
triggered outside the devcontainer; fetched in-container via
`/sonar fetch /workspace/webibex`, revision `12b93fa`). Project-wide baseline
(pre-existing debt, not introduced by any change in this session):

- **478 issues**: 5 BLOCKER, 161 CRITICAL, 215 MAJOR, 94 MINOR, 3 INFO.
- **0 hotspots**.
- Pyright project-wide (per-module, `--outputjson`): 207 errors total
  (`core/` 131, `tests/` 70, `webibex/` 4, `simple_landmarks/` 2,
  `db_management/` 0) — all INFO-severity, predominantly `django-stubs` gaps
  (`.objects` manager, `CharField`/`str` return mismatches, `env()` `NoValue`
  defaults), consistent with the known gap tracked since
  `docs/changes/2026-07-27-boto3-stubs-typing.md`.

**Concretely verified this session** (scoped to the
`ruff-coverage-gate-expansion` CR's two touched files): 6x `python:S6553`
("Remove this `null=True` flag") on `core/models.py` lines 17, 18, 21, 22,
29, 65 — all on `Animal`/`Region`/`Location` `CharField`/`FloatField`
declarations **outside** that CR's diff hunks, pre-existing, left untouched
(a `null=True` removal on a live Django field is a migration-adjacent
change, not a lint fix — out of scope for a lint CR).

- Trigger: dedicated SonarQube triage session — start with the 5 BLOCKER +
  161 CRITICAL issues (highest severity first), then decide whether MAJOR/MINOR
  get systematically worked through or left as tracked debt. The 6 verified
  `S6553` findings above are a concrete starting point for the `django`-tagged
  rule category. django-stubs installation (tracked separately, see
  `docs/changes/2026-07-27-boto3-stubs-typing.md`) would also collapse a large
  fraction of the 207 pyright errors in one pass.
- Not actioned this session — scope was the ruff coverage-gate expansion CR,
  not a general SonarQube cleanup.

## TODO — no automated database backup, Railway free tier (found 2026-07-27)

Production database (`DATABASES["default"]` via `dj_database_url.parse(env("DATABASE_URL"))`
when `ENVIRONMENT == "production"`, `webibex/settings.py:151`) has **no backup
mechanism of any kind**. Confirmed via repo-wide grep (`backup`, case-insensitive,
across `.md`/`.py`/`.toml`/`.yml`/`Procfile`) — zero hits describing an actual DB
backup process. User confirmed directly: Railway's automated Postgres backups are a
paid-plan feature, and this account is not on the Pro plan, so nothing is currently
capturing point-in-time recovery data for this app's only database.

**Risk**: any bad migration, accidental bulk delete, or Railway-side incident is
unrecoverable data loss for the entire ibex photo-ID dataset (images, locations,
landmarks, region/owner assignments) — there is no local `db.sqlite3` fallback in
production (that file only exists for local dev; production always uses
`DATABASE_URL`/Postgres per the `ENVIRONMENT == "production"` branch above).

**Options, not yet evaluated in depth**:
- Upgrade the Railway plan to unlock its built-in automated Postgres backups
  (simplest, but recurring cost — needs the professor's budget sign-off).
- Scheduled `pg_dump` pushed to the B2 bucket this app already uses for image
  storage (`core/b2_utils.py` — credentials and bucket access already wired up,
  no new external dependency). Needs a scheduler (Railway cron job / GitHub Actions
  on a schedule, since this repo has no CI infra yet — see the CI-scaffold TODO
  above) and a retention/rotation policy.
- Railway CLI-driven export triggered by an external scheduler (same scheduling
  gap as above).

- Trigger: professor confirms acceptable risk tolerance / budget for a paid tier,
  or this becomes the next priority item given the CI-scaffold gap already blocks
  the B2-cron option from being automated cleanly.

**Cost research (2026-07-31)**, direct from Railway's own docs
([backups](https://docs.railway.com/volumes/backups),
[point-in-time recovery](https://docs.railway.com/volumes/point-in-time-recovery),
[pricing](https://railway.com/pricing)):
- Volume/backup storage is billed at **$0.00000006/GB/second ≈ $2.59/GB-month**,
  incremental/copy-on-write (only the bytes unique to each backup are charged).
  Scheduled backups: daily kept 6 days, weekly kept 1 month, monthly kept 3
  months. PITR retains roughly the last 4 full backups (~4 weeks). Manual
  backups are capped at 50% of the volume's size.
- The docs themselves don't explicitly gate backups/PITR to a paid tier, but this
  contradicts what the user directly observed in the actual Railway dashboard
  (paid-plan-only) — trusting the live-account observation over ambiguous docs.
  Under that constraint, the real cost isn't the per-GB storage rate (our DB is
  small, this would be low single-digit dollars/month) — it's the **plan-tier
  delta**: Hobby is $5/mo, Pro is $20/mo, so unlocking Railway's built-in backups
  costs **~$15/mo more** than what's already being paid, regardless of actual
  data volume.
- Compare to the already-planned DIY option (this session's fix batch, R1: a
  `backup_db` management command doing `pg_dump` → gzip → push to the existing
  B2 bucket): Backblaze B2 storage runs roughly $6/TB/month
  (~$0.006/GB-month), and there's no new service/account needed — `core/b2_utils.py`
  already has working credentials/access. Marginal cost is effectively cents/month,
  not $15/mo.
- **Revised decision (2026-07-31, supersedes the "skip Pro" recommendation
  above)**: the DIY `backup_db` → B2 script is PAUSED, not being built right
  now. Cost isn't the only variable — the DIY path also costs *developer time*
  (build + the mandatory restore-drill verification below) before it's trusted
  for production, whereas Railway Pro's built-in backups are available
  immediately, no dev/verification wait. This is genuinely the professor's
  call, not an engineering one: pay ~$15/mo more starting now for an
  immediately-available backup, vs. free but wait (roughly) a couple of dev
  sessions for the DIY script to be built and restore-verified. Question is
  in the outstanding-questions email draft
  (`ibex_stambecchi/tmp/draft-email-professor-open-questions.md`). Do not
  resume building the DIY script until she answers.

**Professor answered (2026-08-08 email reply)**: no urgency — fine to keep
waiting for now. When the app is actually opened to more users, re-assess how
much dev work the DIY B2 script needs at that point; do it that way if there's
still time/resources, otherwise she'll arrange payment for Railway Pro. Net
effect: **decision is "revisit when opening to other users," not "pick now"**
— do not resume building `backup_db` yet; re-open this question as part of
the opening-to-other-users planning, not before.

**Storage destination found (2026-08-11)**: the webibex project email
(`wibex@ikmail.com`) has a dedicated Infomaniak kDrive (kSuite), 15GB free
— a natural destination for encrypted DB backup artifacts (the interim
manual backup plan's `.dump.enc` output, e.g.), better suited than raw
email attachments for anything recurring (no size juggling, stays
organized in one place). The actual kDrive URL isn't recorded in this
doc — ask the user for the current link when needed. Used this session:
the real GATE-evidence dump (`webibex_restore_drill.dump.enc`, produced
by the live restore-drill run — see
`docs/changes/2026-08-09-db-restore-drill.md`) was uploaded there as
`webibex_restore_drill_20260811-1558.dump.enc.zip`, zipped purely to
keep it as a standard file type for cloud storage convenience, not for
compression (AES-256 ciphertext doesn't compress) or added security (the
zip itself is not separately encrypted — no security benefit over the
already-strong `.dump.enc` layer, so not worth the extra
password-management friction). The decryption passphrase still travels
out-of-band from wherever the file itself lands, per the original plan.

## GATE — restore drill required before the id_code max_length migration ships (added 2026-07-31, updated 2026-07-31)

The current MVP-safety batch includes a schema migration (`Animal.id_code`
`CharField(max_length=10)` → `max_length=20`, widening only, no data loss). **Hard
requirement, decided by the user 2026-07-31: do not deploy that migration (or any
future migration) to production until whichever backup mechanism is in place has
been proven to actually restore, not just to run and upload successfully.** A
backup that writes a file but has never been restored is unverified — the point of
a backup is recoverability, not the act of dumping.

Which mechanism this applies to now depends on the professor's answer to the
pay-now-vs-wait question above:
- If she picks **Railway Pro**: verify a restore through Railway's own UI/CLI
  restore flow before trusting it (still don't skip verification just because
  it's a vendor feature).
- If she picks **DIY B2 script**: resume building `backup_db` (currently
  paused), then run the drill below.

Restore-drill checklist (applies either way):
1. Produce a backup via whichever mechanism was chosen, against a real (or
   realistic staging-equivalent) Postgres instance.
2. Actually restore that backup into a separate/scratch Postgres database.
3. Verify the restored DB is usable — at minimum, row counts match on `Animal`,
   `Region`, `Location`, `IbexImage`, `IbexChip`, `Embedding`, and a spot-check
   query (e.g. fetch one `Animal` by `id_code`) returns the expected data.
4. Only once that restore drill passes does the migration go out.

- Trigger: this gate is checked once, immediately before this batch's production
  deploy — not a recurring requirement, but every future backup-mechanism change
  should re-trigger a fresh restore drill before its next prod migration.

**GATE SATISFIED (2026-08-11):** real live run of `scripts/db_restore_drill.py`
(see `docs/changes/2026-08-09-db-restore-drill.md`) — real Railway GraphQL fetch,
real `pg_dump` against prod (`DATABASE_PUBLIC_URL`, not the private
`postgres.railway.internal` host), real `pg_restore` into a fresh ephemeral
`testcontainers` Postgres, all via `docker run --rm --entrypoint <binary>`
(the entrypoint-dispatch fix from this same session). Checklist items 1-3
confirmed: row counts matched on all 6 tables (`core_animal`: 110,
`core_region`: 3, `core_location`: 143, `core_ibeximage`: 143,
`core_ibexchip`: 142, `core_embedding`: 142) plus the `Animal` spot-check —
`=== overall: PASS ===`. Item 4 (the `id_code` `max_length` migration itself)
is unblocked; migrating and deploying it is a separate, not-yet-scoped step.
A handful of benign `WARNING: database "railway" has a collation version
mismatch` lines appeared (client/server glibc collation version drift,
non-fatal, stderr-only — never touches the piped binary dump stream) — noted
here in case it recurs, not something that needs fixing to trust this result.

## TODO — image/chip backup is a separate question from the DB backup above (found 2026-07-31)

Everything above (GATE, cost research) is about the Postgres DB — it does NOT cover
the actual photos/chips, which live in Backblaze B2 (`core/b2_utils.py`), a wholly
separate storage system. B2 itself is already durable cloud object storage, so it
isn't exposed to the "Railway volume gets wiped" risk driving the DB backup
discussion. The real risk to images is our own app code deleting them: confirmed,
`core/b2_utils.py:66-74`, `delete_files()`, performs a real hard-delete via the
S3-compatible `delete_objects` API — this is what the working single-image Delete
button calls today, and would be what any future real multi-delete implementation
(see the Tools-menu Delete TODO above) calls too.

**Mitigation is a B2 bucket-level setting, not app code**: Backblaze B2 supports file
versioning/Object Lock at the bucket level — if enabled, an accidental delete keeps
the prior version recoverable for a retention window. Whether this is currently
enabled on the production bucket is unknown — needs checking directly in the B2
console, not visible from this repo.

- Trigger: decide alongside the still-open "what should Delete actually do" design
  question above, so real delete semantics and B2 versioning settings end up
  consistent (e.g., even a hard delete from the app stays recoverable at the
  storage layer for N days if versioning is on). Not blocking current work.

## TODO — ransomware/mass-deletion remediation: immutable backups (found 2026-07-31)

Raised explicitly: is immutable/versioned backup "a thing" against ransomware-style
attacks (compromised credentials used to delete or overwrite both live data and its
backups)? Yes — this is a standard, well-established pattern, not exotic:

- **Object Lock / WORM (write-once-read-many)** is the mechanism. Both Backblaze B2
  and AWS S3 (and S3-compatible stores generally) support it at the bucket level. Two
  modes: **governance mode** (privileged/root credentials can still override/delete)
  and **compliance mode** (nobody — not even the account owner — can delete or
  overwrite a locked object before its retention period expires, full stop). This is
  distinct from the plain versioning discussed in the TODO above (which protects
  against *accidental* app-level deletes but not a *malicious* actor with delete
  permissions on the account).
- **Applies to two places, not just one**: (a) wherever the Postgres DB backups land
  (whichever mechanism the professor picks — Railway Pro's built-in backups or the
  paused DIY B2 script), and (b) the B2 bucket holding the actual photos/chips
  themselves. A ransomware scenario that compromises the app's B2 credentials could
  otherwise delete live images AND any backup copies sitting in the same bucket with
  the same credentials — Object Lock in compliance mode is specifically designed to
  prevent exactly that.
- **Credential blast-radius matters as much as Object Lock itself**: if backups are
  written using the same B2 application key the live app uses (which already has
  delete permission, per the TODO above), a compromise of that one credential
  threatens both live data and backups together. Best practice: use a separate,
  minimally-privileged B2 application key for writing backups (write-only, no delete
  permission if B2's key-scoping supports it), ideally to a separate bucket, so a
  compromised app credential can't reach the backup copies at all.
- **Not evaluated in depth yet**: exact retention period, governance vs. compliance
  mode choice (compliance mode is stronger but also means genuinely nobody can delete
  early, including us, if we ever need to for a legitimate reason — e.g. GDPR-style
  erasure requests), and whether a fully separate provider/account (true offsite,
  beyond even a compromised B2 account) is warranted given the actual threat model at
  this scale (~tens of users, academic research data, not high-value financial data).

**Feasibility check (2026-07-31), direct from Backblaze's own docs**: Object Lock
works with the current plan, no upgrade needed — *"There is no extra cost to use
Object Lock. However, you are responsible for the normal charges that are
associated with storing the locked file."* No plan-tier gating found anywhere in
Backblaze's documentation; it reads as a standard feature, not an upsell. Practical
notes:
- Can be added to an **existing** bucket (not just at creation), via the web console
  or the `b2_update_bucket` API call — the current webibex bucket doesn't need to be
  recreated.
- **Once enabled, it cannot be disabled** — a one-way switch, so retention period
  should be chosen deliberately before flipping it on, not left as a default.
- Cannot be enabled on restricted buckets, shared buckets, snapshots, or buckets with
  replication configured. The webibex bucket appears to be a plain private bucket
  (per `core/b2_utils.py`), so this exclusion likely doesn't apply — not confirmed,
  needs checking directly in the B2 console (not visible from this repo).
- Managing retention settings needs specific app-key capabilities
  (`readBucketRetentions`/`writeBucketRetentions`) — a one-time setup/admin action,
  not something the app's day-to-day runtime credentials need.

- Trigger: decide alongside the DB-backup mechanism choice (professor's pay-vs-wait
  call) and the B2-versioning decision above, so all three data-protection layers
  (DB backup, image versioning, ransomware/immutability) get resolved together rather
  than piecemeal. Not blocking current work — this is a hardening layer on top of the
  basic backup, not a prerequisite for it.

## TODO — no real browser-automation tool available for manual E2E checks (found 2026-08-01)

While manually verifying the Tools-menu Delete-crash fix, no browser-automation tool
was actually available in this sandbox: `chromium-cli` not installed, `npm install
playwright` returns a 403 from the npm registry, and system `pip install playwright`
is blocked (externally-managed environment). Fell back to driving the live dev server
directly over HTTP with `requests` instead — valid for this specific check (the Tools
dropdown is plain server-rendered HTML, no client-side option generation), but not a
substitute for a real browser/JS-level check in general.

**User has a playwright-vnc Docker image available** that could be bound in for this —
would let this kind of manual E2E check (and possibly a future proper automated
Playwright suite, see the CI-scaffold gap already tracked above) actually drive a real
browser instead of HTTP-only checks. Not set up yet, explicitly deferred ("maybe not
right now").

- Trigger: next time a change needs real browser/JS-level verification (not just
  server-rendered HTML), or when the CI-scaffold work above is picked up and a decision
  is made on whether to invest in an automated Playwright suite at all.

## TODO — no test-strategy-review has ever been logged for this project (found 2026-08-01)

Surfaced by post-production's periodic check (`~/.claude/feedback/log.jsonl` has zero
`test-strategy-review` entries for `webibex`). Deferred this session — the change being
reviewed was small and already had 28 targeted tests; a full project-wide test-strategy
audit against the `python-testing.md` checklist (test type coverage, deferred
candidates, mutation operators, property styles, fuzzing corpus) is a separate,
deliberate task, not a side effect of a small commit.

- Trigger: next dedicated test-infrastructure session, or when this keeps resurfacing
  across future post-production runs.

## TODO — `simple_landmarks/views.py` is dead startapp scaffold (found 2026-07-28)

Surfaced during the pre-refactor test-coverage push (see the "ruff-baseline deferred
files" TODO above — this file is one of the 11 files deferred from the ruff gate for
below-100% coverage). `simple_landmarks/views.py` is a 1-statement file (`from
django.shortcuts import render` + a boilerplate comment) at 0% measured coverage.

Confirmed dead, not just currently-disconnected: `git log --follow -- simple_landmarks/
views.py` shows exactly one commit (`83f73dc "started landmarking standalone app"`,
2025), never modified since — untouched `django-admin startapp` scaffold. Repo-wide
grep confirms zero imports of `simple_landmarks.views` anywhere, and the
`simple_landmarks` app has no `urls.py` to wire it up. `simple_landmarks` itself is a
real, used app (`INSTALLED_APPS`, `models.py`/`admin.py` actively used by
`core/admin.py`, `core/signals.py`, `core/views.py`) — only `views.py` is unused.

**Decision (user, explicit, 2026-07-28)**: skip writing a coverage test for this file
(testing dead code doesn't buy real safety) — flag for deletion in a separate, small
cleanup CR instead of folding it into the coverage-improvement CR.

- Trigger: dedicated cleanup CR — delete `simple_landmarks/views.py`, remove its entry
  from `ruff.toml`'s per-file-ignores deferred block (line ~24), confirm `pytest`
  coverage config (`pytest.ini`'s `--cov=simple_landmarks`) still resolves cleanly with
  the file gone.

## TODO — `UnboundLocalError` risk in `create_folder_for_animal_on_change` (found 2026-07-28)

Surfaced by code-planner while scoping the `core/signals.py` coverage CR (2nd file in
the pre-refactor test-coverage push, after `core/admin.py`). `core/signals.py`'s
`create_folder_for_animal_on_change` (`post_save` receiver on `IbexImage`, ~line 264
onward) has:

```python
if instance.side == "L":
    target_folder = left_folder
elif instance.side == "R":
    target_folder = right_folder
elif instance.side == "O":
    target_folder = other_folder
else:
    pass
# unconditionally, a few lines later:
instance.folder = target_folder
```

For any `instance.side` outside `{"L", "R", "O"}` (including `None`, the field's
default/unset state), `target_folder` is referenced before assignment —
**`UnboundLocalError`**, which crashes the signal and, with it, the `.save()` call that
triggered it. This only fires on the `instance.animal_id != instance._original_animal_id`
branch, i.e. when an `IbexImage`'s `animal` is set or changed after creation — a
plausible production sequence if animal assignment happens before manual side-tagging.

**Decision (user, explicit, 2026-07-28)**: stay test-only for the `core/signals.py`
coverage CR — add a test that documents the crash via `pytest.raises(UnboundLocalError)`
rather than fixing it inline. Tracked here as its own bug-fix item.

- Trigger: dedicated bug-fix CR — add an explicit `else: target_folder = None` (or
  raise a clearer domain error, or skip the folder-move entirely) before line
  `instance.folder = target_folder`; decide the right fallback behavior with the
  professor (silently skip vs. surface an error) before implementing.

## TODO — `get_decimal_from_dms` raises `TypeError` on malformed input instead of returning `None` (found 2026-07-28)

Surfaced by code-analyst while writing the `core/signals.py` coverage CR's test spec.
`core/signals.py`'s `get_decimal_from_dms(dms, ref)` has an inner `to_float(value)`
helper that catches conversion failures and returns `None` on bad input — but the
function's own arithmetic (`degrees + minutes / 60.0 + seconds / 3600.0`) that consumes
those `to_float` results sits **outside** the `try/except` that wraps the three
`to_float` calls. A DMS component that's indexable-but-non-numeric (e.g. the tuple
`(46.0, 30.0, "abc")`) makes `to_float` return `None` for `seconds`, and the very next
line's `None / 3600.0` raises an uncaught **`TypeError`**, not the `None` the function's
apparent contract implies for "any conversion failure."

**Not currently reachable from production traffic in a crashing way**: the only caller,
`extract_gps_coords`, wraps its `get_decimal_from_dms` calls in its own outer
`try/except Exception`, which swallows this `TypeError` and returns `(None, None)` —
so `process_uploaded_image` never sees the crash. This is a latent bug in a function
whose behavior doesn't match its own internal contract, not an active production
incident.

**Decision (user, explicit, 2026-07-28)**: stay test-only for the `core/signals.py`
coverage CR — tests assert the actual `pytest.raises(TypeError)` behavior on malformed
input (documenting current behavior), not fixed inline.

- Trigger: dedicated bug-fix CR — move the `degrees + minutes/60.0 + seconds/3600.0`
  arithmetic inside the same `try/except` (or add an explicit `if None in (degrees,
  minutes, seconds): return None` guard) so the function's return-`None`-on-bad-input
  contract actually holds for all three components, not just outright indexing
  failures.

## TODO — dead/unreachable branches in `core/signals.py` (found 2026-07-28)

Surfaced by code-analyst while writing the `core/signals.py` coverage CR's test spec —
two branches that no test (however constructed) can reach without changing the source,
which caps that file's achievable measured coverage at ~99%, not 100% (same situation
`core/models.py` (98%) and `custom_template_tags.py` (measured at **94%** as of this
session's coverage runs — 16 stmts, 1 miss; the 2026-07-27 TODO above cites 93% for
the same file, not re-verified here whether that reflects a since-changed stmt count
or a coverage.py rounding difference) already have a documented case-by-case
`ruff.toml` exception for — see the "ruff-baseline deferred files" TODO above).

1. **`core/signals.py:192-193`** — inside `process_uploaded_image`, the `else` branch
   after `if isinstance(dt_object, datetime.datetime):` is unreachable: `dt_object` is
   always the direct return value of `datetime.datetime.strptime(...)` on the line
   immediately above, which either returns a `datetime` instance or raises — it can
   never reach the `isinstance` check as a non-`datetime` value.
2. **`core/signals.py:272,274`** (`except User.DoesNotExist:` header at 272, `return`
   body at 274 — line 273 is a comment, not a counted statement) — inside
   `create_folder_for_animal_on_change`, this handler cannot fire for a `None` owner:
   `instance.owner` (line 271) on a null-FK field evaluates to `None` (not a
   `DoesNotExist` exception), so the very next line (`user.username`, line 275) raises
   `AttributeError` first, before the `except` clause it's paired with could ever catch
   anything.

**Decision (user, explicit, 2026-07-28)**: stay test-only for the `core/signals.py`
coverage CR — both branches left as documented, deliberate coverage gaps rather than
chased with contrived tests. **Confirmed (2026-07-28, coverage CR landed):**
`core/signals.py` is at **98% (180 statements, 3 missing: lines 193, 272, 274 — exactly
these two dead branches, nothing else)** — case-by-case-eligible for the `ruff.toml`
per-file-ignores removal, same as its two existing precedents (`core/models.py` 98%,
`custom_template_tags.py` 94%).

- Trigger: dedicated cleanup CR (can be combined with the bug-fix CRs above, since both
  touch the same functions) — remove the dead `else` at line 192-193; add an explicit
  `None`-owner guard after line 271 (`if user is None: return`) so the paired `except
  User.DoesNotExist` either becomes reachable via a correct check or gets removed as
  dead defensive code, whichever the professor prefers.

## TODO — no mutation testing yet (raised 2026-07-28)

Raised by the professor while the `core/admin.py`/`core/signals.py` coverage push was
in progress. Confirmed: no `mutmut`/`cosmic-ray` in `pyproject.toml` — no automated
mutation-testing tool is set up in this project. The only precedent in this repo is a
one-off manual technique, not a tool: `docs/changes/2026-07-26-auth-hardening-test-
coverage-gaps.md` used "empirical mutation probes" via a Fable5 adversarial trace (11
manual probes against live Django/allauth source) to verify specific tests genuinely
fail when the property they claim to test is removed — a one-time verification, not
repeatable infra.

Per this project's own `python-testing.md` convention: mutation testing runs *after* a
test suite exists, not before. The pre-refactor coverage initiative (`core/admin.py`
done; `core/signals.py` in progress; `core/utils.py`/`core/views.py`/`webibex/urls.py`
queued next in dependency order) is actively building that suite now — a real
mutation-testing pass makes most sense as its own follow-up once that initiative wraps.

**Decision (user, explicit, 2026-07-28)**: track as a TODO, sequenced between the
coverage initiative and the refactor it's gating — NOT deferred indefinitely. Updated
2026-07-28 (same day, user clarified sequencing): mutation testing is a **hard gate
before the refactor starts**, not a someday-follow-up. High line coverage alone doesn't
prove the tests catch real regressions once files start moving/changing shape — a
mutation-testing pass is what verifies that. Sequencing is now:

1. Coverage initiative (in progress): `core/admin.py` done, `core/signals.py` in
   progress, `core/utils.py`/`core/views.py`/`webibex/urls.py` queued next in
   dependency order.
2. Install `mutmut`, run against every file the coverage initiative touched.
3. Triage survivors (killable → write a test; equivalent → note why; NR → low-value,
   note why) per `python-testing.md`'s workflow.
4. Only then does the planned refactor start.

**Prime candidates already flagged during coverage work** (branch-dense functions,
newly tested this session — good first mutation-testing targets): `core/admin.py`'s
`CustomFolderAdmin.tag_left`/`tag_right`/`tag_other` (loop-completeness mutants),
`core/signals.py`'s `create_folder_for_animal_on_change` (L/R/O branch matrix + two
filename-parts branches) and `get_decimal_from_dms`/`extract_gps_coords`
(SIGN/BOUNDARY/FORMAT arithmetic mutants — see the `get_decimal_from_dms` `TypeError`
finding above, which is exactly the kind of contract violation mutation testing is
designed to surface). **Addendum (2026-07-30, `core/utils.py` coverage CR):**
`process_horn_chip`'s cloud branch (`POSTGRES_LOCALLY=True`) side="R" flip variant
(mirroring `mirror_coordinate` calls under the cloud, not local, storage path) was
deliberately left untested this session — a locked decision scoped T09 to the
local-branch side="L"/"R"/"O" flip matrix only, to keep the CR bounded. It's a good
mutation-testing candidate once the tool is installed: a mutant that flips
`image.side == "R"` to always/never true under the cloud branch would currently
survive.

- Trigger: coverage initiative completes (all deferred-in-`ruff.toml` files at
  100%/documented-ceiling) → install `mutmut` (per `python-testing.md`'s "mutmut for
  new projects" guidance) → run against every newly-tested file, starting with
  `core/admin.py` and `core/signals.py` → triage survivors → THEN start the refactor.

## TODO — `generate_animal_id_code`: >999 rollover collision, no prefix scoping, first-3-digit-run misparse (found 2026-07-30)

Surfaced by code-analyst while scoping the `core/utils.py` coverage CR (3rd file in
the pre-refactor test-coverage push, after `core/admin.py` and `core/signals.py`).
`core/utils.py`'s `generate_animal_id_code` (~line 170 onward) has three distinct
issues, all in the same ~20-line function:

1. **`>999` rollover collision** (line 186-187): `id_number = max(matched_numbers) + 1
   if matched_numbers else 1` followed by `new_code = f"{prefix}_{id_number:03}"`. The
   `:03` format spec is a *minimum* width, not a truncation — once `id_number` reaches
   1000, `f"{1000:03}"` renders as `"1000"` (4 digits), identical to how a
   pre-existing `"PN24_1000"` row would already render. Seeding both `"PN24_999"` and
   `"PN24_1000"` and calling `generate_animal_id_code("PN24_---_....jpg")` returns
   `"PN24_1000"` again — a genuine duplicate `id_code` in the `Animal` table, not just
   a display quirk (confirmed via `Animal.objects.filter(id_code=result).exists()`).
2. **No prefix scoping** (line 176): `Animal.objects.filter(id_code__contains="_")`
   pulls every underscore-containing `id_code` in the entire table, regardless of the
   new filename's location/year prefix. A row like `"ZZZZ_050"` (a different
   location/year) contributes to the `max()` used for a `"PN24_..."` generation,
   producing `"PN24_051"` instead of `"PN24_001"`.
3. **First-3-digit-run misparse** (line 180-185): `re.findall(r"\d{3}", i)` returns
   *every* non-overlapping 3-digit run in the id_code string, and `i[0]` (not `i[-1]`)
   is used — the *first* match, not the counter suffix. For an id_code like
   `"PN2024_001"`, the year digits `"2024"` produce a 3-digit match `"202"` before the
   scan ever reaches the real counter `"001"`; `"202"` (not `"001"`) is what feeds the
   `max()`, so the next generated code is `"PN2024_203"`, not `"PN2024_002"`.

**Backward-compatible fix direction** (no DB migration needed — `id_code` stays a
plain `CharField`): anchor the digit extraction at the *end* of the string instead of
scanning for any 3-digit run, e.g. `code.split("_")[-1]` (if the suffix is guaranteed
numeric) or `re.search(r"(\d+)$", code)`; and scope the queryset with
`id_code__startswith=prefix` instead of the blanket `id_code__contains="_"` to fix the
no-prefix-scoping issue. The `>999` rollover needs an explicit decision (reject/renumber/
widen the field) since `Animal.id_code` is `max_length=10` (`core/models.py:17`) and a
padding scheme change could affect existing sort/display assumptions.

**Decision (this CR, test-only by design, 2026-07-30)**: pin current behavior via
`pytest.raises`-free assertions on the literal (buggy) output — `test_generate_animal_id_code_rollover_collision`
(plus its just-under-rollover boundary control) and the parametrized
`test_generate_animal_id_code_known_bugs` (rollover simple form, first-3-digit-run
misparse, no prefix scoping) in `tests/core/test_utils_db.py` — not fixed inline.
`logger.debug()` instrumentation for this function is explicitly deferred to a
separate future change, not part of this CR.

- Trigger: dedicated bug-fix CR — decide the prefix-scoping and rollover-format
  scheme with the professor (reject id_codes >999? widen to 4 digits? per-prefix
  counters?) before implementing; touches the same function as the mutation-testing
  TODO above, could be combined with that gate.

**Professor answered on scope, not on scheme (2026-08-08 email reply)**: from her
side, 999/year is "molto più che sufficienti" for actual animal counts per prefix,
and no decision is needed from her — she'll keep prefix usage under that cap
herself. She's also still reconsidering what a "prefix" (`GP` etc.) should mean
(currently more of a per-project label than a literal geographic region). **User
override (2026-08-08, this session)**: the dev still considers >999 support worth
doing if it's cheap, regardless of the professor's "not necessary" read — noted as
a live priority, not dropped.

**Scope split worth knowing before picking this up**: the prefix-scoping bug and
the first-3-digit-run misparse are both plain logic fixes, no schema change (per
the backward-compatible fix direction above). The `>999` rollover fix, if done via
widening `id_code` past `max_length=10`, is the *same* migration already gated at
"GATE — restore drill required before the id_code max_length migration ships"
above — and that GATE is currently blocked: the professor's backup answer
(2026-08-08, same email) was "revisit when opening to other users," not "proceed
now," so the restore-drill precondition isn't going to be satisfied imminently.
A `max_length` widen could ship now without conflict; deploying it to production
would not, per the standing 2026-07-31 GATE decision. Reject/renumber schemes
that stay within `max_length=10` (e.g. per-prefix counters, or simply erroring
past 999 instead of colliding) would sidestep the migration/GATE entirely — worth
weighing against a genuine widen when this is scoped.

## TODO — dead code and missing guards in `process_horn_chip` (found 2026-07-30)

Surfaced by code-analyst while writing the `core/utils.py` coverage CR's test spec.
`process_horn_chip` (core/utils.py:345-483) has four separate findings:

1. **Dead `chip_url` assignments** (lines 359 and 479): both branches compute
   `chip_url` (local: `os.path.join(os.path.split(image.url)[0], chip_name)`; cloud:
   `ibex_chip.file.url`), but the variable is never read again in the function or
   returned to the caller — the assignment executes (counts as "covered" by coverage.py)
   but has no effect. Not a coverage gap, a dead-code smell.
2. **Commented-out `b2_utils.delete_files` call** (lines 398-399, inside the `if
   file_exists:` block): the very next `print()` (lines 400-402) claims `"File
   {chip_bucket_path} deleted from B2 bucket and IbexChip deleted from the
   database"` — but only the DB row is actually deleted; the B2 file deletion call
   itself is commented out. The log message is factually false, and every replaced
   cloud chip leaves an orphaned file on Backblaze B2. Pinned via
   `test_process_horn_chip_cloud_replaces_existing_chip_no_b2_delete`'s
   `delete_mock.assert_not_called()`.
3. **Missing `None`-guard asymmetry** (line 415 vs. `embed_new_chip`'s line 274-276):
   `img_object = b2_utils.download_file(...)` in `process_horn_chip`'s cloud branch has
   no `if img_object is None: raise ValueError(...)` guard, unlike `embed_new_chip`'s
   equivalent call, which explicitly checks and raises `ValueError("Failed to download
   image from Backblaze B2.")`. The asymmetry means `process_horn_chip` raises a raw,
   un-domained `TypeError: a bytes-like object is required, not 'NoneType'` from
   `np.frombuffer(None, ...)` (line 417) instead of a clear domain error. Pinned via
   `test_process_horn_chip_cloud_download_returns_none_raises_type_error`. Related: an
   empty-bytes `b""` download (distinct from `None`) hits a *third*, uncaught failure
   mode — a raw `cv2.error` (`!buf.empty()` assertion) from `cv2.imdecode`, before the
   existing `img is None` check at line 422 is ever reached — pinned via
   `test_process_horn_chip_cloud_empty_bytes_raises_cv2_error`.
4. **File/row desync produces two different failure modes** (local branch): if the
   chip *file* exists on disk but no matching `IbexChip` DB row exists,
   `get_object_or_404(IbexChip, ibex_image_id=image.id)` (line 365) raises `Http404`
   (pinned via `test_process_horn_chip_local_file_without_row_raises_404`). If the DB
   *row* exists but the chip file is missing, the `chip_path.is_file()` guard (line
   363) is `False`, so the function skips straight to `IbexChip.objects.create(...)`
   (line 461) — which collides with the OneToOne `ibex_image` constraint on the
   already-existing row, raising `IntegrityError` (pinned via
   `test_process_horn_chip_local_row_without_file_raises_integrity_error`). Same
   underlying "state inconsistency" bug, but the caller sees a 404 or a 500 depending
   on which side of the (file, row) pair is missing.

**Fix direction**: (1) remove the two dead `chip_url` assignments, or wire the value
through to whatever was meant to consume it; (2) either restore
`b2_utils.delete_files([chip_bucket_path])` (with B2 test coverage added) or correct
the log message to stop claiming a deletion that never happens; (3) add the missing
`None`-guard mirroring `embed_new_chip`'s, and decide whether `b""` should be
special-cased before it reaches `cv2.imdecode`; (4) decide the desired file/row-desync
behavior (always 404? an automatic repair path?) and implement it consistently across
both the local and cloud branches.

**Decision (this CR, test-only by design, 2026-07-30)**: all four findings pinned via
`pytest.raises`/negative mock assertions in `tests/core/test_utils_process_horn_chip.py`,
none fixed inline. Do not uncomment `b2_utils.delete_files` without updating the
pinning test (`delete_mock.assert_not_called()`) to the opposite assertion first.

- Trigger: dedicated bug-fix/cleanup CR — touches both the local and cloud branches
  plus B2 bucket cleanup semantics; decide desired behavior with the professor before
  implementing. Could combine with the mutation-testing TODO's flagged candidates
  once `process_horn_chip` itself becomes a mutation-testing target.

## TODO — `parse_coordinates` validates request input with bare `assert` (found 2026-07-30)

Surfaced by code-analyst while writing the `core/utils.py` coverage CR's test spec.
`core/utils.py`'s `parse_coordinates` (lines 36-45) validates the incoming request's
query-string coordinates with two bare `assert` statements: `assert len(keys) == 1`
(line 38) and `assert len(coordinates.split(",")) == 2` (line 41). Per
`python-security.md`'s Dangerous Patterns table, `assert` in production code is a
double risk on untrusted input:

1. **Stripped by `python -O`**: running Python with optimizations removes all `assert`
   statements entirely — the validation silently disappears, and malformed input
   (zero or multiple query keys, malformed key format) flows straight into
   `coordinates.split(",")`/`int(x)`/`int(y)` unguarded.
2. **Wrong error surface**: even without `-O`, an `AssertionError` is an uncaught
   exception from Django's perspective — it surfaces as a 500 Internal Server Error,
   not the 400 Bad Request that malformed *client* input should produce. The
   function's own comment (line 35) already acknowledges this: `"will crash server if
   they are coming in unexpected format"` — a known, accepted design gap from the
   original author, not something introduced by this CR.

Pinned via `test_parse_coordinates_assertion_errors` (zero keys, two keys, malformed
single-key splits → `AssertionError`, no `match=` since the message is empty) and
`test_parse_coordinates_value_error` (non-integer values → `ValueError`) in
`tests/core/test_utils_pure.py`.

**Decision (this CR, test-only by design, 2026-07-30)**: pin current behavior only —
both asserts and the resulting 500-on-malformed-input behavior left in place.

- Trigger: dedicated bug-fix CR — replace both bare `assert`s with explicit
  `if not (...): raise ...` producing a clear 400-mapped error (e.g. `Http404` or a
  dedicated `BadRequest`-mapped exception), and decide the desired client-facing error
  contract (JSON error body vs. redirect) with the professor before implementing,
  since this function backs the landmarking UI's coordinate submission endpoint.

## TODO — "animals" tab naming, undecided (raised 2026-08-08, professor undecided)

Minor UX question raised while confirming other open items with the professor by
email: the "animals" tab (dashboard view for browsing already-identified
individuals as a catalogue) may be better named — "animals" feels potentially
confusing to her, "catalogue" was floated as an alternative but not settled. No
code implications either way, purely a label. Professor is still thinking about it.

- Trigger: professor confirms a preferred name, or this comes up again during a
  dedicated UX/naming pass.

## TODO — migrate `testcontainers.postgres` → `testcontainers.community.postgres` (found 2026-08-10)

Surfaced while resolving the `db-restore-drill` docker-run-wiring plan's blocking
container-id-accessor question: installing `testcontainers[postgres]==4.15.0` into
the sandbox venv and importing `testcontainers.postgres.PostgresContainer` (the
module `scripts/db_restore_drill.py` lazy-imports) emits a `DeprecationWarning` —
the package has moved this import to `testcontainers.community.postgres`. Still
fully functional in 4.15.0, no behavior change, explicitly left alone in the
docker-run-wiring CR (out of scope for that change) — see
`docs/changes/2026-08-09-db-restore-drill.md`.

- Trigger: next `testcontainers` version bump, or a dedicated dependency-hygiene
  pass — swap the import path in `scripts/db_restore_drill.py`'s lazy import,
  confirm no API surface changed between the two modules, re-run
  `tests/scripts/test_db_restore_drill_restore.py`.

## TODO — upgrade prod Postgres 16.13 → 17.9, to match tmgame (found 2026-08-10, discussed 2026-08-11)

Originally surfaced 2026-08-10 while confirming the docker-run wiring's client image
(`docs/changes/2026-08-09-db-restore-drill.md`): prod's Railway-managed Postgres is
confirmed 16.13 (16.14 available), while `tmgame` runs 17.9. Not scoped or started at
the time.

**2026-08-11 addition**: raised again with a concrete rationale — keeping a single
Postgres major version across projects (webibex, tmgame) simplifies shared DevOps
workflows (client tooling, backup/restore scripts, image pinning). Still genuinely
bigger than it sounds: this is an actual production database engine upgrade on
Railway (not just a client-tooling tag bump), likely needs either an in-place
upgrade path or a dump/restore migration to a new instance, plus a Django/psycopg2
compatibility check.

**Sequencing note, this session**: the restore-drill GATE-evidence live run (proving
backup/restore works against the *current* prod, 16.13 — see the GATE section above)
had not yet run when this came up. Explicitly not decided whether the PG17 upgrade
should wait for that run to complete first, or be scoped independently — deferred
along with the rest of this TODO, not a ruling either way.

- Trigger: next time this is deliberately picked up for scoping — research Railway's
  major-version upgrade mechanism, decide sequencing relative to the restore-drill
  GATE work, confirm Django/psycopg2 compatibility with PG17.

## TODO — containerize `scripts/db_restore_drill.py` itself (raised 2026-08-11)

Once the tool is proven working via the host-`uv`-venv path (`scripts/run_db_restore_drill.sh`
once promoted out of `tmp/`), package it as its own container image for reproducibility
across machines — matches the DHI-hardening pattern already used for the
`dhi.io/postgres:16-alpine-dev` client image.

**Proposed structure** (user, 2026-08-11):
```
lev_root: orchestrator script (host) calls:
  lev_1a: Python-deps container (psycopg2, testcontainers, requests --
          eventually its own DHI Python base image)
  lev_1b: dhi.io/postgres:16-alpine-dev (already exists -- the pg_dump/
          pg_restore client image the script already wraps in `docker run`)
```

**Architectural wrinkle, inherent to this structure, not a separate concern**:
`lev_1a` is where `db_restore_drill.py`'s own process runs, and that process is the
thing issuing the `docker run` calls against `lev_1b` *and* driving `testcontainers`
to spin up the ephemeral Postgres server -- so `lev_1a` containerized still needs a
way to reach the Docker daemon to create `lev_1b`-family containers. This is the
"sibling containers" pattern (`-v /var/run/docker.sock:/var/run/docker.sock` + the
`docker` CLI binary inside `lev_1a`, reaching the *host* daemon directly) -- a
well-trodden pattern (how many CI systems run Docker jobs), not exotic. `dind`
(nested daemon, `--privileged`) is the alternative and the worse fit here (worse
security posture, heavier, storage-driver gotchas). Either way, a container with the
host socket has effectively host-root-equivalent reach -- a real tradeoff to weigh
explicitly, given how carefully this script otherwise scopes its
credential/subprocess/network boundaries (see the docker-run wiring CR,
`docs/changes/2026-08-09-db-restore-drill.md`). `--network container:<id>` for the
restore leg should still resolve correctly under this setup (the host daemon
resolves it, not `lev_1a`'s own netns) -- nothing about the current design needs to
change, just this one tradeoff made on purpose.

- Trigger: after the host-venv path is proven working end to end (live GATE-evidence
  run passes) -- scope `lev_1a`'s Dockerfile + `lev_root`'s orchestrator script,
  decide the socket-mount tradeoff explicitly, route through the planning-TDD
  pipeline given the security-critical surface.

## TODO — e2e test: railway-like container against the restored local Postgres (raised 2026-08-11)

Today's restore-drill verifies the restored database at the *data* level only
(row counts on 6 tables + one `Animal` spot-check row). It does not verify
that the actual webibex Django app can run against that restored database --
migrations applying cleanly against the restored schema state, the app
successfully booting and connecting, ORM queries actually working (not just
raw row counts via `psycopg2`). A restored DB that passes today's checks
could still fail to serve the real app if e.g. a Postgres extension is
missing, permissions differ, or migration state is inconsistent.

Idea (user, 2026-08-11): spin up a container mimicking the Railway deployment
environment (the actual webibex app image, or something close to it) and
point it at the ephemeral `testcontainers` Postgres once
`restore_and_verify` has restored it into that container -- run the app for
real against the restored data as the final, strongest verification step.

- Trigger: next deliberate scoping pass on `scripts/db_restore_drill.py` --
  decide how "railway-like" the container needs to be (the real Docker image
  webibex deploys with, if one exists / gets built per the
  containerization TODO above, vs. a lighter Django-only smoke container),
  what a minimal "app actually works" check looks like (management command,
  a single ORM query, a health-check endpoint), and whether this becomes a
  4th restore-drill checklist item or stays a separate, optional deeper
  verification tier.

## TODO — add progress/debug logging to `scripts/db_restore_drill.py` (raised 2026-08-11)

During a future refactor pass, add stage-level progress logging through the
pipeline -- e.g. "railway db connection established", "railway db read start",
"railway db dump ended", equivalent markers for the restore leg. Currently the
script is silent until the final PASS/FAIL report (`_print_report`); a long-running
live invocation gives no feedback on which stage it's in. Should route through the
existing `_emit()` helper (the documented `print()` exception already used for
user-facing CLI output) rather than introducing a separate logging mechanism --
consistent with the script's own established pattern.

- Trigger: next deliberate refactor pass on this script -- not urgent, purely
  observability/UX, no functional change.
