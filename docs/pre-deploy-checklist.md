# Pre-deploy checklist — professor-only pre-release

Scope: the release currently being prepared is a **pre-release to the
professor only** (`wibex.up.railway.app`), not yet opening to other users —
confirmed 2026-08-08 (`docs/session-notes-2026-08-08-webibex-followups.md`
"Release-scope clarification"). This checklist covers what's actually
blocking *that* release. A separate, stricter pass is needed before opening
to any wider user base — see "Explicitly deferred" at the bottom, not to be
treated as done just because this checklist is green.

Modeled after tmgame's `pre-push-checklist.md`, adapted for webibex's actual
stack: Railway (Nixpacks auto-build, no Dockerfile/CI in this repo), Django,
Postgres, Backblaze B2, RunPod inference.

## TODO — release blockers, must land before the professor sees this build

Concrete "not just committed locally" items. Pushing to `origin_gitlab`
(the GitLab mirror, `git@gitlab.com:aletrn/webibex.git`) does **not**
satisfy this — confirmed 2026-08-14: that push landed (mirror now 1 commit
behind), but `origin` (`git@github.com:laubohlen/webibex.git`, the repo
Railway almost certainly builds from) is still **39 commits behind,
unchanged**. Nothing below is live until it reaches `origin`.

- [ ] **Push local `main` to `origin` (github.com/laubohlen/webibex)**, not
      just the GitLab mirror — see §0 below for the access/trigger caveats.
- [ ] **Delete-menu-option fix** (`3d5168d`, guards the Tools-menu Delete
      crash and hides the option until real semantics are decided) —
      committed locally, confirmed correct by the professor 2026-08-08, but
      not deployed. Currently-live code (whatever's on `origin`/Railway
      today) still has the unguarded crash.
- [ ] **Already-fixed dependency CVEs** (Django 5.0.14→5.2.16, pillow,
      django-allauth, django-filer, lxml, requests, setuptools — commit
      `480607b` and later) — done locally, not deployed. Currently-live
      production is still running the pre-bump, CVE-exposed versions.
- [ ] **Still-open critical dependency risk, decide before shipping**:
      `urllib3==1.26.20` (CRITICAL, multiple open CVEs) / `boto3==1.26.0` /
      `botocore==1.29.165` — deliberately pinned old for Backblaze B2
      compatibility, blocked on a dedicated B2 test bucket, genuinely not
      fixed anywhere (local or deployed). Either accept explicitly for this
      release or treat as a hard blocker — don't let it ship as an
      unexamined default.
- [ ] Auth/session hardening settings (`SESSION_COOKIE_SECURE` etc.,
      `webibex/settings.py`, 2026-07-25) — same story, done locally, not on
      `origin`/Railway yet.
- [x] **Unauthenticated-reachable views** (found 2026-08-14, full detail in
      `security-remediation-plan.md`'s IDOR section): `save_landmarks_view`,
      `results_over_view`, `default_chip_compare_view`,
      `project_chip_compare_view`, `geographic_chip_compare_view`,
      `rerun_view` (`core/views.py`) all lack `@login_required`, and no
      global auth middleware covers the gap. `save_landmarks_view`
      specifically also triggers a real billed RunPod inference call with no
      login and no ownership check on the `image-id` it's given. Unlike the
      other deferred items below, this isn't gated by "how many users we
      open the app to" — it's reachable by anyone on the internet the moment
      the app is live, regardless of release scope. Treat as blocking this
      release, not the next one.
      **Re-verified 2026-08-14: no legitimate reason for any of the 6 to be
      unauthenticated** — traced every template that links to each one
      (`unidentified_images.html`, `animal_images.html`,
      `animal_images_owner.html`, `multi_landmarking.html`, `header.html`'s
      nav), and every single entry point is already gated (`@login_required`
      on the linking view, or `{% if user.is_authenticated %}` in the nav).
      `rerun_view` has no entry point at all (dead-linked). No demo/preview
      flow depends on anonymous access anywhere — `welcome_view` is the only
      view meant to stay public.
      **Routing through the full planning-TDD pipeline** (code-planner →
      code-analyst → code-executioner), not applying directly — the fix
      itself is mechanical (6 one-line `@login_required` additions, copying
      an existing pattern used 20+ times elsewhere in the same file), but
      it's a security boundary change and needs real regression tests
      alongside it: each view must be proven to reject anonymous access
      (302/403, not the crash it'd otherwise be), proven to still work for
      an authenticated user, and — for `save_landmarks_view` specifically —
      proven the RunPod-triggering path is actually blocked pre-auth, not
      just visually gated. Simplicity of the diff doesn't reduce the test
      bar here. **To be implemented in a devcontainer-guard (dcg) session,
      not on the host** — this batch will also need a project version bump,
      which the user doesn't do on the host.
      **Fix landed (2026-08-14)**: all 6 views now carry `@login_required`
      (`core/views.py`), added bottom-up by line number, zero other lines
      touched. New test file `tests/core/test_views_auth_required.py` (46
      tests, T01-T25 per the planning-TDD spec) — anonymous access proven
      302/redirected for all 6 (never 200, never a pre-existing-bug 404),
      the RunPod/no-network R2 proof for `save_landmarks_view` specifically,
      authenticated behavior proven unchanged, and the two pre-existing bugs
      (`rerun_view`'s `TemplateDoesNotExist`, `save_landmarks_view` GET's
      `ValueError`) pinned as still-present and explicitly out of scope.
      Full suite: 325 passed (279 pre-fix baseline + 46 new), 1 skipped, 1
      xfailed — zero regressions. Manual mutant-matrix (each decorator
      removed one at a time, tests re-run, decorator restored) confirmed
      every one of the 6 is independently covered — see
      `docs/changes/2026-08-14-login-required-unauthenticated-views.md` for
      the per-view kill list. **Version-bump note**: the project-version-bump
      step mentioned above for this batch was explicitly deferred by the
      user — this repo has no `VERSION` file or `pyproject.toml` yet (only
      `requirements.txt`); a proper `pyproject.toml` (which would carry a
      `[project.version]`) is planned as separate future work, not part of
      this fix. Still not pushed to `origin` — see the blocker note at the
      top of this section.

## 0. Push to origin (webibex-specific — do this check first)

- [ ] **Confirm current ahead/behind against `origin/main`** (`git@github.com:laubohlen/webibex.git`,
      not the GitLab mirror): re-check with
      `git rev-list --left-right --count origin/main...main` — was 39/0 as of
      2026-08-14, unaffected by the same-day GitLab-mirror push.
- [ ] **Confirm push access to `laubohlen/webibex`** — repo is owned by the
      original developer (Lauren), not this account; verify write access
      exists before assuming a push will succeed.
- [ ] **Confirm Railway's actual deploy trigger** — verify in the Railway
      dashboard whether it auto-deploys on push to this branch, or is a
      manual deploy step, before assuming "push" == "live." Not confirmed
      from repo content alone (no `railway.json`/`railway.toml` in this repo).
- [ ] Git-signing note: commits made from a sandboxed devcontainer have
      previously failed to sign (`commit.gpgsign=true`, Secretive SSH path is
      macOS-host-only) — push/commit from the host if this recurs.

## 1. Known bug — dead-linked, not a release blocker

- [ ] **`rerun_view` 500s** (`core/views.py:440`, confirmed still present):
      renders `"core/result.html"`, which doesn't exist as a file (only
      `result_default.html`/`result_refined.html` do) — throws
      `TemplateDoesNotExist`. **Correction (2026-08-14): confirmed
      unreachable from the UI** — `grep -rln "run-again\|run_again"
      templates/` returns nothing, no template links to it. Only hit by
      manually typing `/run_again/<oid>/`. Not blocking this release;
      cheap opportunistic fix (point at `result_default.html`/
      `result_refined.html`, whichever context shape matches) or delete
      the dead route (view's own docstring flags it as never-finished).

## 2. Backend tests / lint / type checks

- [ ] `python manage.py test` (or `pytest`, per `pytest.ini`) — full suite green
- [ ] `ruff check` clean (curated ruleset, `ruff.toml`, gated per-file to
      100%-coverage files — check no new file needs adding to
      `per-file-ignores` or graduating out of it)
- [ ] `pyright` (`pyrightconfig.json`, basic mode) clean
- [ ] `manage.py check --deploy` — Django's own deploy-readiness check;
      referenced in `security-remediation-plan.md` as never actually run
      (T06 was skipped as "manual verification, not scripted")

## 3. Database

- [ ] **Migrations**: `manage.py showmigrations` — confirm nothing pending
      against prod
- [ ] **Restore-drill GATE**: already satisfied — real PASS run 2026-08-11
      (`docs/session-notes-2026-08-11-db-restore-drill.md`), all 6 table row
      counts + `Animal` spot-check matched. Re-run if the schema changes
      before this deploy.
- [ ] **Backup artifact exists somewhere durable**: encrypted dump uploaded
      to project kDrive 2026-08-11 — confirm it (or a fresher one) still
      exists before deploying schema changes.

## 4. Production env vars actually set on Railway

Not just documented — verify in the Railway dashboard itself:

- [ ] `SECRET_KEY`, `DATABASE_URL` (+ `DATABASE_PUBLIC_URL` if the restore
      drill needs external access again)
- [ ] `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` / `AWS_S3_ENDPOINT_URL` /
      `AWS_STORAGE_BUCKET_NAME` / `AWS_S3_REGION_NAME` (B2)
- [ ] `RUNPOD_ENDPOINT_ID` / `RUNPOD_API_KEY`
- [ ] `EMAIL_ADRESS` / `EMAIL_HOST_PASSWORD` (note: typo'd var name is real,
      not a mistake to "fix" without checking Railway's actual var name first)
- [ ] `ENVIRONMENT=production` (or unset — defaults to `"production"`)
- [ ] No `MAPTILER_API_KEY` needed yet — OSM-direct tile fetch is still in
      place by design (see §5 deferred items)

## 5. Static assets

- [ ] `staticfiles/admin/*` and `staticfiles/filer/*` already refreshed to
      match Django 5.2.16/django-filer 3.3.0 (`54c35d6`) — confirm no further
      `Django`/`django-filer`/`django-allauth` bump has landed since without
      a matching `collectstatic` re-run.

## 6. Security settings (already implemented — verify still live)

- [ ] `SESSION_COOKIE_SECURE`, `CSRF_COOKIE_SECURE`, `SECURE_SSL_REDIRECT`,
      `SECURE_HSTS_SECONDS=3600`, `SECURE_PROXY_SSL_HEADER` — all set under
      `ENVIRONMENT == "production"` (`webibex/settings.py`, done 2026-07-25).
      Confirm the HSTS header actually appears on a live prod response
      (`curl -I https://wibex.up.railway.app/` → `Strict-Transport-Security`).
- [ ] `ALLOWED_HOSTS`/`CSRF_TRUSTED_ORIGINS` still match the actual prod
      domain (`wibex.up.railway.app`) — check if this deploy changes the domain.

## 7. Supply chain (known open risk — accept explicitly, don't silently ship)

- [ ] `urllib3==1.26.20` / `boto3==1.26.0` / `botocore==1.29.165` are still
      **deliberately pinned old** (Backblaze B2 `x-amz-checksum-algorithm`
      incompatibility). Blocked on a dedicated B2 test bucket, not fixed.
      **Exploitability assessed 2026-08-14**: urllib3 1.26.20's real CVEs
      (CVE-2025-66418/66471, CVE-2026-21441 — decompression-bomb/resource
      exhaustion; CVE-2026-44431 — cross-origin header leak on redirect via
      proxy; CVE-2025-50181 — open redirect) all require the attacker to
      control either the HTTP *response* or a proxy in the request path.
      Checked both places urllib3 is actually reached (`core/b2_utils.py:39`
      via boto3, `core/utils.py:247` via `requests.post`) — both hit fixed,
      developer-configured endpoints (B2, RunPod), no proxy config, TLS
      verification on by default, no user-supplied URL ever reaches either
      call. Practical exploitability against webibex itself: **low** —
      would need B2 or RunPod compromised/MITM'd, not something reachable by
      hitting webibex directly. Still worth bumping when unblocked (defense
      in depth, and the CVSS ratings are real for the general case), but not
      the most urgent open item — the unauthenticated-view gap above is.

## Explicitly deferred — NOT required for this professor-only pre-release

Confirmed 2026-08-08 that these are gated on the *next* release (opening to
other users), not this one — listed here so they aren't mistaken for
deploy blockers:

- IDOR fix (`location-id`/`oid`/`region_id` ownership checks in
  `save_image_location`/`create_loaction`)
- Region coordinate cross-owner exposure UX (poaching-sensitivity question,
  still open with the professor)
- MapTiler swap (OSM ToS) — headroom confirmed 6-10x at current scale
- `id_code` >999 rollover + collision-fix migration — professor said not
  necessary at current scale; migration itself is separately gated on the
  DB-backup-mechanism decision
- `allauth.mfa` evaluation — not decided whether wanted at all
- CI scaffold — repo has zero CI today, deliberately deferred
- Railway hardened-base-image / pinned Dockerfile — Nixpacks auto-build
  stays as-is for now
- `ruff.toml` stale `per-file-ignores` cleanup (2 files already at 100%,
  not yet removed from the exemption list) — cosmetic, not urgent

## Notes

- No staging environment exists — localhost is canary for prod.
- No CI/Dockerfile in this repo; Railway's Nixpacks auto-detects the build.
- Threat model at this scale: ~1 trusted user (the professor), soon
  20-50 trusted researchers on the next release — not public-facing yet.
