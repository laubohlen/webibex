# Deploy platform checklist — GitHub / Railway / RunPod

Recurring manual checklist for verifying the deploy platforms themselves
(as opposed to code/test readiness, which lives in
`pre-deploy-checklist.md` — that one is scoped to a specific release's
blockers and gets superseded each time; this one is meant to be reused
as-is on every deploy). Nothing here is repo-content-verifiable — every
item requires actually looking at the dashboard/CLI at deploy time, not
assuming last time's answer still holds.

First written 2026-08-14, ahead of the first `origin` push this user
personally drove for this app (login-required-views fix + the batch of
CVE bumps/auth-hardening/delete-fix that had been sitting on `main`
unpushed). Update this file directly (not via dated addenda) as the
actual platform setup is learned/changes — it's meant to converge into
an accurate reference, not accumulate history.

## GitHub (`origin` = `laubohlen/webibex`)

- [ ] `git remote -v` — confirm `origin` is `git@github.com:laubohlen/webibex.git`,
      not accidentally the GitLab mirror (`origin_gitlab`)
- [ ] Push access actually works (not just configured) — the repo is
      owned by the original developer (Lauren), not this account
- [ ] After push: open the repo on github.com, confirm `main`'s HEAD SHA
      matches the local `main` you just pushed

## Railway

**Before push:**

- [ ] Confirm which repo + branch this Railway service is tracking —
      should be `laubohlen/webibex` / `main`
- [ ] Confirm the deploy trigger: auto-deploy on push, or a manual
      "Deploy" step in the dashboard — not discoverable from repo
      content (no `railway.json`/`railway.toml` exists)
- [ ] Confirm env vars are set (verify existing values, don't create
      blind):
  - [ ] `SECRET_KEY`, `DATABASE_URL` (+ `DATABASE_PUBLIC_URL` if
        external DB access is needed, e.g. for a restore drill)
  - [ ] `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` /
        `AWS_S3_ENDPOINT_URL` / `AWS_STORAGE_BUCKET_NAME` /
        `AWS_S3_REGION_NAME` (Backblaze B2)
  - [ ] `RUNPOD_ENDPOINT_ID` / `RUNPOD_API_KEY`
  - [ ] `EMAIL_ADRESS` / `EMAIL_HOST_PASSWORD` (typo'd var name is
        correct — don't "fix" it without checking Railway's actual
        var name first)
  - [ ] `ENVIRONMENT=production` (or unset — defaults to production)
  - [ ] `MAPTILER_API_KEY` — confirm still not needed (OSM-direct
        tile fetch in use by design); add this check once the
        MapTiler swap ships
- [ ] Locate the "redeploy previous version" / rollback action in the
      dashboard now, before you need it

**During/after deploy:**

- [ ] Watch the build log live rather than assume it succeeded —
      especially valuable right after a dependency-version bump
- [ ] `curl -I https://wibex.up.railway.app/` → confirm
      `Strict-Transport-Security` header is present
- [ ] Confirm `ALLOWED_HOSTS`/`CSRF_TRUSTED_ORIGINS` still match
      `wibex.up.railway.app` (unless this deploy changes the domain)
- [ ] If Railway exposes a shell/CLI (`railway run ...`):
      `python manage.py showmigrations` — confirm nothing pending
      against prod
- [ ] Smoke test as a real logged-in account: exercise the actual
      upload/identify/landmark flow end-to-end, not just the landing
      page — catches auth-boundary or view regressions that a bare
      200-check on `/` would miss

## RunPod

- [ ] Confirm the endpoint (`RUNPOD_ENDPOINT_ID`) is active/awake, not
      paused or cold — serverless endpoints can idle out
- [ ] No separate action needed beyond that — the Railway smoke test's
      landmark-save step already exercises a real RunPod inference call
