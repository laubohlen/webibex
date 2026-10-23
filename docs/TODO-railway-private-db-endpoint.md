# TODO — Switch the web service's database connection to Railway's private endpoint

Raised 2026-10-22 (after the Railway env-var exposure incident). Status: **planned, not started**.
Facts below come from the session; nothing here has been tested.

## Why

- Railway warns that `DATABASE_PUBLIC_URL` references the public TCP proxy
  (`RAILWAY_TCP_PROXY_DOMAIN`): connecting through it incurs egress fees.
- The web service currently uses a pasted `DATABASE_URL` that points at the public proxy
  (`autorack.proxy.rlwy.net:<port>`), so the database is reachable from the internet and every
  app query is egress traffic.
- The Postgres service already defines a private `DATABASE_URL`
  (`…@${{RAILWAY_PRIVATE_DOMAIN}}:5432/${{PGDATABASE}}`), which avoids both problems.

## Precondition to verify first

- Is the web service in the **same Railway project and environment** as the Postgres service?
  Private networking and reference variables (`${{<service>.DATABASE_URL}}`) only work inside one
  project/environment. If they are separate projects, this plan needs a different route
  (move the database into the web project, or keep the public endpoint and accept the fees).
  Unknown as of 2026-10-22.

## Plan (after the app is back up on the pasted URL)

1. In the web service variables, replace the pasted `DATABASE_URL` with the reference
   `${{<Postgres service name>.DATABASE_URL}}` (use the name shown in Railway's reference picker).
2. Deploy once; check the deploy logs for database authentication errors and load a DB-backed page.
3. Find every other consumer of the public URL (`DATABASE_PUBLIC_URL`, `SOURCE_DSN` in the dump and
   restore scripts, local tools). Update them or note why they still need public access.
4. If nothing needs the public endpoint, disable the TCP proxy on the Postgres service. If dumps
   still need it, keep it only while they run. I have not verified whether Railway can restrict the
   proxy by source address.
5. `scripts/db_url_to_clipboard.sh` is then only needed for local tools; the web service no longer
   depends on a pasted password.

## Rollback

Put the previous pasted URL back (compose it with `scripts/db_url_to_clipboard.sh`) and redeploy.

## Related

Railway security incident, 2026-10-21 (env vars possibly read by a third party). Same-day credential
rotation: database password, B2 key, `SECRET_KEY`, RunPod key, email password.
