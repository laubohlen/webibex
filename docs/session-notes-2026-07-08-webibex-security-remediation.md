# Session notes — 2026-07-08 — webibex security remediation

Written to `ibex_stambecchi/docs/` because the `webibex` working directory was mounted
read-only for this session (planned to be RW next session).

## Context

Continuing the security remediation plan already reconstructed at
`webibex/docs/security-remediation-plan.md` (see that file for the full CVE audit,
boto3/urllib3 pinning rationale, and OSM ToS findings). This session verified that plan's
findings still hold against current repo state (same pinned versions, no CI, TF1.15
SavedModel and `core/test_model.py` both still present) and moved into planning execution
for the four unblocked items.

## Scope decided this session

Batch to execute now: **CI scaffold, routine CVE bumps, TensorFlow local-dev cleanup,
JS build-tooling cleanup**. Excluded/blocked: boto3/botocore/urllib3 triangle (needs a
dedicated B2 test bucket, unchanged status) and modernizing the actual RunPod-hosted
embedding model to TF2/PyTorch (new idea raised this session, see below) — blocked on
RunPod account access.

## Key finding: local-dev TensorFlow branch is already dead code

`core/utils.py:283-285`:
```python
model_is_local = not (settings.ENVIRONMENT == "production" or settings.ENDPOINT_LOCALLY == True)
```
`ENDPOINT_LOCALLY` is hardcoded `True` at `webibex/settings.py:141`, so the `or` is always
`True` and `model_is_local` evaluates to `False` in every environment — the two
TF-based branches in `embed_new_chip()` are unreachable today regardless of the flag's
apparent intent. Additionally, `tensorflow` is not declared in `requirements.txt` (no
other requirements file exists either), so even if the boolean logic were fixed,
`get_tf()`'s `import tensorflow` would fail. This resolves a question carried over from a
prior session about whether a developer could use the local TF1 branch for local dev —
answer: not currently, on two independent counts. Deleting this code (per the plan) removes
dead code, not a working capability.

## RunPod embedding model modernization (new track, blocked)

Raised this session: since `embed_new_chip()`'s production path only calls a RunPod HTTP
endpoint (`core/utils.py:246-268`, `endpoint_inference()`), the actual TF1.15 serving code
is NOT in the webibex repo — it lives in a separate repo/image not yet accessible.

Two paths identified via `ibex_stambecchi/docs/migration-plan.md` (already-done research):
- **TF2 swap**: same model weights, already validated to load in TF2 with identical
  accuracy (91.27% mAP) — existing gallery embeddings in Postgres stay valid, no
  re-embedding needed. Estimated ~0.5-1d once the handler repo is accessible.
- **PyTorch v23b swap**: different architecture/weights (ResNet50, 92.32% mAP on holdout)
  — would require re-embedding the entire production gallery from original chips, higher
  risk/effort (~2-4d+), no production validation yet (only holdout backtest).

Decision: pursue the TF2 swap as a new remediation task once RunPod access is granted.
Draft access-request email written to
`ibex_stambecchi/tmp/draft-email-runpod-access.md` — recipient is "Lauren"
(git remote is `laubohlen/webibex`, presumably Lau Bohlen, the original author who
hardcoded the `ENDPOINT_LOCALLY` flag and the `/Users/lau/...` path in
`core/test_model.py`). User confirmed Lauren already has context and just needs to grant
access — no need to re-explain in the email.

## Deploy workflow (Railway) — confirmed this session

- No `Dockerfile`/`railway.json`/`nixpacks.toml`; Railway auto-detects Python via
  `runtime.txt` (`python-3.12.5`) and uses its default Nixpacks builder.
- `Procfile`: `web: gunicorn webibex.wsgi --timeout 120`.
- `ENVIRONMENT = env("ENVIRONMENT", default="production")` (`settings.py:22`) — Railway
  doesn't need to set this explicitly; the app defaults safely to production behavior
  (S3/Backblaze B2 storage, `DATABASE_URL`-parsed Postgres) if unset.
- Static files via `whitenoise`; DB via `psycopg2-binary` + `dj-database-url`.
- This informed the decision to use plain dummy env values (not GitHub repo secrets) in
  the new CI workflow — matches how the app already expects non-production environments
  to be configured.

## GitHub Dependabot check — inconclusive this session

`gh` CLI is not authenticated in this session (`gh auth status` → not logged in), so
`gh api repos/laubohlen/webibex/dependabot/alerts` could not be tried. Also unconfirmed
whether Dependabot alerts are even enabled on the repo (Settings → Security → Dependabot
alerts). Open item for next session if the user wants ongoing CVE tracking beyond the
one-off `pip-audit`/`npm audit` runs planned in this batch.

## Plan produced this session (code-planner, Opus)

Full ordered plan for the four-item batch (CI scaffold → CVE bumps → TF cleanup → JS
cleanup) was produced and is captured in this conversation's transcript. Four open
questions were raised to the user:

1. Django target **5.2 LTS** (5.0 is EOL) — user asked for the concrete step-by-step
   first rather than confirming outright; steps given (bump target, check
   `django-polymorphic`/`django-filer` compat, read 5.1+5.2 release notes, run CI checks,
   manually smoke-test auth/upload/compare-view). **Not yet explicitly confirmed to
   proceed** — revisit at start of next session before executing this bump.
2. `core/embedding_model/` (1.1MB TF1.15 SavedModel) — recommended deletion, confirmed OK
   by user (recoverable from git history).
3. CI dummy env vars (plain workflow-level placeholder values, not GitHub secrets) —
   user said "ok" contingent on understanding the deploy workflow first; the
   deploy-workflow finding above was given as supporting rationale, but the user moved to
   a different topic afterward without an explicit final re-confirmation. Treat as
   likely-OK, not hard-confirmed.
4. Exact fixed package versions still need a live PyPI/OSV check — blocked on egress
   being open in this sandboxed session; a temp-egress domain list (PyPI, OSV, npm
   registry, Django/allauth/filer docs, GitHub) was given to the user to open egress with
   their own mechanism. `pip-audit`-clean is the acceptance oracle regardless of which
   exact versions get pinned.

## Status / next steps

- Plan not yet executed (no code changes made to webibex this session; directory was
  read-only).
- Next session: webibex will be mounted read-write. Pick up with the code-analyst test-spec
  pass on the approved plan, then code-executioner implementation, starting with the CI
  scaffold (item 1, since it gates the CVE-bump regression check).
- Send the RunPod access-request email (draft at
  `ibex_stambecchi/tmp/draft-email-runpod-access.md`) to Lauren separately — not blocking
  the four-item batch.
