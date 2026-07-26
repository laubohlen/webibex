# webibex Roadmap

Single entry point for this repo's active work tracks. Each track has its own
detailed plan doc (kept in place, not merged in full here — both grow with
forensic/session detail that would make one giant file harder to scan, not
easier). This doc is the short version: what each track is, current status,
and where to go for the full story.

## Track: Security & Tech Debt Remediation

Full detail: [`docs/security-remediation-plan.md`](security-remediation-plan.md)

Started 2026-07-07 — an ongoing security/tech-debt audit and remediation
effort for webibex itself (the live Django app on Railway).

**Done**: routine CVE bumps (`480607b`, 2026-07-09), JS build-tooling
cleanup, region-visibility fix (2026-07-24), auth/session hardening settings
(`95b2932`, 2026-07-25) + follow-up test coverage (`978f785`, 2026-07-26),
TensorFlow local-dev removal (`ee1839f`, 2026-07-26), `moto`-based S3-mock
test tier for `core/b2_utils.py` (`2cbfb84`, 2026-07-26) — all 4 functions
covered, verified with real execution (100% coverage on that file), plus a
`pytest_collection_modifyitems` hardening hook closing a marker-bypass gap
found during adversarial review.

**Blocked**: boto3/botocore/urllib3 version-bump triangle — needs a
dedicated B2 test bucket before it can start. The moto tier confirms
`moto==4.2.14` is compatible with the exact pinned triangle, but doesn't
unblock this itself (moto simulates S3, not Backblaze B2's actual
behavior) — see `security-remediation-plan.md`'s landmine section.

**Open / deferred**: CI scaffold, region-detail cross-owner exposure
decision, IDOR fix (`location-id`/`oid` ownership checks), ruff baseline
config, logging/observability pass, documentation gaps, `allauth.mfa`
evaluation, "Delete" tool implementation decision.

## Track: TF1 → TF2 Migration (`triplet-reid` pipeline)

Full detail: [`docs/tf1-to-tf2-migration-plan.md`](tf1-to-tf2-migration-plan.md)

Started 2026-07-17 — a separate effort making the `triplet-reid`
training/export pipeline (RunPod-side serving, `ibex_stambecchi`-related)
reproducible. **Not** a production fix — production was independently
verified sound on TF1 before this track began; the goal is a clean,
committed pipeline that reproduces the shipped model on purpose, not by
accident.

**Done**: root-caused the historical export bugs (graph/export mechanics,
not a retrain issue); ported `nets/`/`heads/` to TF2 via `tf_slim`;
consolidated into `training/triplet-reid/` (committed to webibex's own git
history); verified across TF 2.18.0/2.18.1/2.21.0 with identical numeric
results; **TF 2.21.0 (`tf2210`) is the default** (decided + full e2e
verified 2026-07-23, including driving the real Django upload → landmark →
crop → embed flow through the browser against a locally-served export).

**Open**: whether retraining is ever actually in scope, or this stays
permanently "same weights, new export wrapper" (needs confirming with
Laurens directly); the `dhi.io/tensorflow-serving:2` hardened-base-image
swap for the production RunPod serving container (draft patch not yet
applied, separate from the migration itself).

## Related but out of scope for both tracks above

- `docs/changes/*.md` — individual completed-CR docs (what changed, why,
  how to verify/roll back), one per meaningful change. Not a roadmap, a
  changelog.
- `docs/session-notes-*.md` — dated, ephemeral working notes consumed by
  the next session that touches the same area; not meant to be a durable
  reference (superseded by the tracks above or by CR docs once their
  findings are folded in).
