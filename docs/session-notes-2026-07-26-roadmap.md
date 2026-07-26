# Session notes — 2026-07-26 — docs/roadmap.md created

## What happened

User asked whether a generic roadmap.md existed (it didn't — only
`docs/security-remediation-plan.md` and `docs/tf1-to-tf2-migration-plan.md`,
two narrow-scope tracking docs). Asked to merge the two into one for a
clearer single reference point.

## Decision

Given `docs/security-remediation-plan.md` (708 lines, growing every session
via append-only "Resolved" sections) + `docs/tf1-to-tf2-migration-plan.md`
(412 lines) — combined ~1120 lines and growing — presented 3 structural
options via AskUserQuestion:
1. Lean roadmap + linked detail docs (recommended)
2. Full literal merge, one file, delete originals
3. Full merge now, split later if it grows

User picked option 1. Created `docs/roadmap.md`: a short, scannable entry
point with one section per track (Security & Tech Debt Remediation, TF1→TF2
Migration), each summarizing current status (done/blocked/open items) in a
few lines and linking to the full detail doc. The two original docs were
**kept at their existing paths**, not moved or renamed — both are
cross-referenced from many other files in this repo (session notes, CR
docs under `docs/changes/`), and moving them would require updating every
reference for no real benefit over just linking from the new index.

Also added a one-line pointer from `README.md` to `docs/roadmap.md` per a
follow-up request, so it's discoverable from the repo's front door.

## Content notes (for future accuracy checks)

`docs/roadmap.md`'s two track summaries were written from a full read of
both source docs this session (not skimmed) — status as of 2026-07-26:
- Security track: TF removal (`ee1839f`) and auth-hardening test coverage
  (`978f785`) both landed this session; boto3/botocore/urllib3 triangle
  still blocked on a B2 test bucket; moto S3-mock test tier is planned but
  not yet executed.
- TF1→TF2 track: TF 2.21.0 (`tf2210`) is the committed default as of
  2026-07-23, verified via a full e2e pass through the real Django app;
  open questions are whether retraining is ever in scope, and the
  `dhi.io/tensorflow-serving:2` hardened-base-image swap (draft, not
  applied).

If either source doc changes significantly in a future session,
`docs/roadmap.md`'s per-track summary paragraphs need a matching update —
they're a snapshot, not auto-derived.

## Pattern logged

Saved a feedback memory ([[feedback-prefer-lean-index-over-full-doc-merge]])
recommending this lean-index approach as the default proposal whenever
asked to merge growing/large tracking docs in the future, on this project
or others.
