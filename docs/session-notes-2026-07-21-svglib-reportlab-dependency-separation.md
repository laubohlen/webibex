# Session notes — 2026-07-21 — svglib/reportlab dependency cleanup + tree separation

Continuation of the unresolved dependency conflict flagged in
`docs/session-notes-2026-07-20-tf2-export-pipeline-verification.md`
("Real, unresolved dependency conflict" section).

## Investigation

- `svglib==1.5.1` (exact pin in root `requirements.txt`) has no wheel on
  PyPI, only an sdist. Wheel-having versions (`1.6.0`, `2.0.2+`) require
  `reportlab>=4.4.3`, but `reportlab==4.3.1` was also exact-pinned —
  unsatisfiable under a wheels-only install policy.
- Confirmed `webibex/settings.py:224`, inside `FILER_ADD_FILE_VALIDATORS`:
  `"image/svg+xml": ["filer.validation.deny"]` (the dict also has a
  `text/html` → `deny_html` entry, unrelated) — SVG uploads are denied
  outright on the Django site. This is what made `svglib`/`reportlab` safe
  to drop: they only backed `easy-thumbnails[svg]`'s SVG-thumbnail
  rendering for `django-filer` uploads (confirmed via web search —
  django-filer's SVG support is implemented through
  `easy-thumbnails[svg]`, which pulls in `svglib`+`reportlab`), and that
  code path can never execute.
- Repo-wide grep confirmed zero application-code references to `svglib`,
  `reportlab`, `cssselect2`, `tinycss2`, or `webencodings` — the only other
  hits were vendored third-party site-packages under
  `tmp/inference/.runpod/` (pip's vendored pygments/rich/setuptools, false
  positives) and this investigation's own doc trail.

## Deployment-target dependency-tree separation (confirmed)

Three independent trees, no tracked overlap:

| Target | Requirements source | Tracked in git? | TF? | Django? |
|---|---|---|---|---|
| Railway (Django app) | `requirements.txt` + `init_prod_requirements.txt` | yes | no | yes |
| RunPod training/export (`training/triplet-reid/`) | `dockerfiles/{tf2180,tf2181,tf2210}/Dockerfile` | **no — untracked** | yes | no |
| RunPod inference/serving | `tmp/inference/*/builder/requirements.txt` | **no — `tmp/` fully gitignored** | yes | no |

The `svglib`/`reportlab` conflict is Railway-only; it was never a RunPod
concern (this was double-checked after an initial scoping mix-up mid-session
— see the new [[feedback_verify_dependency_tree_boundaries]] memory).
Also noted in passing: of the three `tmp/inference/*/builder/requirements.txt`
files, two pin TensorFlow and disagree with each other
(`runpod_ibex_embedding_endpoint` → `2.18.1`, `ibex_identification` →
`2.18.0`; the third, `worker-template`, pins no TF at all). None is a
tracked source of truth — separate open item, not acted on this session.

## Change made

Edited root `requirements.txt`: removed `svglib==1.5.1`, `reportlab==4.3.1`,
and their now-orphaned transitives `cssselect2==0.8.0`, `tinycss2==1.4.0`,
`webencodings==0.5.1` (nothing else in the dependency graph needs them —
checked). `easy-thumbnails==2.10` kept as-is; `django-filer` still needs it
for non-SVG thumbnails, which is the live code path.

**Not committed.** A commit message was written to `tmp/commit-message.txt`
(gitignored scratch — the user will copy it into their own `git commit`
manually). That file previously held the stale 2026-07-08 security-bump
message (already committed as `480607b`) and was overwritten.

## Verification — resolved 2026-07-23 (continuation session)

- Egress block resolved (temp whitelist fired by the user); `.venv`
  bootstrapped via `uv pip install -r requirements.txt` — 41 packages
  resolved cleanly, Django 5.2.15 installed, no errors.
- **Correction to the removal rationale**: `svglib`/`reportlab` are not
  actually absent from the resolved environment — `easy-thumbnails==2.10`
  pulls them in as a non-optional transitive dependency. The resolver
  picked newer, wheel-having versions (`svglib==2.0.2`, `reportlab==5.0.0`)
  once the broken exact pins (`1.5.1`/`4.3.1`) were dropped. See the
  updated `docs/changes/2026-07-21-remove-svg-thumbnail-deps.md` for the
  full correction — the functional conclusion (safe to unpin, SVG code
  path unreachable) still holds.
- `core/test_model.py` deliberately **not** run: it's not a real test (no
  assertions, `print`/`try-except` only) and hardcodes a path to the
  original developer's machine (`/Users/lau/Documents/...`) that doesn't
  exist here. A much stronger check ran instead: a full e2e test
  (`training/triplet-reid/dockerfiles/tf2210` export → local RunPod
  container → real Django upload flow) exercised real file uploads and
  thumbnail display repeatedly and successfully.
- Confirmed separately: there is no pytest/coverage setup in this repo at
  all (no `pyproject.toml`, `pytest.ini`, `.coveragerc`; `pytest` isn't even
  a listed dependency). Test files that exist: `core/test_model.py`
  (not a real test, see above) and `training/triplet-reid/tests/test_export_pipeline.py`
  (TF export pipeline, separate tree). Neither has a coverage harness.

## Also still open (carried forward, unchanged this session)

- `training/triplet-reid/dockerfiles/` — committed 2026-07-23 (continuation
  session), targeting `tf2210` (TF 2.21.0) as default, after a full e2e
  verification. See `docs/tf1-to-tf2-migration-plan.md`.
- `tmp/inference/*` TF version mismatch (2.18.1 vs 2.18.0) between the two
  builder requirements.txt files — not yet reconciled, no tracked source of
  truth for the deployed RunPod inference endpoint.
- Everything else carried from the 2026-07-20 notes (`verify_gate.sh`
  export persistence, TF version runbook default decision, TF Serving image
  swap, `tmp/inference/` directory sprawl) — untouched this session.
