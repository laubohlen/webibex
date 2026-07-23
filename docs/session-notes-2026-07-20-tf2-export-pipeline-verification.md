# Session notes — 2026-07-20 — TF migration wrap-up + e2e test attempt

Continuation of the TF1→TF2 `triplet-reid` export pipeline work. The durable
record for everything through the 3-version verification is
`docs/tf1-to-tf2-migration-plan.md` — this file covers only what happened
*after* that (the e2e test attempt) and hasn't been folded in yet.

## State at start of this segment

- `training/triplet-reid/` committed (`cd7b5b0`), TF 2.18.1 verified baseline.
- `training/triplet-reid/dockerfiles/{tf2180,tf2181,tf2210}/Dockerfile` +
  `dockerfiles/verify_gate.sh` — all three TF versions (2.18.0, 2.18.1, 2.21.0)
  gate-passed with identical numeric result (`max abs diff: 3.099e-06`).
  **`dockerfiles/` is untracked/uncommitted** as of this session's end.
- OSV.dev full-database scan (1,707 tensorflow advisories): zero affecting
  2.18.1 or 2.21.0.

## e2e test attempt: Django frontend + locally-loaded model

Goal: exercise the real upload → chip → embed → retrieval flow against a
freshly-verified export, not just the isolated numeric gate.

**Architecture found** in `core/utils.py::embed_new_chip()`: four branches
keyed on `settings.ENDPOINT_LOCALLY` / `settings.POSTGRES_LOCALLY`. The
"local model" branch (`model_is_local`) loads `core/embedding_model/`
directly inside the Django process via `get_tf()` (lazy `import tensorflow`)
— no container involved. `endpoint_inference()` (the RunPod path) hardcodes
`https://api.runpod.ai/v2/{endpoint_id}/runsync`, no override mechanism.

**Real blocker found, not yet fixed**: `webibex/settings.py:141` sets
`ENDPOINT_LOCALLY = True` unconditionally (not `env()`-backed, no default
fallback) — combined with `model_is_local = not (ENVIRONMENT == "production"
or ENDPOINT_LOCALLY == True)`, this makes the "local model" branch
**unreachable by default**, even outside production. Nobody running this
app locally today would hit the local-model path without manually editing
this line.

**Proposed but NOT applied** (discussed, not written):
1. `webibex/settings.py:141`: `ENDPOINT_LOCALLY = True` →
   `ENDPOINT_LOCALLY = env.bool("ENDPOINT_LOCALLY", default=True)` —
   preserves current default everywhere, only changes behavior if the env
   var is explicitly set.
2. `core/utils.py:320,338`: hardcoded `"core/embedding_model/"` → a
   settings-configurable `EMBEDDING_MODEL_PATH`, default unchanged — lets a
   test point at a separate export dir without touching the committed
   `core/embedding_model/`.

**Sandbox constraints hit while setting up a test venv**:
- `pypi.org`/`files.pythonhosted.org` egress blocked (403) intermittently —
  root cause confirmed by the user: the session's temp-egress window had
  expired, not a proxy-side cache issue (that was a wrong hypothesis pursued
  for ~10 min before the user clarified).
- `pip install` blocked by a `pip-guard`/`uv-guard` post-install audit
  reporting a malformed CRITICAL finding (`package: null`) for ordinary
  packages (`requests`) — confirmed the install actually succeeds and isn't
  rolled back; this is a bug in the guard's vulnerability-scan response
  parsing, not a real finding. Override: `DEPENDENCY_SCAN=warn`.
- `uv pip install` (default cache mode) hit `Invalid cross-device link` on
  `~/.cache/uv/archive-v0/` (overlayfs quirk) — worked around with
  `--no-cache` or a relocated `UV_CACHE_DIR`.
- `uv venv --python 3.12` failed (`runtime.txt` pins Python 3.12.5) — no
  write access to `~/.local/share/uv/python` to download a managed
  toolchain. User confirmed Python 3.13 (already on `PATH`) is fine for this
  devcontainer test — not investigating the 3.12 mismatch further.

**Real, unresolved dependency conflict** in `requirements.txt` as currently
pinned: `svglib==1.5.1` (exact pin) has no wheel on PyPI at all, only an
sdist. The only `svglib` versions with wheels (`1.6.0`, `2.0.2+`) require
`reportlab>=4.4.3`, but `requirements.txt` pins `reportlab==4.3.1` (also
exact). `svglib` is not skippable — `django-filer==3.3.0` (pinned) has its
own dependency on `easy-thumbnails[svg]`, which requires `svglib`
regardless of whether app code imports it (it doesn't — confirmed via
repo-wide grep, but that's irrelevant to the resolver). Under a wheels-only
("no arbitrary code execution at install time") install policy, these pins
are simply unsatisfiable as-is — this is not specific to the test venv or
anything done this session.

User's suggested next steps (not yet investigated): check whether bumping
`reportlab` is actually safe/needed, or whether `reportlab` is even still
needed at all; if `svglib` genuinely needs to stay at an unwheeled version,
consider cloning the `svglib` repo under `tmp/` and building a wheel
locally, rather than seeking a `pip-guard`/`uv-guard` policy exception.

**`.venv/` state at session end**: created (`.venv/bin/python` → system
Python 3.13.5), most of `requirements.txt` installs cleanly, blocked on
`svglib`/`reportlab`. Not usable for the e2e test yet.

## Also still open (carried from `tf1-to-tf2-migration-plan.md`)

- `verify_gate.sh` does not persist its export (unlike the original
  `phase4_verification_gate.sh`, which copies to `wibex_export_verified/`)
  — a fresh persisted export would be needed to actually swap into
  `EMBEDDING_MODEL_PATH` for the e2e test once the venv is unblocked.
- `training/triplet-reid/dockerfiles/` not yet committed to git.
- No decision yet on which TF version (2.18.1 vs 2.21.0) becomes the
  runbook default going forward.
- `dhi.io/tensorflow-serving:2` (pinned to TF Serving 2.20.0) hardened-image
  swap for the production RunPod serving container — separate, deferred
  track, version mismatch vs. 2.21.0 already flagged in the migration doc's
  Addendum section. User flagged this again as a "next todo" this session.
- Five-directory sprawl (`wibex_model_v01/v02`, `model/`, `new_model2/`,
  `saved_model/`) under `tmp/inference/` still pending removal.
