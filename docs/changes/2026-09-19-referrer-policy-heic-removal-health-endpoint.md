# CR: OSM referrer-policy fix, remove vulnerable unused pillow_heif, add /health endpoint

**What changed:**
- Modified: `webibex/settings.py` — added `SECURE_REFERRER_POLICY =
  "strict-origin-when-cross-origin"`, set unconditionally (not gated to
  `ENVIRONMENT == "production"` like the other hardening settings — no
  HTTP-breakage risk, and it fixes local-dev map testing too). Placed
  right after `CSRF_TRUSTED_ORIGINS`, before the existing production-only
  hardening block.
- New: `tests/webibex/test_settings_security_hardening.py` gained 3 tests
  (`test_referrer_policy_set_under_ambient_environment_test`,
  `_under_environment_production`, `_under_environment_development`) —
  present in every environment, unlike the 5 production-gated settings the
  file already tested.
- Modified: `requirements.txt` — removed `pillow_heif==0.22.0` outright
  (was pinned since at least 3 prior version bumps: 0.16.0→0.18.0→0.22.0,
  per `git log -p -- requirements.txt`). Also uninstalled it from the dev
  `.venv` itself later in the session (it survived the manifest edit
  because nothing had actually run `uv sync`/reinstalled).
- New: `core/views.py::health_view` + `webibex/urls.py`'s `path("health/",
  health_view, name="health")` — unauthenticated JSON endpoint:
  `{"status": "ok", "commit": <RAILWAY_GIT_COMMIT_SHA or "unknown">,
  "django": <django.get_version()>}`. No DB/B2/RunPod calls.
- New: `tests/core/test_health_view.py` (4 tests: response shape, env-var
  fallback, env-var override, anonymous access).
- Modified: `tests/webibex/test_all_views_smoke.py` — URL-count
  completeness guard bumped 26→27 first-party names (the new `health`
  route), test renamed to match, added `assert "health" in discovered`.
- Modified: `docs/security-remediation-plan.md` — extensive findings log
  for all of the above plus a `debug_toolbar`-dependency investigation
  (reproduced the crash, confirmed a fix works, then deliberately did NOT
  add it as a standing dependency) and a full `/supply-chain` audit
  re-run (unused-dependency sweep, Python + JS CVE scans, socket.dev
  cross-check — see that doc's own dated sections for detail, not
  duplicated here).

**Follow-up action:** all 3 commits (`17bb21b`, `70a14cb`, `e0c544c`) are
already pushed to `origin/docs/ctem-security-review-note` (verified via
`git status`) — but this CR alone does not deploy itself. None of the
fixes (especially the OSM referrer-policy one, which is actively affecting
the professor's live use) take effect on `wibex.up.railway.app` until this
branch is merged/deployed.

**Do NOT:**
- Re-add `pillow_heif` casually if HEIC upload support is ever revisited.
  The removed version bundled `libheif 1.19.7` — the exact vulnerable
  version in a real-world RCE writeup (heap buffer overflow, chained to an
  OpenAI SSO account-takeover PoC). Any future re-add must verify the
  bundled `libheif` version is patched (>=1.23.4) via
  `pillow_heif.libheif_version()`, not by pip version number alone — the
  PyPI package page doesn't state the bundled libheif version.
- Assume `SECURE_REFERRER_POLICY` fully closes the OSM tile-blocking issue.
  It resolves the `403r` case and, per one manual observation this
  session, also the general `403` case under a hardened Firefox config —
  but the doc's own analysis says it shouldn't help the general case in
  principle (extension/UA-driven, not Referer-driven). Single observation,
  not re-tested against the original failure conditions. MapTiler migration
  (already designed in the doc, not scheduled) remains the structural fix.
- Treat the `/supply-chain` audit's 7 Python + 11 JS CVE findings as
  requiring immediate action. All were reachability-triaged this session
  and are routine-bump candidates, not urgent — see
  `docs/security-remediation-plan.md`'s dated audit section for the full
  triage table before re-deciding priority.
- Commit this from inside the sandboxed devcontainer. Commit signing is
  broken here (host-only key) — left as staged/committed-by-the-user
  changes; all 3 commits this session were signed and committed by the
  user on the host from a message file Claude wrote to gitignored `tmp/`.

**Trigger:** none pending on this CR itself beyond the push above. The
SonarQube backlog (5 BLOCKER + 161 CRITICAL, untriaged since 2026-07-27)
and the routine dependency bumps surfaced by this session's CVE scans
(`django`, `setuptools`, `pip`, `sqlparse`, `node/package-lock.json`
regeneration) are tracked separately in `docs/security-remediation-plan.md`,
not part of this CR.

**Why:** the referrer-policy fix directly addresses a live, professor-facing
bug (reproduced `403r`/`403` OSM tile errors). The `pillow_heif` removal
closes a CRITICAL supply-chain finding surfaced by a research article the
user shared mid-session, cross-referencing it against this app's actual
attack surface (confirmed unreachable — dead code — before removing rather
than reflexively patching). The `/health` endpoint fills a genuine
observability gap (zero CI, no Dockerfile, no prior way to check what's
deployed) raised by the user as a direct question.

**Verify:** `uv run pytest -q` from repo root → 706 passed, 5 skipped, 1
xfailed at the point `/health` landed (settled to 705 after the unrelated
`animal-own-images` patch was reverted later — see that separate,
untracked-by-this-CR working-tree cleanup). `webibex/settings.py` at 100%
line coverage; `webibex/urls.py` at 50% (lines 81-88 uncovered — the
pre-existing `if settings.DEBUG:` static-serving block, unrelated to this
CR's changes). Manually verified live via
`scripts/run_local_e2e_server.py` + the user's own `dev-tunnel` to this
container: `curl -I http://127.0.0.1:8000/` shows
`Referrer-Policy: strict-origin-when-cross-origin` on every response;
`curl http://127.0.0.1:8000/health/` returns
`{"status": "ok", "commit": "unknown", "django": "5.2.16"}` locally (no
`RAILWAY_GIT_COMMIT_SHA` outside a real Railway deploy); the actual OSM-tile
fix was confirmed visually in the user's own browser (Firefox +
NoScript + uBlock Origin), not just via HTTP headers.

**Rollback:** revert `SECURE_REFERRER_POLICY` in `webibex/settings.py`
(re-adds the OSM tile-blocking risk); re-add `pillow_heif==0.22.0` to
`requirements.txt` if genuinely needed (re-verify the bundled `libheif`
version first — see "Do NOT" above); delete `core/views.py::health_view`,
its `webibex/urls.py` route, and `tests/core/test_health_view.py`, and
revert `tests/webibex/test_all_views_smoke.py`'s count back to 26. All
changes are additive/isolated — no shared state, no migration, nothing
else depends on any of the three.
