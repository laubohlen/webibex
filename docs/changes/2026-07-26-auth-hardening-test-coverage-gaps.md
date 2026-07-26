# CR: close auth-hardening test coverage gaps (SSL redirect, HSTS, CSRF rejection)

**What changed:**
- Modified: `tests/webibex/test_manual_login_logout_check.py` — 3 new tests:
  - `test_plain_http_request_redirects_to_https_under_simulated_production_settings` — a plain (non-`secure=True`) GET under the full production `@override_settings` block asserts a 301 with an `https://`-prefixed `Location`, exercising the `SECURE_SSL_REDIRECT` redirect path the file's other tests (all `secure=True`) structurally can't reach.
  - `test_strict_transport_security_header_present_on_secure_response` (needs `db`) — asserts `response.headers["Strict-Transport-Security"] == "max-age=3600"` on a live `secure=True` response, not just the settings value.
  - `test_password_change_rejects_post_with_invalid_csrf_token` — logs in with a normal `Client()`, copies the `sessionid` cookie into a second `Client(enforce_csrf_checks=True)`, asserts the copy actually authenticated it (GET returns 200, not a login redirect), then POSTs the password-change form with a deliberately wrong `csrfmiddlewaretoken` and asserts 403.
- Status: committed as `978f785`.

**Follow-up action:** none required — this closes a documented coverage gap, no new pattern to remember for future work.

**Do NOT:**
- Assume `Client(..., secure=True)` alone proves `SECURE_SSL_REDIRECT` works — it makes the request already look secure and bypasses the redirect path entirely. Use a plain (non-`secure=True`) request when testing the redirect itself.
- Trust a bare `assert response.status_code == 403` on a CSRF test to prove the rejection is CSRF-specific — `CsrfViewMiddleware` runs before any `@login_required` check, so an anonymous bad-token POST also 403s. Add an authenticated-precondition assertion (e.g. a GET that only succeeds when logged in) before the CSRF-triggering POST, or the test can pass for the wrong reason.

**Trigger:** any future change to `SECURE_SSL_REDIRECT`/`SECURE_HSTS_SECONDS`/CSRF-related settings in `webibex/settings.py` — re-verify these three behaviors still hold, not just the settings values.

**Why:** a `/request-adherence` check against the prior auth/session hardening CR found these three behaviors were never explicitly requested and remained untested, even though the underlying settings (`SECURE_SSL_REDIRECT`, `SECURE_HSTS_SECONDS`, CSRF enforcement) were already live. Verified as true positives (not accidentally passing) by a Fable5 adversarial trace: 11 empirical mutation probes against live Django/allauth source confirmed each test genuinely fails when the property it claims to test is removed.

**Verify:** `.venv/bin/python -m pytest tests/webibex/test_manual_login_logout_check.py -v` (7/7 pass — 10/10 when counting both auth-hardening test files together with `test_settings_security_hardening.py`); full suite `.venv/bin/python -m pytest -q` (159 passed, 1 skipped, 1 xfailed at the time of this CR).

**Rollback:** `git revert 978f785` — removes the 3 new tests, no production code is touched by this CR so nothing else is affected.
