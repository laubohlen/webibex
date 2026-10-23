# TODO — Error reporting, user feedback and request IDs for the web service

Raised 2026-10-23. Status: **planned, not started**. Urgent but small; deferred on purpose now that the
emergency is over. To be done in a **dedicated devcontainer-guard (dcg) instance session**, not ad hoc.
Facts below come from the 2026-10-23 debugging session; line numbers were read that day and may drift.

## Why (what happened)

- `POST /save-landmarks/` and `GET /result_default/517/` returned 500 after the Railway redeploy.
- Cause: the RunPod serverless endpoint was **paused** (`max_workers=0`). RunPod answered
  `409 ENDPOINT_PAUSED`; `endpoint_inference` (`core/utils.py`, ~line 220) turned it into a `ValueError`.
  Fixed on 2026-10-23 by setting Max workers to 1 and Active workers to 0 in the RunPod console.
- Finding it took hours because nothing reported it:
  - `webibex/settings.py:42` sets `DEBUG = ENVIRONMENT == "development"` and there is no `LOGGING`, so
    Django prints no traceback for a 500 in production.
  - `print()` output was buffered until `PYTHONUNBUFFERED=1` was set on the Railway service (done
    2026-10-23, keep it).
  - `raise_for_status()` drops RunPod's response body, so `ENDPOINT_PAUSED` never appeared in any message.
  - A failed embed leaves an `IbexChip` row and its B2 file with no `Embedding`, so
    `/result_default/<id>/` then fails at `core/views.py:200` (`query.embedding.embedding`).
- The DB-proxy theory (`TCP_ABORT_ON_DATA` toward the DB) was **not** the cause here. Unrelated, unconfirmed.

## Scope

### 1. Error reporting (~half a day with tests, my estimate)
1. Add `LOGGING` sending `django.request` ERROR to the console (~10 lines in `webibex/settings.py`).
2. In `endpoint_inference`, keep the status and the first ~300 characters of RunPod's response body in the
   exception message instead of discarding it.
3. Typed errors: e.g. `EmbeddingUnavailable` (409, timeout, cold start) versus a DB `OperationalError`, so a
   log line says which dependency failed.
4. Clean up after a failed embed (delete the half-made chip or handle the missing embedding in
   `default_chip_compare_view`).
5. Replace the `print()` calls on this path with `logging`.

### 2. User feedback (~2 hours for webibex, my estimate)
- `django.contrib.messages` is already installed (`settings.py:82,99,124`). I found no `messages.*` calls and
  no template loop rendering them; re-check the base template, my grep may have missed a syntax.
- In the view, catch the typed error, call `messages.error(...)`, redirect instead of returning a 500.
- Render the messages as a dismissible banner in the base template. No notification center needed.

### 3. Request ID in logs
- Middleware reads or generates `X-Request-ID`, stores it in a `ContextVar`, echoes it in the response;
  a `logging.Filter` adds it to every record. Do it together with item 1.1 (same settings block).
- Decision: **copy the design from tmgame, not the code.** tmgame's is ASGI/Starlette
  (`src/tmgame/api/request_logging.py`, `log_format.py`, `request_log_policy.py`); webibex is Django on
  gunicorn sync. Revisit sharing `log_format.py` only if it turns out framework-free and tmgame's middleware
  has settled (not checked, still being written there).
- Unknown: whether Railway injects its own request ID header.

## Not in scope
- Sharing a notification component with tmgame: different stacks (Django templates vs FastAPI + Vue). tmgame's
  planned notification system (Brevo + VAPID push, `docs/effort-and-security.md:15`) is a different problem.
  Share the error-code names and user-facing wording at most.

## Verification
- A forced RunPod 409 (endpoint paused, or a stubbed response) shows a traceback and RunPod's `detail` in the
  Railway deploy log, and the user sees a banner instead of a 500.
- A request ID appears on every log line of one request and in the response header.
