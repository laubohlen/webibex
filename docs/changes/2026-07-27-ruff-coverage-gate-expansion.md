# CR: re-enable ruff on two near-100%-coverage deferred files

**What changed:**
- Modified: `ruff.toml` — removed `per-file-ignores` entries for `core/models.py`
  (98% coverage) and `core/templatetags/custom_template_tags.py` (93% coverage).
  Both now enforced under the curated ruleset
  (`E,F,UP,B,S,SIM,PIE,C4,T20,ANN,RUF`). Left the remaining 11 deferred files
  untouched.
- Modified: `core/models.py` — `Region.Meta.constraints`, `Location.SOURCE_CHOICES`,
  `IbexImage.SIDE_CHOICES` annotated `ClassVar` (RUF012, mutable Django
  class attributes); `Location.__str__` given `-> str` return type (ANN204);
  reflowed one over-length inline comment on `created_at` (E501, moved above
  the field instead of trailing).
- Modified: `core/templatetags/custom_template_tags.py` — `dict_get(d, key)`
  typed as `Mapping[int, str], int -> str` (matches its one real call site,
  `id_to_color|dict_get:chip.ibex_image.animal.id` in
  `templates/core/result_default.html`/`result_refined.html`);
  `post_task_redirect` typed `Context, str, *str|int, **str|int -> str`.
  `key`/`args`/`kwargs` avoid bare `typing.Any` (ANN401 forbids it) — narrowed
  to the concrete types actually passed in.
- Modified: `docs/security-remediation-plan.md` — appended an UPDATE note to
  the 2026-07-27 "ruff-baseline deferred files" TODO; deferred-files table
  itself left as-is (append-only doc convention).

**Follow-up action:** none required for these two files. 11 files remain
deferred (`core/admin.py`, `core/signals.py`, `core/utils.py`, `core/views.py`,
`simple_landmarks/views.py`, `webibex/urls.py`, and 5 unmeasured files) — see
`docs/security-remediation-plan.md`'s TODO for the current table.

**Do NOT:**
- Treat this as a policy change to the coverage gate. The documented default
  stays 100% measured coverage; this was an explicit, user-approved one-off
  exception for two files that happened to already be ruff-clean under the
  full ruleset with zero fixes needed on uncovered lines. Don't cite this CR
  to justify re-enabling a file below 100% without asking first.
- Assume `dict_get`'s `Mapping[int, str]`/`int` typing is a general-purpose
  contract — it reflects the filter's one actual call site. If a new call
  site passes a different key/value type, the annotation needs revisiting,
  not a blind `Any` fallback (ruff's `ANN401` blocks that anyway).

**Trigger:** none — this CR is complete. General trigger for the wider
deferred-file backlog: re-run `pytest --cov-report=term-missing`, pick a file
at/near 100%, re-enable, triage findings — same procedure as this CR and the
original `docs/changes/2026-07-27-ruff-baseline-config.md`.

**Why:** continues the coverage-gated ruff rollout the user set up earlier
this session. User explicitly relaxed the gate for this pass only ("not 100%
mandatory, but the coverage must be 'ok'") but then chose to keep the
*documented* policy at 100% and treat these two files as a case-by-case
exception rather than lowering the general threshold — coverage is the
review oracle for `ruff --fix`-adjacent changes, and a blanket lower
threshold would apply to files with real uncovered branches, not just these
two near-100% ones.

**Verify:** `.venv/bin/ruff check . --statistics` → 0 findings project-wide
(was 13 immediately after removing the two `per-file-ignores` entries, before
the fixes above). `.venv/bin/pyright core/models.py
core/templatetags/custom_template_tags.py --outputjson` → 6 errors, identical
(same messages, same relative line offsets) to the pre-CR baseline — all 6
are the pre-existing `django-stubs`-gap errors documented in
`docs/changes/2026-07-27-boto3-stubs-typing.md`, net zero new errors.
`.venv/bin/python -m pytest -q --cov=core --cov=simple_landmarks --cov=webibex`
→ 184 passed, 1 skipped, 1 xfailed, `core/models.py` 98%/`custom_template_tags.py`
94% (unchanged from before — no new tests added, per the coverage-relaxation
decision).

**Rollback:** `git revert <this commit>` — restores the two
`per-file-ignores` entries and pre-fix file content. No schema/migration/
external-service changes; annotation-and-config-only, safe to revert cleanly.
