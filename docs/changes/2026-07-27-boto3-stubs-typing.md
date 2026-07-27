# CR: install boto3-stubs/botocore-stubs, type core/b2_utils.py precisely

**What changed:**
- Modified: `requirements-dev.txt` — added `boto3-stubs[s3]==1.26.0` and `botocore-stubs==1.29.165` (exact pins matching production `boto3==1.26.0`/`botocore==1.29.165`).
- Modified: `core/b2_utils.py` — `get_b2_resource()` return type changed from generic `boto3.resources.base.ServiceResource` to `mypy_boto3_s3.S3ServiceResource`; `delete_files`'s `objects` list explicitly typed as `list[ObjectIdentifierTypeDef]`; `check_file_exists`'s `e.response["Error"]["Code"]` changed to `e.response.get("Error", {}).get("Code")` (strict `TypedDict`, `Error`/`Code` not marked required).
- Modified: `tests/core/test_b2_utils_moto.py` — matching `.get("Error", {}).get("Code")` fix at line 118 (same `TypedDict` issue).
- Status: committed as `863f359`.

**Follow-up action:** none required. This closes the deferred follow-up item from the moto S3-mock CR (`2cbfb84`, `docs/changes/2026-07-26-moto-s3-mock-test-tier.md`) about missing `boto3-stubs`/`botocore-stubs`. `django-stubs` remains uninstalled — separate, much larger scope (Django ORM `.objects` manager, `WSGIRequest` attributes), not part of this CR.

**Do NOT:**
- Assume installing the stub packages alone is sufficient — it changed the error set from 6→8 (more precise, not fewer) until the return-type/TypedDict annotations were also fixed. Any future stub-package bump on this file should re-run `pyright core/b2_utils.py tests/core/test_b2_utils_moto.py` and expect similar fallout if boto3/botocore's own stub shapes change.
- Bump `boto3-stubs`/`botocore-stubs` independently of a real `boto3`/`botocore` version bump — they're pinned to match the exact production versions (`1.26.0`/`1.29.165`), same rationale as the `moto` pin in the prior CR.

**Trigger:** whenever `core/b2_utils.py` gains new boto3/botocore call sites, or whenever the real boto3/botocore/urllib3 triangle bump (see `docs/security-remediation-plan.md`) proceeds — re-verify/re-pin these stub packages at the same time.

**Why:** `core/b2_utils.py`'s boto3 resource/client calls were untyped against boto3's actual generated types, producing false-positive pyright errors that masked the module's real 100% pyright cleanliness (aside from one unrelated pre-existing django-environ stub gap at line 15). Precise typing here also means pyright would now actually catch a real boto3 API misuse in this file (wrong client method name, wrong `Delete.Objects` shape) instead of silently no-oping on an untyped `ServiceResource`.

**Verify:** `source .venv/bin/activate && pyright core/b2_utils.py tests/core/test_b2_utils_moto.py` (1 error — pre-existing, unrelated `env()` NoValue gap at line 15, 0 boto3-related); `ruff check core/b2_utils.py tests/core/test_b2_utils_moto.py` (0 findings); `pytest -q` (184 passed, 1 skipped, 1 xfailed, `core/b2_utils.py` 100% coverage).

**Rollback:** `git revert 863f359` — restores the prior `requirements-dev.txt`/`core/b2_utils.py`/`tests/core/test_b2_utils_moto.py` state (generic `ServiceResource` typing, direct dict-index `ClientError.response` access, no stub packages). No production runtime behavior to unwind — this CR is type-annotation-only plus a dev dependency.
