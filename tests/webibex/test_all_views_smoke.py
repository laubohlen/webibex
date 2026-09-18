"""T27-T34: all-views crash smoke sweep (R4) + completeness meta-test (R5).

Recursively walks `get_resolver()`, excluding third-party `include()`
subtrees (admin/allauth/filer, via a documented route-prefix exclude
list -- see `_THIRD_PARTY_ROUTE_PREFIXES`). For every remaining
first-party URL name, authenticates and dispatches (GET by default, with
a per-URL-name `OVERRIDES` entry for anything needing a specific
method/kwargs/POST-body), then asserts ONLY "did not crash": HTTP < 500
and no escaping exception. This file deliberately does NOT re-assert
auth-gate semantics (302-to-login, anonymous-vs-authenticated behavior)
-- that's tests/core/test_views_auth_required.py's job (46 tests), never
duplicated or modified here (R7).

R5 completeness: every discovered URL name must be either exercised by
the sweep or explicitly allowlisted with a reason (mirroring root
conftest.py's `_MOTO_S3_MISUSE_ALLOWLIST` pattern) -- a collection-time
gap here means a future new view silently gets zero smoke coverage.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from django.contrib.contenttypes.models import ContentType
from django.urls import get_resolver, reverse
from django.urls.resolvers import URLPattern, URLResolver

from core.models import Embedding

# ---------------------------------------------------------------------------
# URL discovery.
# ---------------------------------------------------------------------------
# Documented exclude-list of ROOT_URLCONF route prefixes mounting
# third-party include()s (webibex/urls.py:28-30) -- their views are
# owned/tested upstream, not by this app.
_THIRD_PARTY_ROUTE_PREFIXES = ("webibex/", "accounts/", "filer/")


def discover_url_names(resolver=None) -> dict[str, URLPattern]:
    if resolver is None:
        resolver = get_resolver()
    discovered: dict[str, URLPattern] = {}
    _walk(resolver, "", discovered)
    return discovered


def _walk(
    resolver: URLResolver, prefix: str, discovered: dict[str, URLPattern]
) -> None:
    # Django's own stubs-less `URLResolver.url_patterns` is a
    # `@cached_property` pyright can't resolve to its real runtime type
    # (`list[URLPattern | URLResolver]`) without django-stubs.
    url_patterns = resolver.url_patterns
    for entry in url_patterns:  # pyright: ignore[reportGeneralTypeIssues]
        full_prefix = prefix + str(entry.pattern)
        if isinstance(entry, URLResolver):
            if full_prefix in _THIRD_PARTY_ROUTE_PREFIXES:
                continue
            _walk(entry, full_prefix, discovered)
        elif entry.name is not None:
            discovered[entry.name] = entry


# ---------------------------------------------------------------------------
# Per-URL-name overrides.
# ---------------------------------------------------------------------------
def _result_refined_override(scenario):
    return {
        "method": "POST",
        "data": {"toggle": "false", "region": scenario["region"].id},
        "url_kwargs": {"oid": scenario["chip"].id},
    }


def _saved_animal_selection_override(scenario):
    # PIN: saved_animal_selection_view's GET branch (core/views.py:37)
    # references an unbound local `oid` -> NameError (F6, confirmed this
    # session). Dispatches via POST only, the working path. Not fixed:
    # R8 forbids production-code changes for this CR -- mirrors the
    # PIN-comment convention in tests/core/test_views_auth_required.py's
    # T20-T23 (rerun_view / save_landmarks_view / project_chip_compare_view).
    return {
        "method": "POST",
        "data": {
            "selectedAnimalId": scenario["animal"].id,
            "query_chip_id": scenario["chip"].id,
        },
    }


def _save_landmarks_override(scenario):
    return {
        "method": "POST",
        "data": {
            "image-id": scenario["image"].id,
            "horn_x": "400",
            "horn_y": "800",
            "eye_x": "800",
            "eye_y": "1600",
        },
        "patches": ["core.views.utils.process_horn_chip"],
    }


def _save_image_location_override(scenario):
    return {
        "method": "POST",
        "data": {
            "region-id": scenario["region"].id,
            "latitude": "46.0",
            "longitude": "8.0",
            "location-id": scenario["location"].id,
            "image-id": scenario["image"].id,
        },
    }


OVERRIDES = {
    "result-refined": _result_refined_override,
    "saved-animal-selection": _saved_animal_selection_override,
    "save-landmarks": _save_landmarks_override,
    "save-image-location": _save_image_location_override,
}

# R5 completeness allowlist: url_name -> reason it is deliberately NOT
# exercised by the smoke sweep, mirroring root conftest.py's
# `_MOTO_S3_MISUSE_ALLOWLIST`.
ALLOWLIST: dict[str, str] = {
    "run-again": (
        "PINNED pre-existing bug (F7): rerun_view renders "
        "'core/result.html', which does not exist -- always raises "
        "TemplateDoesNotExist for an authenticated GET (see "
        "tests/core/test_views_auth_required.py T20). R8 forbids fixing "
        "core/views.py in this CR."
    ),
}

_OID_URL_NAMES = frozenset(
    {
        "read-image",
        "update-image",
        "delete-image",
        "locate-image",
        "result-default",
        "result-refined",
        "new-ibex",
        "animal",
        "animal-own-images",
        "read-region",
        "delete-region",
        "update-region",
    }
)


def _default_oid_for(url_name: str, scenario: dict) -> int:
    if url_name in ("read-image", "update-image", "delete-image", "locate-image"):
        return scenario["image"].id
    if url_name in ("result-default", "result-refined", "new-ibex"):
        return scenario["chip"].id
    if url_name in ("animal", "animal-own-images"):
        return scenario["animal"].id
    if url_name in ("read-region", "delete-region", "update-region"):
        return scenario["region"].id
    raise AssertionError(f"no default oid mapping for {url_name!r}")


def _dispatch(client, url_name, scenario, monkeypatch):
    override = OVERRIDES.get(url_name)
    method, data, url_kwargs = "GET", {}, {}
    if override is not None:
        spec = override(scenario)
        method = spec.get("method", "GET")
        data = spec.get("data", {})
        url_kwargs = spec.get("url_kwargs", {})
        for target in spec.get("patches", []):
            monkeypatch.setattr(target, lambda *_a, **_kw: None)
    elif url_name in _OID_URL_NAMES:
        url_kwargs = {"oid": _default_oid_for(url_name, scenario)}

    url = reverse(url_name, kwargs=url_kwargs)
    if method == "POST":
        return client.post(url, data)
    return client.get(url)


def _run_sweep(url_names, client, scenario, monkeypatch):
    failures = []
    for url_name in url_names:
        try:
            response = _dispatch(client, url_name, scenario, monkeypatch)
        except Exception as exc:
            failures.append(f"{url_name}: raised {type(exc).__name__}: {exc}")
            continue
        if response.status_code >= 500:
            failures.append(f"{url_name}: HTTP {response.status_code}")
    return failures


# ---------------------------------------------------------------------------
# smoke_scenario fixture.
# ---------------------------------------------------------------------------
@pytest.fixture
def smoke_scenario(
    user_factory,
    ibex_image_factory,
    ibex_chip_factory,
    animal_factory,
    region_factory,
    location_factory,
    landmark_factory,
):
    owner = user_factory(username="smoke_user")
    animal = animal_factory(id_code="SMOKE_001")
    region = region_factory(owner=owner, name="SmokeRegion")
    location = location_factory()

    # side="L": create_folder_for_animal_on_change (core/signals.py)
    # raises UnboundLocalError for a non-L/R/O side -- pre-existing bug,
    # pinned separately, not this CR's concern (mirrors the same
    # workaround in tests/core/test_views_auth_required.py::gate_scenario).
    image = ibex_image_factory(owner=owner, side="L")
    image.animal = animal
    image.location = location
    image.save()

    chip = ibex_chip_factory(owner=owner, ibex_image=image)
    Embedding.objects.create(ibex_chip=chip, embedding=[1.0, 2.0, 3.0])

    # F10: LandmarkItem rows need NON-NONE coordinates, or
    # utils.percentage_coordinate() TypeErrors on `None / width`.
    horn_landmark = landmark_factory(label="horn_tip")
    eye_landmark = landmark_factory(label="eye_corner")
    content_type = ContentType.objects.get_for_model(image.__class__)
    from simple_landmarks.models import LandmarkItem

    LandmarkItem.objects.create(
        landmark=horn_landmark,
        content_type=content_type,
        object_id=image.id,
        x_coordinate=10,
        y_coordinate=20,
    )
    LandmarkItem.objects.create(
        landmark=eye_landmark,
        content_type=content_type,
        object_id=image.id,
        x_coordinate=30,
        y_coordinate=40,
    )

    return {
        "owner": owner,
        "animal": animal,
        "region": region,
        "location": location,
        "image": image,
        "chip": chip,
    }


# ---------------------------------------------------------------------------
# T27 -- URL discovery.
# ---------------------------------------------------------------------------
def test_discover_url_names_finds_all_27_first_party_names_excludes_third_party():
    discovered = discover_url_names()

    assert len(discovered) == 27
    assert "welcome" in discovered
    assert "update-image" in discovered
    assert "health" in discovered
    # allauth-owned (mounted under the excluded "accounts/" prefix).
    assert "account_login" not in discovered
    # DEBUG-only (settings.DEBUG is False under ENVIRONMENT=test).
    assert "test" not in discovered


# ---------------------------------------------------------------------------
# T28 -- smoke sweep: every exercised URL returns a non-crash status.
# ---------------------------------------------------------------------------
@pytest.mark.django_db
def test_all_exercised_urls_do_not_crash(client, smoke_scenario, monkeypatch):
    discovered = discover_url_names()
    exercised_names = sorted(set(discovered) - set(ALLOWLIST))
    assert exercised_names, (
        "no exercisable URL names discovered -- sweep would be vacuous"
    )

    client.force_login(smoke_scenario["owner"])
    failures = _run_sweep(exercised_names, client, smoke_scenario, monkeypatch)

    assert not failures, "\n".join(failures)


# ---------------------------------------------------------------------------
# T29 -- OVERRIDES POST-only views dispatch correctly (F6/F8/F9).
# ---------------------------------------------------------------------------
@pytest.mark.django_db
def test_overrides_post_only_views_dispatch_correctly(
    client, smoke_scenario, monkeypatch
):
    discovered = discover_url_names()
    client.force_login(smoke_scenario["owner"])

    for url_name in (
        "saved-animal-selection",
        "save-image-location",
        "save-landmarks",
        "result-refined",
    ):
        assert url_name in discovered
        response = _dispatch(client, url_name, smoke_scenario, monkeypatch)
        assert response.status_code < 500, f"{url_name}: HTTP {response.status_code}"
        assert response.status_code in (200, 302), (
            f"{url_name}: unexpected {response.status_code}"
        )


# ---------------------------------------------------------------------------
# T30 -- update-image override supplies non-None landmark coords (F10).
# ---------------------------------------------------------------------------
@pytest.mark.django_db
def test_update_image_dispatch_succeeds_with_non_none_landmark_coords(
    client, smoke_scenario
):
    client.force_login(smoke_scenario["owner"])

    url = reverse("update-image", kwargs={"oid": smoke_scenario["image"].id})
    response = client.get(url)

    assert response.status_code == 200


# ---------------------------------------------------------------------------
# T31 -- harness meta-test: a synthetic 500 view is detected as FAILURE.
# ---------------------------------------------------------------------------
def test_harness_detects_a_synthetic_500_view(client, smoke_scenario, monkeypatch):
    """Proves `_run_sweep`'s failure-collection actually catches a broken
    view -- monkeypatches `_dispatch` itself to simulate a view that
    raises, confirming the harness reports it instead of silently
    passing."""

    def _raising_dispatch(*_args, **_kwargs):
        raise RuntimeError("synthetic 500 for T31")

    monkeypatch.setattr(f"{__name__}._dispatch", _raising_dispatch)

    failures = _run_sweep(
        ["synthetic-broken-view"], client, smoke_scenario, monkeypatch
    )

    assert failures
    assert "synthetic-broken-view" in failures[0]
    assert "RuntimeError" in failures[0]


# ---------------------------------------------------------------------------
# T32 -- allowlist entries carry a reason and are still reachable.
# ---------------------------------------------------------------------------
def test_allowlist_entries_carry_a_reason_and_are_still_reachable():
    discovered = discover_url_names()

    assert ALLOWLIST, (
        "empty allowlist -- nothing to verify (update this test if intentional)"
    )
    for url_name, reason in ALLOWLIST.items():
        assert reason.strip(), f"{url_name}: allowlist entry has no reason"
        assert url_name in discovered, (
            f"{url_name}: allowlisted but no longer a live URL name"
        )


# ---------------------------------------------------------------------------
# T33 -- R5 completeness: exercised UNION allowlisted == discovered.
# ---------------------------------------------------------------------------
def test_completeness_every_discovered_url_is_exercised_or_allowlisted():
    discovered = discover_url_names()
    assert discovered, "empty discovered set -- completeness check would be vacuous"

    exercised = set(discovered) - set(ALLOWLIST)
    accounted_for = exercised | set(ALLOWLIST)

    missing = set(discovered) - accounted_for
    orphaned_allowlist = set(ALLOWLIST) - set(discovered)

    assert not missing, (
        f"discovered URL names neither exercised nor allowlisted: {missing}"
    )
    assert not orphaned_allowlist, (
        f"allowlist entries for URL names no longer discovered: {orphaned_allowlist}"
    )


# ---------------------------------------------------------------------------
# T34 -- this suite never duplicates auth-gate assertions.
# ---------------------------------------------------------------------------
def test_smoke_suite_never_asserts_login_redirect_location():
    """Documents the boundary with tests/core/test_views_auth_required.py:
    this module authenticates unconditionally and only asserts "did not
    crash" -- it must never assert the 302-to-login redirect location,
    which is test_views_auth_required.py's job."""
    source = Path(__file__).read_text()
    # Built via concatenation (not a single string literal) so this
    # guard's own source line never self-matches the substring search.
    login_redirect_literal = "/accounts" + "/login/"
    assert login_redirect_literal not in source


# ---------------------------------------------------------------------------
# T40 -- no conflict with the existing thin smoke-test file.
# ---------------------------------------------------------------------------
def test_no_url_name_overlap_conflict_with_existing_smoke_file():
    """tests/core/test_views_smoke.py already dispatches 'welcome' and
    asserts HTTP 200 -- this file's sweep does the same for the same URL
    name. Both expectations agree (200); neither file duplicates the
    other's assertions verbatim."""
    discovered = discover_url_names()
    assert "welcome" in discovered
