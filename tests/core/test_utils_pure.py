"""T01-T22: pure-function scenarios from core/utils.py.

P0 -- duck-typed stub objects only, no real Django models (per finalized
decision #4). Bugs B1-B3 are pinned via pytest.raises(...), not fixed
(finalized decision #5).

NOTE on property-based tests (T08, T17): the spec calls for Hypothesis
(@given), but `hypothesis` is not in requirements.txt and this sandbox has
no network access to install it for verification. The property/invariant
tests below use `@pytest.mark.parametrize` with a curated set of
representative inputs (boundary, small, large, non-square) instead --
same invariants asserted, deterministic rather than randomly generated.
Swap back to Hypothesis once the dependency is added and verified in an
environment with network access.

NOTE on parse_coordinates (T01-T05): uses `django.test.RequestFactory` to
build real `HttpRequest` objects (for `.GET` query-dict parsing) -- no
database is involved, so these stay in this DB-free module alongside the
other pure-function tests.
"""

import cv2
import numpy as np
import pytest
from django.test import RequestFactory

from core.utils import (
    all_diffs_np,
    cdist_np,
    get_chip_filename,
    id_color_mapping,
    mirror_coordinate,
    overlapping_regions,
    parse_coordinates,
    parse_datetime_from_filename,
    percentage_coordinate,
    scale_coordinate,
    similarityTransform,
)


# T01/T06 --------------------------------------------------------------
@pytest.mark.parametrize(
    "query,expected",
    [
        ("214,243", (214, 243)),
        ("-5,-3", (-5, -3)),  # T06: negative coords accepted unvalidated, no guard
    ],
)
def test_parse_coordinates_happy_path(query, expected):
    request = RequestFactory().get(f"/0.png?{query}")
    assert parse_coordinates(request) == expected


# T02/T03/T04 ------------------------------------------------------------
@pytest.mark.parametrize(
    "url",
    [
        "/0.png",  # T02: zero keys (assert len(keys) == 1, core/utils.py:38)
        "/0.png?214,243&1,2",  # T03: two keys
        "/0.png?214",  # T04a: single key, malformed split (1 part, not 2)
        "/0.png?1,2,3",  # T04b: single key, malformed split (3 parts, not 2)
    ],
)
def test_parse_coordinates_assertion_errors(url):
    """Bug: bare `assert` in production code (core/utils.py:38,41) -- pinned
    as-is, not fixed. No `match=` here: the AssertionError message is empty."""
    request = RequestFactory().get(url)
    with pytest.raises(AssertionError):
        parse_coordinates(request)


# T05 ---------------------------------------------------------------------
@pytest.mark.parametrize("query", ["a,b", "214,b", "1.5,2.5"])
def test_parse_coordinates_value_error(query):
    request = RequestFactory().get(f"/0.png?{query}")
    with pytest.raises(ValueError, match="invalid literal for int"):
        parse_coordinates(request)


# T06 ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "x, y, dst_w, src_w, expected",
    [
        (100, 100, 200, 100, (200, 200)),  # upscale x2
        (100, 100, 50, 100, (50, 50)),  # downscale x0.5
        (100, 100, 100, 100, (100, 100)),  # identity
        (1, 0, 3, 2, (2, 0)),  # rounding: 1*1.5=1.5 -> banker's rounding to 2
    ],
)
def test_scale_coordinate_happy(x, y, dst_w, src_w, expected):
    assert scale_coordinate(x, y, dst_w, src_w) == expected


# T07 ----------------------------------------------------------------------
def test_scale_coordinate_zero_source_width_raises_zero_division_error():
    """Bug B2 (pinned, not fixed): ZeroDivisionError on zero-width source image."""
    with pytest.raises(ZeroDivisionError):
        scale_coordinate(10, 10, 100, 0)


# T08 ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "x, y, dim",
    [
        (0, 0, 1),  # boundary: zero coordinate, minimal dim
        (100, 100, 100),  # identity pin from T06
        (5000, 5000, 5000),  # boundary: max coordinate == max dim
        (1, 4999, 2500),  # asymmetric x/y
        (2500, 1, 1),  # boundary: dim=1
    ],
)
def test_scale_coordinate_identity_when_scale_is_one(x, y, dim):
    assert scale_coordinate(x, y, dim, dim) == (x, y)


@pytest.mark.parametrize(
    "x, y, src_w, dst_w",
    [
        (0, 0, 1, 1),  # boundary: zero coordinate, minimal widths
        (100, 100, 100, 200),  # upscale
        (2000, 2000, 2000, 1),  # extreme downscale
        (1, 999, 500, 1500),  # asymmetric x/y, upscale
        (37, 842, 333, 777),  # arbitrary non-round values
    ],
)
def test_scale_coordinate_round_trip_approx_identity(x, y, src_w, dst_w):
    scaled_x, scaled_y = scale_coordinate(x, y, dst_w, src_w)
    back_x, back_y = scale_coordinate(scaled_x, scaled_y, src_w, dst_w)
    assert abs(back_x - x) <= 1
    assert abs(back_y - y) <= 1


# T09 ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "x, src_w, expected",
    [
        (30, 100, 70),
        (0, 100, 100),
        (100, 100, 0),
        (150, 100, -50),  # edge: x > width -> negative, no guard (pinned as-is)
    ],
)
def test_mirror_coordinate(x, src_w, expected):
    assert mirror_coordinate(x, src_w) == expected


# T10 -----------------------------------------------------------------------
def test_percentage_coordinate_happy():
    left_pct, top_pct = percentage_coordinate(50, 25, 100, 50)
    assert left_pct == pytest.approx(50.0, abs=1e-9)
    assert top_pct == pytest.approx(50.0, abs=1e-9)


# T11 -------------------------------------------------------------------
def test_percentage_coordinate_zero_width_raises_zero_division_error():
    """Bug B2 (pinned, not fixed): ZeroDivisionError on zero-width source image."""
    with pytest.raises(ZeroDivisionError):
        percentage_coordinate(10, 10, 0, 100)


def test_percentage_coordinate_zero_height_raises_zero_division_error():
    """Bug B2 (pinned, not fixed): ZeroDivisionError on zero-height source image."""
    with pytest.raises(ZeroDivisionError):
        percentage_coordinate(10, 10, 100, 0)


# T12 ------------------------------------------------------------------
@pytest.mark.parametrize(
    "filename, dst_ext, expected",
    [
        ("a/b/img.jpg", "png", "img_chip.png"),
        ("x.y.jpeg", "png", "x.y_chip.png"),  # multi-dot
        ("abc", "png", "abc_chip.png"),  # no extension
        ("", "png", "_chip.png"),  # empty filename
        ("a/b/", "png", "_chip.png"),  # trailing slash -> empty basename
    ],
)
def test_get_chip_filename(filename, dst_ext, expected):
    assert get_chip_filename(filename, dst_ext) == expected


# T13 --------------------------------------------------------------------
def test_parse_datetime_from_filename_happy():
    result = parse_datetime_from_filename("PNGP24_---_24_06_15_174811.jpg")
    assert result is not None
    assert result.year == 2024
    assert result.month == 6
    assert result.day == 15


# T14 -----------------------------------------------------------------
@pytest.mark.parametrize(
    "filename",
    [
        "PNGP24_---_noexifdata.jpg",  # explicit noexifdata marker
        "some_random_name.jpg",  # no datetime pattern at all
        "PNGP24_---_24_13_45_999999.jpg",  # invalid: month 13
    ],
)
def test_parse_datetime_from_filename_returns_none(filename):
    assert parse_datetime_from_filename(filename) is None


# T15 ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "metric, expected",
    [
        ("sqeuclidean", 25.0),
        ("euclidean", 5.0),
        ("cityblock", 7.0),
    ],
)
def test_cdist_np_metric_correctness(metric, expected):
    a = np.array([[0.0, 0.0]])
    b = np.array([[3.0, 4.0]])
    result = cdist_np(a, b, metric=metric)
    assert result.shape == (1, 1)
    assert result[0, 0] == pytest.approx(expected, abs=1e-5)


# T16 ------------------------------------------------------------------
def test_cdist_np_unknown_metric_raises_not_implemented_error():
    a = np.array([[0.0, 0.0]])
    b = np.array([[3.0, 4.0]])
    with pytest.raises(NotImplementedError, match="cosine"):
        cdist_np(a, b, metric="cosine")


def _make_array(rows, cols, seed):
    """Deterministic representative float32 array in [-1e3, 1e3] (no NaN/inf)."""
    rng = np.random.default_rng(seed)
    return rng.uniform(-1e3, 1e3, size=(rows, cols)).astype(np.float32)


# T17 -----------------------------------------------------------------------
@pytest.mark.parametrize(
    "a_rows, a_cols, b_batch, seed",
    [
        (1, 1, 1, 0),  # minimal shape all around
        (5, 8, 5, 1),  # max shape (per former Hypothesis bounds)
        (3, 4, 2, 2),  # mid-size, non-square
        (1, 8, 5, 3),  # single row, many cols
        (5, 1, 1, 4),  # many rows, single col
    ],
)
def test_all_diffs_np_shape_invariant(a_rows, a_cols, b_batch, seed):
    a = _make_array(a_rows, a_cols, seed)
    b = np.zeros((b_batch, a.shape[1]), dtype=np.float32)
    result = all_diffs_np(a, b)
    assert result.shape == (a.shape[0], b_batch, a.shape[1])


@pytest.mark.parametrize(
    "rows, cols, seed",
    [
        (1, 1, 10),
        (5, 8, 11),
        (3, 4, 12),
        (2, 6, 13),
    ],
)
def test_cdist_np_self_distance_diagonal_is_near_zero(rows, cols, seed):
    a = _make_array(rows, cols, seed)
    result = cdist_np(a, a, metric="euclidean")
    diag = np.diagonal(result)
    assert np.allclose(diag, 0.0, atol=1e-3)


@pytest.mark.parametrize(
    "a_rows, cols, b_rows, seed_a, seed_b",
    [
        (1, 1, 1, 20, 21),
        (5, 8, 3, 22, 23),
        (2, 4, 5, 24, 25),
        (4, 2, 1, 26, 27),
    ],
)
def test_cdist_np_symmetry(a_rows, cols, b_rows, seed_a, seed_b):
    a = _make_array(a_rows, cols, seed_a)
    b = _make_array(b_rows, cols, seed_b)
    forward = cdist_np(a, b, metric="euclidean")
    backward = cdist_np(b, a, metric="euclidean")
    assert np.allclose(forward, backward.T, atol=1e-3)


# T18 ------------------------------------------------------------------
def test_similarity_transform_maps_in_points_to_out_points():
    in_points = [[100, 200], [100, 50]]
    out_points = [[141, 274], [141, 14]]

    tform = similarityTransform(in_points, out_points)

    assert tform is not None
    assert tform.shape == (2, 3)
    src = np.array([in_points], dtype=np.float32)
    mapped = cv2.transform(src, tform)[0]
    # similarityTransform() derives a 3rd point pair via a 60-degree-rotation
    # trick, rounding that synthetic point to integer coordinates before
    # calling cv2.estimateAffinePartial2D() (a least-squares fit over all 3
    # pairs). That rounding introduces a small, deterministic residual even
    # for the original 2 input/output points -- atol=1e-3 is tighter than
    # the algorithm can achieve by design. atol=0.2 (~1/5 px) comfortably
    # bounds the observed ~0.096 px residual while still catching a real
    # correctness regression.
    np.testing.assert_allclose(mapped, np.array(out_points, dtype=np.float32), atol=0.2)


# T19 ------------------------------------------------------------------
def test_overlapping_regions_returns_overlapping_region(region_stub_cls):
    # ~500m apart (roughly 0.0045 deg latitude), radii sum 4000m > distance
    single = region_stub_cls(origin_latitude=46.0, origin_longitude=8.0, radius=2000)
    other = region_stub_cls(origin_latitude=46.0045, origin_longitude=8.0, radius=2000)

    result = overlapping_regions(single, [other])

    assert result == [other]


def test_overlapping_regions_boundary_exactly_touching_is_not_overlapping(
    region_stub_cls, monkeypatch
):
    import core.utils as utils_module

    single = region_stub_cls(origin_latitude=46.0, origin_longitude=8.0, radius=1000)
    other = region_stub_cls(origin_latitude=46.01, origin_longitude=8.0, radius=1000)

    class _FixedDistance:
        meters = 2000  # exactly single.radius + other.radius

    monkeypatch.setattr(utils_module, "distance", lambda *_a, **_kw: _FixedDistance())

    result = overlapping_regions(single, [other])

    assert result == []  # strict "<" comparison: equal distance is NOT overlapping


# T20 ------------------------------------------------------------------
def test_overlapping_regions_no_overlap_returns_empty_list(region_stub_cls):
    single = region_stub_cls(origin_latitude=46.0, origin_longitude=8.0, radius=2000)
    # ~100km away
    far = region_stub_cls(origin_latitude=47.0, origin_longitude=9.0, radius=2000)

    result = overlapping_regions(single, [far])

    assert result == []


# T21 ------------------------------------------------------------------
@pytest.mark.parametrize(
    "single_kwargs, other_kwargs",
    [
        ({"origin_latitude": None}, {}),
        ({"origin_longitude": None}, {}),
        ({"radius": None}, {}),
        ({}, {"origin_latitude": None}),
        ({}, {"origin_longitude": None}),
        ({}, {"radius": None}),
    ],
)
def test_overlapping_regions_skips_regions_with_none_fields(
    region_stub_cls, single_kwargs, other_kwargs
):
    single_defaults = {"origin_latitude": 46.0, "origin_longitude": 8.0, "radius": 2000}
    other_defaults = {"origin_latitude": 46.0, "origin_longitude": 8.0, "radius": 2000}
    single_defaults.update(single_kwargs)
    other_defaults.update(other_kwargs)

    single = region_stub_cls(**single_defaults)
    other = region_stub_cls(**other_defaults)

    result = overlapping_regions(single, [other])

    assert result == []


# T22 --------------------------------------------------------------------
def test_id_color_mapping_distinct_ids_get_distinct_colors_in_order(chip_stub_cls):
    gallery = [
        (chip_stub_cls(animal_id=1), 1.0),
        (chip_stub_cls(animal_id=2), 2.0),
        (chip_stub_cls(animal_id=3), 3.0),
    ]
    result = id_color_mapping(gallery)
    assert list(result.keys()) == [1, 2, 3]
    assert len(set(result.values())) == 3


def test_id_color_mapping_cycles_modulo_five_beyond_five_ids(chip_stub_cls):
    # 7 distinct ids
    gallery = [(chip_stub_cls(animal_id=i), float(i)) for i in range(1, 8)]
    result = id_color_mapping(gallery)
    assert len(result) == 7
    assert result[1] == result[6]  # 6th new id (index 5) cycles back to color index 0


def test_id_color_mapping_duplicate_id_keeps_same_color(chip_stub_cls):
    gallery = [
        (chip_stub_cls(animal_id=1), 1.0),
        (chip_stub_cls(animal_id=2), 2.0),
        (chip_stub_cls(animal_id=1), 3.0),  # duplicate id, later in gallery
    ]
    result = id_color_mapping(gallery)
    assert len(result) == 2
    assert result[1] != result[2]


def test_id_color_mapping_empty_gallery_returns_empty_dict():
    assert id_color_mapping([]) == {}
