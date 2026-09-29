"""Bound Euclidean scratch space without changing direct-difference numerics."""

import numpy as np
import pytest

from deepface.modules import verification


def direct_reference(source, target):
    return np.linalg.norm(
        np.asarray(source)[None, :, :] - np.asarray(target)[:, None, :], axis=2
    )


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64, np.int32])
@pytest.mark.parametrize("shape", [(11, 5, 7), (3, 17, 5)])
def test_chunked_distances_match_original_values_and_dtype(monkeypatch, dtype, shape):
    n_source, n_target, dimensions = shape
    rng = np.random.default_rng(137)
    source = (rng.normal(size=(n_source, dimensions)) * 3).astype(dtype)
    target = (rng.normal(size=(n_target, dimensions)) * 3).astype(dtype)
    before_source, before_target = source.copy(), target.copy()
    monkeypatch.setattr(
        verification, "_EUCLIDEAN_DISTANCE_BATCH_BYTES", 256, raising=False
    )
    actual = verification.find_euclidean_distance(source, target)
    expected = direct_reference(source, target)
    assert actual.shape == (n_target, n_source)
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(source, before_source)
    np.testing.assert_array_equal(target, before_target)


@pytest.mark.parametrize(
    "shape,budget", [((11, 5, 7), 256), ((3, 17, 5), 512), ((17, 3, 64), 1024)]
)
def test_norm_receives_only_bounded_tiles(monkeypatch, shape, budget):
    n_source, n_target, dimensions = shape
    source = np.arange(n_source * dimensions, dtype=np.float64).reshape(
        n_source, dimensions
    )
    target = np.arange(n_target * dimensions, dtype=np.float64).reshape(
        n_target, dimensions
    )
    expected = direct_reference(source, target)
    original_norm = np.linalg.norm
    tiles = []

    def bounded_norm(array, *args, **kwargs):
        if np.ndim(array) == 3:
            assert (
                array.nbytes <= budget
            ), f"unbounded difference allocation: {array.nbytes} bytes"
            tiles.append(array.shape)
        return original_norm(array, *args, **kwargs)

    monkeypatch.setattr(
        verification, "_EUCLIDEAN_DISTANCE_BATCH_BYTES", budget, raising=False
    )
    monkeypatch.setattr(verification.np.linalg, "norm", bounded_norm)
    actual = verification.find_euclidean_distance(source, target)
    assert len(tiles) > 1
    assert sum(shape[0] * shape[1] for shape in tiles) == n_source * n_target
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("shape", [(0, 4, 7), (3, 0, 7), (0, 0, 7), (3, 2, 0)])
@pytest.mark.parametrize("dtype", [np.float32, np.int64])
def test_empty_axes_keep_original_shape_and_dtype(monkeypatch, shape, dtype):
    n_source, n_target, dimensions = shape
    source = np.zeros((n_source, dimensions), dtype=dtype)
    target = np.zeros((n_target, dimensions), dtype=dtype)
    monkeypatch.setattr(
        verification, "_EUCLIDEAN_DISTANCE_BATCH_BYTES", 64, raising=False
    )
    actual = verification.find_euclidean_distance(source, target)
    expected = direct_reference(source, target)
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual, expected)


def test_readonly_noncontiguous_and_mixed_dtype_inputs(monkeypatch):
    source = np.arange(132, dtype=np.float32).reshape(11, 12)[:, ::2]
    target = np.arange(60, dtype=np.float64).reshape(5, 12)[:, ::2]
    source.flags.writeable = False
    target.flags.writeable = False
    monkeypatch.setattr(
        verification, "_EUCLIDEAN_DISTANCE_BATCH_BYTES", 256, raising=False
    )
    np.testing.assert_array_equal(
        verification.find_euclidean_distance(source, target),
        direct_reference(source, target),
    )


def test_large_offset_small_separations_do_not_cancel(monkeypatch):
    source = np.full((5, 32), 1e8, dtype=np.float64)
    target = source[:3].copy()
    source[1, 0] += 1e-3
    target[2, 1] += 2e-3
    monkeypatch.setattr(
        verification, "_EUCLIDEAN_DISTANCE_BATCH_BYTES", 512, raising=False
    )
    actual = verification.find_euclidean_distance(source, target)
    np.testing.assert_array_equal(actual, direct_reference(source, target))
    assert actual[0, 0] == 0
    assert actual[0, 1] > 0
    assert actual[2, 0] > 0


@pytest.mark.parametrize("metric", ["euclidean", "euclidean_l2"])
def test_public_distance_dispatch_rounding_and_threshold_decisions(monkeypatch, metric):
    source = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    target = np.array([[1.0, 0.0], [0.2, 0.8]])
    monkeypatch.setattr(
        verification, "_EUCLIDEAN_DISTANCE_BATCH_BYTES", 32, raising=False
    )
    actual = verification.find_distance(source, target, metric)
    if metric == "euclidean_l2":
        source = verification.l2_normalize(source, axis=1)
        target = verification.l2_normalize(target, axis=1)
    reference = np.round(direct_reference(source, target), 6)
    np.testing.assert_array_equal(actual, reference)
    threshold = verification.find_threshold("Facenet512", metric)
    np.testing.assert_array_equal(actual <= threshold, reference <= threshold)


def test_python_lists_and_scalar_path_are_unchanged():
    np.testing.assert_array_equal(
        verification.find_euclidean_distance([[1.0, 2.0]], [[4.0, 6.0]]), [[5.0]]
    )
    assert verification.find_euclidean_distance([1.0, 2.0], [4.0, 6.0]) == 5.0


@pytest.mark.parametrize(
    "source,target",
    [
        (np.zeros((2, 3)), np.zeros((4, 5))),
        (np.zeros((2, 3)), np.zeros(3)),
        (np.zeros((2, 3, 1)), np.zeros((2, 3, 1))),
    ],
)
def test_invalid_shapes_still_raise(source, target):
    with pytest.raises(ValueError):
        verification.find_euclidean_distance(source, target)


def test_single_embedding_larger_than_tile_budget_still_computes(monkeypatch):
    source = np.arange(64, dtype=np.float64).reshape(2, 32)
    target = -source
    monkeypatch.setattr(
        verification, "_EUCLIDEAN_DISTANCE_BATCH_BYTES", 8, raising=False
    )
    np.testing.assert_array_equal(
        verification.find_euclidean_distance(source, target),
        direct_reference(source, target),
    )


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
@pytest.mark.parametrize("orders", [("C", "C"), ("C", "F"), ("F", "C"), ("F", "F")])
def test_mixed_memory_layouts_preserve_the_original_reduction(
    monkeypatch, dtype, orders
):
    # Force singleton tail tiles. A direct allocation without preserving layout
    # can switch between pairwise and sequential accumulation in np.linalg.norm.
    source = np.array(
        np.random.default_rng(42).normal(size=(11, 127)), dtype=dtype, order=orders[0]
    )
    target = np.array(
        np.random.default_rng(32).normal(size=(3, 127)), dtype=dtype, order=orders[1]
    )
    monkeypatch.setattr(
        verification, "_EUCLIDEAN_DISTANCE_BATCH_BYTES", 512, raising=False
    )
    np.testing.assert_array_equal(
        verification.find_euclidean_distance(source, target),
        direct_reference(source, target),
    )


def test_fortran_singleton_tail_stays_bounded(monkeypatch):
    source = np.asfortranarray(np.arange(15, dtype=np.float64).reshape(5, 3))
    target = np.asfortranarray(np.arange(9, dtype=np.float64).reshape(3, 3))
    expected = direct_reference(source, target)
    norm = np.linalg.norm
    tile_count = []

    def bounded_norm(array, *args, **kwargs):
        if np.ndim(array) == 3:
            assert array.nbytes <= 48
            tile_count.append(array.shape)
        return norm(array, *args, **kwargs)

    monkeypatch.setattr(
        verification, "_EUCLIDEAN_DISTANCE_BATCH_BYTES", 48, raising=False
    )
    monkeypatch.setattr(verification.np.linalg, "norm", bounded_norm)
    actual = verification.find_euclidean_distance(source, target)
    assert len(tile_count) > 1
    np.testing.assert_array_equal(actual, expected)
