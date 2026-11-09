"""Metric invariants for precomputed directions and integer embeddings."""

import numpy as np
import pytest

from deepface.modules import verification


@pytest.mark.parametrize("scale", [1e250, 1e-250])
@pytest.mark.parametrize("metric", ["cosine", "angular"])
def test_direction_is_independent_of_vector_scale(scale, metric):
    left = np.array([3.0, 4.0])
    right = np.array([4.0, -3.0])
    expected = 1.0 if metric == "cosine" else 0.5
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        scalar = verification.find_distance(left * scale, right * scale, metric)
        matrix = verification.find_distance(
            (left * scale)[None, :], (right * scale)[None, :], metric
        )
    np.testing.assert_allclose([scalar, matrix[0, 0]], expected)


@pytest.mark.parametrize("metric", ["cosine", "angular", "euclidean"])
def test_integer_distance_uses_real_arithmetic(metric):
    a = np.array([0, 240], np.uint8)
    b = np.array([255, 240], np.uint8)
    expected = verification.find_distance(a.astype(float), b.astype(float), metric)
    assert verification.find_distance(a, b, metric) == expected
    assert verification.find_distance(a[None], b[None], metric)[0, 0] == expected
