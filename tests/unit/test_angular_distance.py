import pytest
import numpy as np
from deepface.modules import verification as target


@pytest.mark.parametrize("dim", [2, 7, 8, 128])
@pytest.mark.parametrize("sign", [1, -1])
def test_single_parallel_vectors(dim, sign):
    a = np.ones(dim, dtype=np.float32)
    with np.errstate(invalid="raise"):
        actual = target.find_angular_distance(a, sign * a)
    assert np.isfinite(actual)
    assert float(actual) == pytest.approx(0 if sign == 1 else 1, abs=2e-4)


@pytest.mark.parametrize("dim", [7, 31])
def test_batch_parallel_vectors(dim):
    a = np.ones((1, dim), dtype=np.float32)
    with np.errstate(invalid="raise"):
        actual = target.find_angular_distance(a, np.concatenate([a, -a]))
    np.testing.assert_allclose(actual, [[0], [1]], atol=2e-4)


@pytest.mark.parametrize(
    "a,b,expected",
    [
        ([1.0, 0.0], [0.0, 1.0], 0.5),
        ([1.0, 0.0], [-1.0, 0.0], 1),
        ([1.0, 0.0], [1.0, 0.0], 0),
    ],
)
def test_float64_controls(a, b, expected):
    assert target.find_angular_distance(a, b) == pytest.approx(expected)


def test_invalid_rank():
    with pytest.raises(ValueError):
        target.find_angular_distance(np.zeros((1, 1, 2)), np.zeros((1, 1, 2)))
