"""Check min-max postprocessing without loading recognition models."""

# built-in dependencies
from copy import deepcopy

# third-party dependencies
import numpy as np
import pytest

# project dependencies
from deepface.config.minmax import minmax_values
from deepface.modules.normalization import normalize_embedding_minmax


@pytest.mark.parametrize("model_name", sorted(minmax_values))
@pytest.mark.parametrize("layout", ["flat", "batch", "ragged"])
def test_minmax_preserves_values_and_layout(model_name, layout):
    lower, upper = minmax_values[model_name]
    values = [lower - 10, lower, (lower + upper) / 2, upper, upper + 10, 0]
    rows = [values] if layout == "flat" else [values, list(reversed(values))]
    if layout == "ragged":
        rows = [values[:2], [], values]
    source = rows[0] if layout == "flat" else rows
    original = deepcopy(source)
    expected = [
        [(min(max(value, lower), upper) - lower) / (upper - lower) for value in row]
        for row in rows
    ]
    result = normalize_embedding_minmax(model_name, source)
    result_rows = [result] if layout == "flat" else result
    for actual, reference in zip(result_rows, expected):
        assert isinstance(actual, list)
        np.testing.assert_array_equal(actual, reference)
    assert source == original


def test_minmax_handles_empty_and_unconfigured_models():
    assert normalize_embedding_minmax("Facenet", []) == []
    values = [[-2.0, 3.0]]
    assert normalize_embedding_minmax("unconfigured", values) is values


def test_minmax_clips_large_integers_and_infinities_before_scaling():
    values = [-(10**1000), float("-inf"), 0, float("inf"), 10**1000, float("nan")]
    result = normalize_embedding_minmax("VGG-Face", values)
    np.testing.assert_array_equal(result, [0.0, 0.0, 0.0, 1.0, 1.0, float("nan")])
