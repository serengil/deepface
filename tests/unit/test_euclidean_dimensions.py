# 3rd party dependencies
import numpy as np
import pytest

# project dependencies
from deepface.modules import verification


@pytest.mark.parametrize("source, target", [([1.0], [1.0, 1.0, 1.0]),
                                         ([1.0, 1.0, 1.0], [1.0]),
                                         ([[1.0], [2.0]], [[1.0, 1.0, 1.0]]),
                                         ([[1.0, 1.0, 1.0]], [[1.0], [2.0]])])
@pytest.mark.parametrize("metric", ["euclidean", "euclidean_l2"])
def test_incompatible_embeddings_cannot_broadcast(source, target, metric):
    with pytest.raises(ValueError, match="dimension"):
        verification.find_distance(source, target, metric)


def test_direct_euclidean_call_rejects_broadcasting():
    with pytest.raises(ValueError, match="dimension"):
        verification.find_euclidean_distance([1.0], [1.0, 1.0])


def test_valid_batched_distances_keep_query_by_gallery_layout():
    actual = verification.find_euclidean_distance([[0.0, 0.0], [3.0, 0.0]],
                                                  [[0.0, 4.0], [3.0, 4.0], [0.0, 0.0]])
    np.testing.assert_allclose(actual, [[4, 5], [5, 4], [0, 3]])


def test_custom_metric_can_handle_different_dimensions():
    result = verification.find_distance([1.0], [1.0, 1.0], lambda source, target: 7.0)
    assert result == 7.0
