# built-in dependencies
import io

# 3rd party dependencies
import numpy as np
import pytest

# project dependencies
from deepface.modules import recognition


def _arrays():
    arrays = {name: np.zeros(2, dtype=np.int64) for name in recognition.DATASTORE_COLUMNS}
    arrays.update(identity=np.array(["first.jpg", "second.jpg"]),
                  hash=np.array(["first-hash", "second-hash"]),
                  embedding=np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32),
                  has_embedding=np.array([True, False]))
    return arrays


def _encode(arrays):
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    return buffer.getvalue()


@pytest.mark.parametrize("column", recognition.DATASTORE_COLUMNS + ["has_embedding"])
def test_truncated_column_cannot_silently_drop_identities(column):
    arrays = _arrays()
    arrays[column] = arrays[column][:1]
    with pytest.raises(ValueError, match="gallery.npz"):
        recognition.__deserialize_npz(_encode(arrays), "gallery.npz")


@pytest.mark.parametrize("column", recognition.DATASTORE_COLUMNS + ["has_embedding"])
def test_wrong_rank_is_rejected(column):
    arrays = _arrays()
    arrays[column] = arrays[column].reshape(-1)
    if column != "embedding":
        arrays[column] = arrays[column].reshape(2, 1)
    with pytest.raises(ValueError, match="gallery.npz"):
        recognition.__deserialize_npz(_encode(arrays), "gallery.npz")


def test_valid_columns_preserve_missing_embedding_and_row_order():
    result = recognition.__deserialize_npz(_encode(_arrays()), "gallery.npz")
    assert result["identity"] == ["first.jpg", "second.jpg"]
    assert result["embedding"] == [[0.0, 1.0], None]


def test_empty_cache_roundtrip():
    result = recognition.__deserialize_npz(recognition.__serialize_npz([]), "gallery.npz")
    assert all(value == [] for value in result.values())
