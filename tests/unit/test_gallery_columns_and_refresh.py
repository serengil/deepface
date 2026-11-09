"""Column integrity and filesystem work when refreshing gallery embeddings."""

import io
from unittest.mock import Mock

import numpy as np
import pytest

from deepface.modules import recognition


def rows(count):
    return [
        dict(
            identity="group.png",
            hash="unchanged",
            embedding=[float(i), 2.0],
            target_x=i,
            target_y=0,
            target_w=10,
            target_h=10,
        )
        for i in range(count)
    ]


def test_mismatched_npz_columns_are_not_silently_truncated():
    serialized = recognition.__serialize_npz(rows(3))
    with np.load(io.BytesIO(serialized), allow_pickle=False) as source:
        columns = dict(source)
    columns["has_embedding"] = np.array([True, False])
    out = io.BytesIO()
    np.savez(out, **columns)
    store = Mock()
    store.read_bytes.return_value = out.getvalue()
    with pytest.raises(ValueError, match="broken.npz"):
        recognition.__load_representations(store, "broken.npz", "npz")


def test_refresh_checks_a_multiface_source_once(monkeypatch):
    store = Mock()
    store.exists.return_value = True
    store.list_images.return_value = ["group.png"]
    store.find_hash.return_value = "unchanged"
    store.read_bytes.return_value = recognition.__serialize_npz(rows(5))
    monkeypatch.setattr(recognition, "build_file_store", lambda _: store)
    monkeypatch.setattr(recognition.detection, "extract_faces", lambda **kwargs: [])
    monkeypatch.setattr(recognition, "__find_as_dicts", lambda **kwargs: [])
    recognition.find(np.zeros((2, 2, 3), np.uint8), "gallery", return_type="dict", silent=True)
    store.find_hash.assert_called_once_with("group.png")


def test_mixed_optional_embeddings_roundtrip():
    gallery = rows(3)
    gallery[1]["embedding"] = None
    content = recognition.__serialize_npz(gallery)
    assert recognition.__deserialize_npz(content, "test.npz")["embedding"] == [
        [0.0, 2.0],
        None,
        [2.0, 2.0],
    ]


def test_top_k_returns_matching_face_metadata(monkeypatch):
    monkeypatch.setattr(recognition.modeling, "build_model", lambda **unused: Mock(output_shape=2))
    monkeypatch.setattr(
        recognition.representation, "represent", lambda **unused: [{"embedding": [3.0, 2.0]}]
    )
    query = {"face": np.zeros((2, 2, 3)), "facial_area": dict(x=1, y=2, w=3, h=4)}
    results = recognition.__find_as_dicts(
        rows(8), [query], distance_metric="euclidean", threshold=100, k=1
    )
    assert [r["target_x"] for r in results[0]] == [3]
    assert results[0][0]["source_y"] == 2
