from pathlib import Path

import cv2
import numpy as np
import pytest

from deepface import DeepFace
from deepface.models.FacialRecognition import FacialRecognition
from deepface.modules import modeling, recognition
from deepface.modules.filestore import build_file_store


class BinaryColorModel(FacialRecognition):
    input_shape = (4, 4)
    output_shape = 3

    def forward(self, img):
        embeddings = np.rint(img.mean(axis=(1, 2))).tolist()
        return embeddings[0] if len(embeddings) == 1 else embeddings


@pytest.mark.parametrize("dtype", [np.int64, np.float32, np.float64])
@pytest.mark.parametrize("readonly", [False, True])
@pytest.mark.parametrize("missing_embedding", [False, True])
@pytest.mark.parametrize("similarity_search", [False, True])
def test_custom_distance_result_is_not_modified(
    tmp_path, monkeypatch, dtype, readonly, missing_embedding, similarity_search
):
    monkeypatch.setattr(modeling, "build_model", lambda **kwargs: BinaryColorModel())
    colors = {"same.png": (255, 0, 0), "one.png": (255, 255, 0), "two.png": (0, 255, 0)}
    for name, color in colors.items():
        image = np.full((4, 4, 3), color, dtype=np.uint8)
        assert cv2.imwrite(str(tmp_path / name), image)

    query = np.full((4, 4, 3), colors["same.png"], dtype=np.uint8)
    options = dict(
        img_path=query,
        db_path=str(tmp_path),
        model_name="Facenet",
        detector_backend="skip",
        align=False,
        return_type="dict",
        silent=True,
    )
    DeepFace.find(**options, distance_metric="euclidean", threshold=3)

    if missing_embedding:
        store = build_file_store(str(tmp_path))
        cache = str(next(tmp_path.glob("*.npz")))
        records = recognition.__load_representations(store, cache, "npz")
        records.append(
            {
                "identity": str(tmp_path / "unavailable.png"),
                "hash": "unavailable",
                "embedding": None,
                "target_x": 0,
                "target_y": 0,
                "target_w": 0,
                "target_h": 0,
            }
        )
        recognition.__save_representations(store, cache, "npz", records)

    returned = []
    originals = []

    def hamming_count(gallery, queries):
        assert gallery.ndim == queries.ndim == 2
        counts = np.count_nonzero(gallery[None, :, :] != queries[:, None, :], axis=-1)
        distances = counts.astype(dtype)
        originals.append(distances.copy())
        distances.setflags(write=not readonly)
        returned.append(distances)
        return distances

    result = DeepFace.find(
        **options,
        distance_metric=hamming_count,
        threshold=1,
        refresh_database=False,
        similarity_search=similarity_search,
    )

    assert len(returned) == 1
    np.testing.assert_array_equal(returned[0], originals[0])
    assert returned[0].flags.writeable == (not readonly)
    assert len(result) == 1
    names = [Path(row["identity"]).name for row in result[0]]
    expected_names = ["same.png", "one.png"]
    expected_distances = [0, 1]
    if similarity_search:
        expected_names.append("two.png")
        expected_distances.append(2)
        if missing_embedding:
            expected_names.append("unavailable.png")
            expected_distances.append(np.inf)
    assert names == expected_names
    np.testing.assert_array_equal([row["distance"] for row in result[0]], expected_distances)
    for row in result[0]:
        assert (row["source_x"], row["source_y"], row["source_w"], row["source_h"]) == (
            0,
            0,
            4,
            4,
        )
