"""A gallery cache must retain the detection policy used to compute its embeddings."""

import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from deepface.models.FacialRecognition import FacialRecognition
from deepface.modules import modeling, recognition
from deepface.modules.filestore import LocalFileStore


class ColorModel(FacialRecognition):
    input_shape = (8, 8)
    output_shape = 3

    def forward(self, image):
        embeddings = image.mean(axis=(1, 2)).tolist()
        return embeddings[0] if len(embeddings) == 1 else embeddings


@pytest.fixture
def gallery(tmp_path, monkeypatch):
    image = cv2.imread(str(Path(__file__).parent / "dataset" / "img1.jpg"))
    assert image is not None
    image = cv2.resize(image, None, fx=0.25, fy=0.25)
    image_path = tmp_path / "face.jpg"
    assert cv2.imwrite(str(image_path), image)
    assert cv2.imwrite(str(tmp_path / "blank.png"), np.zeros((128, 128, 3), np.uint8))
    model = ColorModel()
    original_build = modeling.build_model

    def build_model(task, model_name):
        if task == "facial_recognition":
            return model
        return original_build(task=task, model_name=model_name)

    monkeypatch.setattr(modeling, "build_model", build_model)
    return dict(
        img_path=str(image_path),
        db_path=str(tmp_path),
        model_name="Facenet512",
        detector_backend="opencv",
        align=False,
        distance_metric="euclidean",
        similarity_search=True,
        silent=True,
    )


def blank_distance(results, return_type):
    rows = results[0] if return_type == "dict" else results[0].to_dict("records")
    return next(row["distance"] for row in rows if Path(row["identity"]).name == "blank.png")


@pytest.mark.parametrize("first_enforced", [False, True])
@pytest.mark.parametrize("return_type", ["dict", "pandas"])
def test_switching_detection_policy_rebuilds_the_correct_gallery(
    gallery, first_enforced, return_type, monkeypatch
):
    first = recognition.find(**gallery, enforce_detection=first_enforced, return_type=return_type)
    assert math.isfinite(blank_distance(first, return_type)) is (not first_enforced)

    second = recognition.find(
        **gallery, enforce_detection=not first_enforced, return_type=return_type
    )
    assert math.isfinite(blank_distance(second, return_type)) is first_enforced

    def unexpected_listing(_store):
        raise AssertionError("A matching cache must remain usable without a gallery scan")

    monkeypatch.setattr(LocalFileStore, "list_images", unexpected_listing)
    for enforced in (False, True):
        cached = recognition.find(
            **gallery, enforce_detection=enforced, return_type=return_type, refresh_database=False
        )
        assert math.isfinite(blank_distance(cached, return_type)) is (not enforced)


def test_default_and_explicit_enforcement_share_the_same_cache(gallery, monkeypatch):
    original = recognition.find(**gallery, return_type="dict")

    def unexpected_listing(_store):
        raise AssertionError("Explicit True must reuse the default policy cache")

    monkeypatch.setattr(LocalFileStore, "list_images", unexpected_listing)
    cached = recognition.find(
        **gallery, enforce_detection=True, return_type="dict", refresh_database=False
    )
    assert cached == original
    assert len(list(Path(gallery["db_path"]).glob("*.npz"))) == 1


def test_skipped_detection_shares_a_cache_across_equivalent_policies(gallery, monkeypatch):
    gallery["detector_backend"] = "skip"
    original = recognition.find(**gallery, enforce_detection=False, return_type="dict")

    def unexpected_listing(_store):
        raise AssertionError("Skipped detection does not depend on enforcement")

    monkeypatch.setattr(LocalFileStore, "list_images", unexpected_listing)
    cached = recognition.find(
        **gallery, enforce_detection=True, return_type="dict", refresh_database=False
    )
    assert cached == original
    assert len(list(Path(gallery["db_path"]).glob("*.npz"))) == 1
