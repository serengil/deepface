"""Verify every detected face using one representation batch per input image."""

import json
import statistics
import time

import numpy as np
import pytest

from deepface import DeepFace
from deepface.modules import modeling, representation, verification
from deepface.modules.exceptions import SpoofDetected


class ColorModel:
    input_shape = (8, 8)
    output_shape = 128

    def __init__(self):
        self.calls = []

    def forward(self, images):
        self.calls.append(images.copy())
        colors = images.mean(axis=(1, 2))
        vectors = np.tile(colors, (1, 43))[:, :128]
        result = vectors.tolist()
        return result[0] if len(result) == 1 else result


def faces(count):
    return [
        {
            "face": np.full((5 + i, 7 + i, 3), [0.15 + 0.1 * i, 0.4, 0.8 - 0.1 * i]),
            "facial_area": {"x": 10 * i, "y": i, "w": 7 + i, "h": 5 + i},
            "confidence": 0.9,
            "is_real": True,
        }
        for i in range(count)
    ]


def scalar_metric(first, second):
    assert first.ndim == second.ndim == 1
    return float(np.linalg.norm(first - second))


@pytest.mark.parametrize("count", [1, 3])
@pytest.mark.parametrize(
    "metric", ["cosine", "euclidean", "euclidean_l2", "angular", scalar_metric]
)
@pytest.mark.parametrize("normalization", ["base", "Facenet2018"])
def test_verify_batches_faces_without_changing_matching(monkeypatch, count, metric, normalization):
    model = ColorModel()
    monkeypatch.setattr(modeling, "build_model", lambda **kwargs: model)
    first_faces = faces(count)
    second_faces = list(reversed(faces(count)))
    extracted = {"first": first_faces, "second": second_faces}
    monkeypatch.setattr(
        verification.detection, "extract_faces", lambda img_path, **kwargs: extracted[img_path]
    )
    references = []
    for items in (first_faces, second_faces):
        references.append(
            [
                representation.represent(
                    item["face"][:, :, ::-1],
                    model_name="Facenet",
                    detector_backend="skip",
                    normalization=normalization,
                )[0]["embedding"]
                for item in items
            ]
        )
    candidates = [
        (float(verification.find_distance(a, b, metric)), i, j)
        for i, a in enumerate(references[0])
        for j, b in enumerate(references[1])
    ]
    distance, first_index, second_index = min(candidates)
    model.calls.clear()

    result = DeepFace.verify(
        "first",
        "second",
        model_name="Facenet",
        detector_backend="skip",
        distance_metric=metric,
        normalization=normalization,
        threshold=0.2,
    )

    assert result["distance"] == distance
    assert result["verified"] is (distance <= 0.2)
    assert result["facial_areas"] == {
        "img1": first_faces[first_index]["facial_area"],
        "img2": second_faces[second_index]["facial_area"],
    }
    assert [len(call) for call in model.calls] == [count, count]


@pytest.mark.parametrize("spoof_position", [0, 2])
def test_spoof_rejection_precedes_embedding_inference(monkeypatch, spoof_position):
    model = ColorModel()
    monkeypatch.setattr(modeling, "build_model", lambda **kwargs: model)
    items = faces(3)
    items[spoof_position]["is_real"] = False
    monkeypatch.setattr(verification.detection, "extract_faces", lambda **kwargs: items)
    with pytest.raises(SpoofDetected, match="Spoof detected"):
        verification.__extract_faces_and_embeddings(
            "faces", model_name="Facenet", anti_spoofing=True
        )
    assert not model.calls


def test_empty_detection_keeps_empty_embeddings(monkeypatch):
    model = ColorModel()
    monkeypatch.setattr(modeling, "build_model", lambda **kwargs: model)
    monkeypatch.setattr(verification.detection, "extract_faces", lambda **kwargs: [])
    assert verification.__extract_faces_and_embeddings("empty") == ([], [])
    assert not model.calls


def test_precalculated_embeddings_do_not_run_inference(monkeypatch):
    model = ColorModel()
    monkeypatch.setattr(modeling, "build_model", lambda **kwargs: model)
    result = DeepFace.verify([0.5] * 128, [0.5] * 128, model_name="Facenet", silent=True)
    assert result["verified"]
    assert result["distance"] == 0
    assert not model.calls


def test_real_facenet_batch_outputs_and_timing(monkeypatch):
    """Measure pretrained FaceNet inference; detection supplies deterministic face crops."""
    import onnxruntime as ort

    model = modeling.build_model(task="facial_recognition", model_name="Facenet")
    from deepface.commons import folder_utils

    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    options.log_severity_level = 3
    model.model = ort.InferenceSession(
        f"{folder_utils.get_deepface_home()}/.deepface/weights/facenet_weights.onnx",
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    rng = np.random.default_rng(919)
    items = [
        {
            "face": rng.uniform(0, 1, (48 + i, 52 + i, 3)).astype("float32"),
            "facial_area": {"x": i * 10, "y": 2 * i, "w": 52 + i, "h": 48 + i},
            "confidence": 1.0,
        }
        for i in range(8)
    ]
    monkeypatch.setattr(verification.detection, "extract_faces", lambda **kwargs: items)
    expected = [
        representation.represent(
            item["face"][:, :, ::-1], model_name="Facenet", detector_backend="skip"
        )[0]["embedding"]
        for item in items
    ]
    original_forward = model.forward
    calls = []

    def measured_forward(images):
        calls.append(len(images))
        return original_forward(images)

    monkeypatch.setattr(model, "forward", measured_forward)

    def operation():
        return verification.__extract_faces_and_embeddings(
            "pre-extracted", model_name="Facenet", detector_backend="skip"
        )

    actual, regions = operation()
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
    assert regions == [item["facial_area"] for item in items]
    observed_calls = list(calls)
    durations = []
    for _ in range(5):
        start = time.perf_counter()
        embeddings, actual_regions = operation()
        durations.append((time.perf_counter() - start) * 1000)
        np.testing.assert_allclose(embeddings, expected, rtol=2e-5, atol=2e-5)
        assert actual_regions == regions
    print(
        json.dumps(
            {
                "onnxruntime": ort.__version__,
                "model": "pretrained Facenet",
                "faces": len(items),
                "threads": 2,
                "forward_batch_sizes": observed_calls,
                "median_ms": statistics.median(durations),
                "max_absolute_embedding_difference": float(
                    np.max(np.abs(np.asarray(actual) - np.asarray(expected)))
                ),
            }
        )
    )
