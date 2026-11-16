# built-in dependencies
import base64
import io
from types import SimpleNamespace
from unittest.mock import Mock

# 3rd party dependencies
import cv2
import numpy as np
import pytest

# project dependencies
from deepface import DeepFace
from deepface.modules import modeling
from deepface.modules.exceptions import SpoofDetected


@pytest.fixture
def models(monkeypatch):
    def forward(images):
        embeddings = images.mean(axis=(1, 2)).tolist()
        return embeddings[0] if len(embeddings) == 1 else embeddings

    recognition = SimpleNamespace(
        input_shape=(6, 4), output_shape=3, forward=Mock(side_effect=forward)
    )
    antispoof = SimpleNamespace(analyze=Mock(return_value=(True, 0.99)))

    def build(task, model_name):
        if task == "facial_recognition":
            return recognition
        assert task == "spoofing"
        assert model_name == "Fasnet"
        return antispoof

    build_model = Mock(side_effect=build)
    monkeypatch.setattr(modeling, "build_model", build_model)
    return recognition, antispoof, build_model


def _image():
    return np.arange(8 * 12 * 3, dtype=np.uint8).reshape(8, 12, 3)


def _source(image, kind, tmp_path):
    if kind == "array":
        return image
    if kind == "list":
        return [image, np.flip(image, axis=0).copy()]
    if kind == "numpy_batch":
        return np.stack([image, np.flip(image, axis=0)])
    success, encoded = cv2.imencode(".png", image)
    assert success
    if kind == "path":
        path = tmp_path / "face.png"
        path.write_bytes(encoded.tobytes())
        return str(path)
    if kind == "stream":
        return io.BytesIO(encoded.tobytes())
    assert kind == "base64"
    return "data:image/png;base64," + base64.b64encode(encoded).decode("ascii")


@pytest.mark.parametrize(
    "kind", ["array", "path", "stream", "base64", "list", "numpy_batch"]
)
def test_skipped_spoof_is_rejected_before_embedding_inference(models, tmp_path, kind):
    recognition, antispoof, _ = models
    antispoof.analyze.return_value = (False, 0.99)
    source = _source(_image(), kind, tmp_path)

    with pytest.raises(SpoofDetected, match="Spoof detected"):
        DeepFace.represent(source, detector_backend="skip", anti_spoofing=True)

    antispoof.analyze.assert_called_once()
    recognition.forward.assert_not_called()


@pytest.mark.parametrize("kind", ["array", "list", "numpy_batch"])
@pytest.mark.parametrize("normalized", [False, True])
def test_real_skipped_faces_keep_pixels_embeddings_and_metadata(
    models, tmp_path, kind, normalized
):
    recognition, antispoof, _ = models
    image = _image()
    if normalized:
        image = image.astype(np.float32) / 255
    original = image.copy()
    source = _source(image, kind, tmp_path)

    expected = DeepFace.represent(source, detector_backend="skip", return_face=True)
    expected_inputs = recognition.forward.call_args.args[0].copy()
    antispoof.analyze.assert_not_called()
    recognition.forward.reset_mock()

    actual = DeepFace.represent(
        source, detector_backend="skip", anti_spoofing=True, return_face=True
    )

    recognition.forward.assert_called_once()
    np.testing.assert_array_equal(
        recognition.forward.call_args.args[0], expected_inputs
    )
    np.testing.assert_array_equal(image, original)
    input_images = [image] if kind == "array" else list(source)
    assert antispoof.analyze.call_count == len(input_images)
    for call, input_image in zip(antispoof.analyze.call_args_list, input_images):
        assert call.kwargs["facial_area"] == (0, 0, 12, 8)
        np.testing.assert_array_equal(call.kwargs["img"], input_image)
        assert call.kwargs["img"].dtype == input_image.dtype

    expected_faces = (
        expected if kind == "array" else [face for faces in expected for face in faces]
    )
    actual_faces = (
        actual if kind == "array" else [face for faces in actual for face in faces]
    )
    assert len(actual) == len(expected)
    assert len(actual_faces) == len(expected_faces)
    for actual_face, expected_face in zip(actual_faces, expected_faces):
        assert actual_face.keys() == expected_face.keys()
        assert actual_face["facial_area"] == {"x": 0, "y": 0, "w": 12, "h": 8}
        assert actual_face["face_confidence"] == expected_face["face_confidence"] == 0
        assert actual_face["embedding"] == expected_face["embedding"]
        np.testing.assert_array_equal(actual_face["face"], expected_face["face"])


@pytest.mark.parametrize("options", [{}, {"anti_spoofing": False}])
def test_disabled_antispoofing_keeps_skipped_representation_lightweight(
    models, options
):
    recognition, antispoof, build_model = models

    result = DeepFace.represent(_image(), detector_backend="skip", **options)

    assert len(result) == 1
    recognition.forward.assert_called_once()
    antispoof.analyze.assert_not_called()
    build_model.assert_called_once_with(
        task="facial_recognition", model_name="VGG-Face"
    )


@pytest.mark.parametrize("kind", ["list", "numpy_batch"])
def test_spoof_later_in_batch_prevents_all_embedding_inference(models, tmp_path, kind):
    recognition, antispoof, _ = models
    antispoof.analyze.side_effect = [(True, 0.99), (False, 0.99)]

    with pytest.raises(SpoofDetected, match="Spoof detected"):
        DeepFace.represent(
            _source(_image(), kind, tmp_path),
            detector_backend="skip",
            anti_spoofing=True,
        )

    assert antispoof.analyze.call_count == 2
    recognition.forward.assert_not_called()


@pytest.mark.parametrize("is_real", [False, True])
def test_extract_faces_still_reports_one_skipped_face_verdict(models, is_real):
    recognition, antispoof, _ = models
    antispoof.analyze.return_value = (is_real, 0.99)

    result = DeepFace.extract_faces(
        _image(), detector_backend="skip", anti_spoofing=True
    )

    assert result[0]["is_real"] is is_real
    antispoof.analyze.assert_called_once()
    recognition.forward.assert_not_called()


@pytest.mark.parametrize("is_real", [False, True])
def test_verify_does_not_repeat_checks_in_internal_skipped_representations(
    models, is_real
):
    recognition, antispoof, _ = models
    antispoof.analyze.return_value = (is_real, 0.99)

    if is_real:
        result = DeepFace.verify(
            _image(), _image(), detector_backend="skip", anti_spoofing=True
        )
        assert result["verified"]
        assert antispoof.analyze.call_count == 2
        assert recognition.forward.call_count == 2
    else:
        with pytest.raises(
            ValueError, match="Exception while processing img1_path"
        ) as exc:
            DeepFace.verify(
                _image(), _image(), detector_backend="skip", anti_spoofing=True
            )
        assert isinstance(exc.value.__cause__, SpoofDetected)
        antispoof.analyze.assert_called_once()
        recognition.forward.assert_not_called()
