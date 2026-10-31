import cv2
import numpy as np
import pytest

from deepface import DeepFace


@pytest.mark.parametrize(
    "height,width", [(20, 30), (30, 20), (20, 20), (1, 20), (20, 1), (1, 1)]
)
@pytest.mark.parametrize("file_input", [False, True])
def test_skipped_representation_reports_original_image_dimensions(
    tmp_path, height, width, file_input
):
    image = np.full((height, width, 3), [20, 100, 220], dtype=np.uint8)
    source = image
    if file_input:
        path = tmp_path / "input.png"
        assert cv2.imwrite(str(path), image)
        source = str(path)

    result = DeepFace.represent(source, model_name="SFace", detector_backend="skip")[0]
    extracted = DeepFace.extract_faces(source, detector_backend="skip")[0]

    expected = {"x": 0, "y": 0, "w": width, "h": height}
    assert result["facial_area"] == expected
    assert {key: extracted["facial_area"][key] for key in expected} == expected
    assert len(result["embedding"]) == 128
    assert np.isfinite(result["embedding"]).all()
