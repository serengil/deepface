# built-in dependencies
from unittest.mock import Mock

# 3rd party dependencies
import numpy as np
import pytest

# project dependencies
from deepface.modules import streaming


@pytest.mark.parametrize("failure", ["detect", "display", "interrupt", "writer_setup", "directory"])
def test_stream_releases_resources_after_failure(monkeypatch, tmp_path, failure):
    for name in ("__validate_opencv_gui", "build_demography_models",
                 "build_facial_recognition_model", "search_identity"):
        monkeypatch.setattr(streaming, name, Mock())
    cv = Mock()
    cap = cv.VideoCapture.return_value
    cap.get.side_effect = [32, 32, 30]
    cap.read.return_value = (True, np.zeros((32, 32, 3), dtype=np.uint8))
    monkeypatch.setattr(streaming, "cv2", cv)
    monkeypatch.setattr(streaming, "grab_facial_areas", Mock(return_value=[]))
    error = KeyboardInterrupt if failure == "interrupt" else RuntimeError
    if failure in ("detect", "interrupt"):
        streaming.grab_facial_areas.side_effect = error("processing failed")
    elif failure == "display":
        cv.imshow.side_effect = error("display failed")
    elif failure == "writer_setup":
        cv.VideoWriter.side_effect = error("writer creation failed")
    else:
        monkeypatch.setattr(streaming.os, "makedirs", Mock(side_effect=error("directory failed")))

    with pytest.raises(error):
        streaming.analysis("gallery", output_path=str(tmp_path / "output.mp4"))

    cap.release.assert_called_once()
    if failure not in ("writer_setup", "directory"):
        cv.VideoWriter.return_value.release.assert_called_once()
    cv.destroyAllWindows.assert_called_once()


def test_failed_capture_is_released(monkeypatch):
    for name in ("__validate_opencv_gui", "build_demography_models",
                 "build_facial_recognition_model", "search_identity"):
        monkeypatch.setattr(streaming, name, Mock())
    cv = Mock()
    cv.VideoCapture.return_value.isOpened.return_value = False
    monkeypatch.setattr(streaming, "cv2", cv)
    streaming.analysis("gallery")
    cv.VideoCapture.return_value.release.assert_called_once()
