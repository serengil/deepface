# built-in dependencies
import sys

# 3rd party dependencies
import cv2
import numpy as np
import pytest

# project dependencies
from deepface.modules import streaming
from deepface.commons.logger import Logger

logger = Logger()

ARABIC_LABEL = "my_db/محمد.jpg (85.3%)"


def blank() -> np.ndarray:
    return np.zeros((40, 320, 3), np.uint8)


@pytest.mark.parametrize(
    "label", ["my_db/img1.jpg (85.3%)", "my_db/José.jpg (85.3%)", "my_db/王小明.jpg (85.3%)"]
)
def test_labels_without_arabic_are_written_by_opencv(label):
    img = blank()
    streaming.put_label(img, label, (5, 25))

    expected = blank()
    cv2.putText(expected, label, (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, streaming.TEXT_COLOR, 1)
    assert np.array_equal(img, expected)
    logger.info("✅ test labels without arabic are written by opencv done")


def test_arabic_label_is_written_by_khatt():
    khatt = pytest.importorskip("khatt")
    img = blank()
    streaming.put_label(img, ARABIC_LABEL, (5, 25))

    expected = blank()
    khatt.put_text(
        expected, ARABIC_LABEL, (5, 25), font="cairo", size_px=15, color=streaming.TEXT_COLOR
    )
    assert img.any()
    assert np.array_equal(img, expected)
    logger.info("✅ test arabic label is written by khatt done")


def test_arabic_label_falls_back_to_opencv_without_khatt(monkeypatch):
    # importing khatt raises ImportError, as if it were not installed
    monkeypatch.setitem(sys.modules, "khatt", None)
    img = blank()
    streaming.put_label(img, ARABIC_LABEL, (5, 25))

    expected = blank()
    cv2.putText(
        expected, ARABIC_LABEL, (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, streaming.TEXT_COLOR, 1
    )
    assert np.array_equal(img, expected)
    logger.info("✅ test arabic label falls back to opencv without khatt done")
