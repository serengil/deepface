"""Cached Haar classifiers must finish one native call before accepting another."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest

from deepface.models.face_detection.opencv.OpenCv import OpenCvClient


def make_client(monkeypatch, operation, classify):
    def face_call(image, *_args, **_kwargs):
        boxes = classify(image) if operation == "face" else []
        return boxes, [], [0.9] * len(boxes)

    def eye_call(image, *_args):
        return classify(image) if operation == "eyes" else []

    monkeypatch.setattr(
        OpenCvClient,
        "build_model",
        lambda self: {
            "face_detector": SimpleNamespace(detectMultiScale3=face_call),
            "eye_detector": SimpleNamespace(detectMultiScale=eye_call),
        },
    )
    return OpenCvClient()


def request(client, operation, has_face):
    image = np.full((16, 16, 3), int(has_face), dtype=np.uint8)
    return client.detect_faces(image) if operation == "face" else client.find_eyes(image)


@pytest.mark.parametrize("operation", ["face", "eyes"])
def test_shared_classifier_keeps_results_with_their_request(monkeypatch, operation):
    first_entered, second_entered, second_started, release_first = (Event() for _ in range(4))
    output = []

    def classify(image):
        nonlocal output
        output = [[2, 2, 4, 4], [8, 2, 4, 4]] if image.any() else []
        if not first_entered.is_set():
            first_entered.set()
            assert release_first.wait(5)
        else:
            second_entered.set()
        return output

    client = make_client(monkeypatch, operation, classify)

    def second_request():
        second_started.set()
        return request(client, operation, True)

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(request, client, operation, False)
        try:
            assert first_entered.wait(5)
            second = pool.submit(second_request)
            assert second_started.wait(5)
            second_entered.wait(0.2)
        finally:
            release_first.set()
        assert first.result(timeout=5) == ([] if operation == "face" else (None, None))
        assert second.result(timeout=5) != ([] if operation == "face" else (None, None))


@pytest.mark.parametrize("operation", ["face", "eyes"])
def test_independent_classifiers_can_run_concurrently(monkeypatch, operation):
    first_entered, second_entered = Event(), Event()

    def classify(image):
        entered, other = (
            (first_entered, second_entered) if image.any() else (second_entered, first_entered)
        )
        entered.set()
        assert other.wait(5)
        return []

    first = make_client(monkeypatch, operation, classify)
    second = make_client(monkeypatch, operation, classify)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = [
            pool.submit(request, first, operation, True),
            pool.submit(request, second, operation, False),
        ]
        empty = [] if operation == "face" else (None, None)
        assert [result.result(timeout=5) for result in results] == [empty, empty]


@pytest.mark.parametrize("operation", ["face", "eyes"])
def test_failed_native_call_releases_the_classifier(monkeypatch, operation):
    def classify(image):
        if not image.any():
            raise ValueError("invalid frame")
        return []

    client = make_client(monkeypatch, operation, classify)
    if operation == "face":
        # The face-detector API already handles native errors as an empty result.
        assert request(client, operation, False) == []
    else:
        with pytest.raises(ValueError, match="invalid frame"):
            request(client, operation, False)
    with ThreadPoolExecutor(max_workers=1) as pool:
        result = pool.submit(request, client, operation, True)
        assert result.result(timeout=5) == ([] if operation == "face" else (None, None))
