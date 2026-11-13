"""Keep each shared OpenCV network input paired with its inference result."""

# built-in dependencies
from concurrent.futures import ThreadPoolExecutor
from threading import Event

# third-party dependencies
import numpy as np

# project dependencies
from deepface.models.face_detection.opencv import Ssd


def test_concurrent_ssd_calls_keep_their_own_inputs(monkeypatch):
    first_input = Event()
    second_input = Event()
    release_first = Event()
    second_started = Event()

    class StatefulNet:
        def __init__(self):
            self.input = None

        def setInput(self, blob):  # pylint: disable=invalid-name
            self.input = blob
            if not first_input.is_set():
                first_input.set()
                assert release_first.wait(5)
            else:
                second_input.set()

        def forward(self):
            # Encode the input into the face location so crossed requests
            # produce an observable wrong result through the actual decoder.
            left = float(self.input.mean()) / 300
            return np.array([[[[0, 1, 0.99, left, 0.1, left + 0.2, 0.5]]]])

    monkeypatch.setattr(Ssd.SsdClient, "build_model", lambda self: StatefulNet())
    detector = Ssd.SsdClient()

    def detect_second():
        second_started.set()
        return detector.detect_faces(np.full((300, 300, 3), 80, dtype=np.uint8))

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(
            detector.detect_faces, np.full((300, 300, 3), 40, dtype=np.uint8)
        )
        try:
            assert first_input.wait(5)
            second = executor.submit(detect_second)
            assert second_started.wait(5)
            # Without serialization, the second request overwrites the input
            # while the first request is paused inside setInput.
            second_input.wait(1)
        finally:
            release_first.set()
        assert first.result(timeout=5)[0].x == 40
        assert second.result(timeout=5)[0].x == 80
