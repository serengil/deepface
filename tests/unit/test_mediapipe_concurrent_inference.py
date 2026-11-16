"""Shared MediaPipe graphs must finish each request before accepting another."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest

from deepface.models.face_detection.external.MediaPipe import MediaPipeClient


def test_concurrent_requests_keep_their_own_detections(monkeypatch):
    first_entered = Event()
    second_entered = Event()
    second_started = Event()
    release_first = Event()
    face = SimpleNamespace(
        score=[0.9],
        location_data=SimpleNamespace(
            relative_bounding_box=SimpleNamespace(xmin=0.25, ymin=0.25, width=0.5, height=0.5),
            relative_keypoints=[SimpleNamespace(x=0.4, y=0.4), SimpleNamespace(x=0.6, y=0.4)],
        ),
    )

    class StatefulGraph:
        def __init__(self):
            self.output = None

        def process(self, image):
            self.output = SimpleNamespace(detections=[face] if image.any() else None)
            if not first_entered.is_set():
                first_entered.set()
                assert release_first.wait(5)
            else:
                second_entered.set()
            return self.output

    monkeypatch.setattr(MediaPipeClient, "build_model", lambda self: StatefulGraph())
    client = MediaPipeClient()

    def second_request():
        second_started.set()
        return client.detect_faces(np.ones((16, 16, 3), dtype=np.uint8))

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(client.detect_faces, np.zeros((16, 16, 3), dtype=np.uint8))
        try:
            assert first_entered.wait(5)
            second = pool.submit(second_request)
            assert second_started.wait(5)
            second_entered.wait(0.2)
        finally:
            release_first.set()
        assert first.result(timeout=5) == []
        regions = second.result(timeout=5)
        assert len(regions) == 1
        assert (regions[0].x, regions[0].y, regions[0].w, regions[0].h) == (4, 4, 8, 8)


def test_independent_clients_can_process_at_the_same_time(monkeypatch):
    first_entered = Event()
    second_entered = Event()

    class IndependentGraph:
        def process(self, image):
            entered, other = (
                (first_entered, second_entered)
                if image.any()
                else (
                    second_entered,
                    first_entered,
                )
            )
            entered.set()
            assert other.wait(5)
            return SimpleNamespace(detections=None)

    monkeypatch.setattr(MediaPipeClient, "build_model", lambda self: IndependentGraph())
    first, second = MediaPipeClient(), MediaPipeClient()
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = [
            pool.submit(first.detect_faces, np.ones((16, 16, 3), dtype=np.uint8)),
            pool.submit(second.detect_faces, np.zeros((16, 16, 3), dtype=np.uint8)),
        ]
        assert [result.result(timeout=5) for result in results] == [[], []]


def test_failed_request_releases_the_graph_for_another_thread(monkeypatch):
    class RecoveringGraph:
        def process(self, image):
            if not image.any():
                raise ValueError("invalid frame")
            return SimpleNamespace(detections=None)

    monkeypatch.setattr(MediaPipeClient, "build_model", lambda self: RecoveringGraph())
    client = MediaPipeClient()
    with pytest.raises(ValueError, match="invalid frame"):
        client.detect_faces(np.zeros((16, 16, 3), dtype=np.uint8))
    with ThreadPoolExecutor(max_workers=1) as pool:
        result = pool.submit(client.detect_faces, np.ones((16, 16, 3), dtype=np.uint8))
        assert result.result(timeout=5) == []
