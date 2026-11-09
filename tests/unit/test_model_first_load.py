from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from time import sleep
from unittest import TestCase
from unittest.mock import patch

from deepface.modules import modeling


class FirstLoadTests(TestCase):
    def test_concurrent_first_requests_share_the_model(self):
        workers = 6
        start = Barrier(workers)

        def build():
            sleep(0.05)
            return object()

        def load():
            start.wait(timeout=5)
            return modeling.build_model("face_detector", "concurrent")

        with patch.object(modeling, "cached_models", {"face_detector": {}}, create=True), \
             patch.object(modeling, "get_model_class", return_value=build) as factory, \
             ThreadPoolExecutor(workers) as pool:
            models = list(pool.map(lambda _: load(), range(workers)))
        self.assertTrue(all(model is models[0] for model in models))
        factory.assert_called_once()
