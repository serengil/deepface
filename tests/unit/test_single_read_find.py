import io
from unittest import TestCase
from unittest.mock import Mock, patch

import numpy as np

from deepface.modules import recognition


class SingleReadTests(TestCase):
    def test_query_stream_is_consumed_once(self):
        class Source:
            def __init__(self):
                self.data = io.BytesIO(b"sample")
                self.reads = 0

            def read(self):
                self.reads += 1
                return self.data.read()

        source = Source()
        pixels = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
        store = Mock()
        store.exists.return_value = True
        store.list_images.return_value = []
        row = dict(identity="known", hash="cached", embedding=[1.0], target_x=0, target_y=0,
                   target_w=1, target_h=1)
        store.read_bytes.return_value = getattr(recognition, "__serialize_npz")([row])

        def decode(value):
            if isinstance(value, np.ndarray):
                return value, "array"
            self.assertEqual(value.read(), b"sample")
            return pixels, "io object"

        with patch.object(recognition, "build_file_store", return_value=store), \
             patch.object(recognition.image_utils, "load_image", side_effect=decode), \
             patch.object(recognition.modeling, "build_model", return_value=Mock(output_shape=1)), \
             patch.object(recognition.representation, "represent", return_value=[{"embedding":[1.0]}]):
            result = recognition.find(source, "gallery", detector_backend="skip",
                                      distance_metric="euclidean", refresh_database=False,
                                      return_type="dict")
        self.assertEqual(result[0][0]["identity"], "known")
        self.assertEqual(source.reads, 1)
