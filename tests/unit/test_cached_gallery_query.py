from unittest import TestCase
from unittest.mock import Mock, patch

import numpy as np

from deepface.modules import recognition


class CachedListingTests(TestCase):
    def test_existing_cache_can_be_searched_without_listing_objects(self):
        store = Mock()
        store.exists.return_value = True
        store.list_images.side_effect = PermissionError("list denied")
        row = dict(identity="known", hash="cached", embedding=[0.0], target_x=0, target_y=0,
                   target_w=1, target_h=1)
        store.read_bytes.return_value = getattr(recognition, "__serialize_npz")([row])
        with patch.object(recognition, "build_file_store", return_value=store), \
             patch.object(recognition.modeling, "build_model", return_value=Mock(output_shape=1)), \
             patch.object(recognition.representation, "represent", return_value=[{"embedding":[0.0]}]):
            result = recognition.find(np.zeros((2, 2, 3)), "gallery", detector_backend="skip",
                                      distance_metric="euclidean", refresh_database=False,
                                      return_type="dict")
        self.assertEqual(result[0][0]["identity"], "known")
        store.list_images.assert_not_called()
        store.write_bytes.assert_not_called()
