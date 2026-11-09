from unittest import TestCase
from unittest.mock import Mock, patch

import numpy as np

from deepface.api.src.modules.core import routes


class MixedImagesTests(TestCase):
    def test_file_and_reference_can_be_supplied_in_either_order(self):
        pixels = np.zeros((2, 2, 3), dtype=np.uint8)
        for file_key, text_key in (("img1", "img2"), ("img2", "img1")):
            with self.subTest(file_key=file_key):
                upload = Mock(filename="query.png")
                form = Mock()
                form.to_dict.return_value = {text_key: "gallery.jpg"}
                request = Mock(is_json=False, files={file_key: upload}, form=form)
                with patch.object(routes, "request", request), \
                     patch.object(routes.image_utils, "load_image_from_file_storage",
                                  return_value=pixels) as decode:
                    self.assertIs(routes.extract_image_from_request(file_key), pixels)
                    self.assertEqual(routes.extract_image_from_request(text_key), "gallery.jpg")
                decode.assert_called_once_with(upload)
