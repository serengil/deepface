from unittest import TestCase
from unittest.mock import Mock, patch

import numpy as np

from deepface.modules import recognition


class SimilarityConfidenceTests(TestCase):
    def test_lookalikes_retain_the_real_threshold_and_nonmatch_confidence(self):
        gallery = [dict(identity="near", embedding=[1.0]), dict(identity="far", embedding=[8.0])]
        faces = [dict(face=np.zeros((2, 2, 3)), facial_area=dict(x=0, y=0, w=2, h=2))]
        with patch.object(recognition.modeling, "build_model", return_value=Mock(output_shape=1)), \
             patch.object(recognition.representation, "represent", return_value=[{"embedding":[1.0]}]):
            rows = getattr(recognition, "__find_as_dicts")(
                gallery, faces, distance_metric="euclidean", threshold=0.5, similarity_search=True
            )[0]
        self.assertEqual([row["identity"] for row in rows], ["near", "far"])
        self.assertEqual([row["threshold"] for row in rows], [0.5, 0.5])
        self.assertGreater(rows[0]["confidence"], 50)
        self.assertLess(rows[1]["confidence"], 50)
