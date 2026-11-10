from unittest import TestCase
from unittest.mock import Mock, patch

import numpy as np

from deepface.modules import recognition


class DefaultDictThresholdTests(TestCase):
    def find_matches(self, **kwargs):
        gallery = [
            dict(identity="same", embedding=[1.0]),
            dict(identity="near", embedding=[1.25]),
            dict(identity="far", embedding=[8.0]),
        ]
        faces = [dict(face=np.zeros((2, 2, 3)), facial_area=dict(x=0, y=0, w=2, h=2))]
        with (
            patch.object(
                recognition.modeling, "build_model", return_value=Mock(output_shape=1)
            ),
            patch.object(
                recognition.representation,
                "represent",
                return_value=[{"embedding": [1.0]}],
            ),
        ):
            return getattr(recognition, "__find_as_dicts")(
                gallery, faces, distance_metric="euclidean", **kwargs
            )[0]

    def test_missing_threshold_uses_the_model_default(self):
        rows = self.find_matches()
        threshold = recognition.verification.find_threshold("VGG-Face", "euclidean")
        self.assertEqual([row["identity"] for row in rows], ["same", "near"])
        self.assertEqual([row["threshold"] for row in rows], [threshold, threshold])

    def test_explicit_zero_threshold_is_preserved(self):
        rows = self.find_matches(threshold=0.0)
        self.assertEqual([row["identity"] for row in rows], ["same"])
        self.assertEqual(rows[0]["threshold"], 0.0)

    def test_similarity_search_keeps_the_existing_infinite_threshold(self):
        rows = self.find_matches(similarity_search=True)
        self.assertEqual([row["identity"] for row in rows], ["same", "near", "far"])
        self.assertTrue(all(np.isinf(row["threshold"]) for row in rows))
