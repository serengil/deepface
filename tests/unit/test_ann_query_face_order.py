from unittest import TestCase
from unittest.mock import Mock, patch

from deepface.modules import datastore


class ANNFaceOrderTests(TestCase):
    def test_rejected_first_face_does_not_shift_the_second_match(self):
        db = Mock()
        db.search_by_vector.side_effect = [
            [{"id": "far", "img_name": "other", "distance": 100.0}],
            [{"id": "near", "img_name": "self", "distance": 0.0}],
        ]
        faces = [{"embedding": [1.0]}, {"embedding": [2.0]}]
        with patch.object(datastore, "__connect_database", return_value=db), \
             patch.object(datastore, "__get_embeddings", return_value=faces):
            result = datastore.search("query", database_type="qdrant", search_method="ann",
                                      distance_metric="euclidean", return_type="dict")
        self.assertEqual([[row["id"] for row in face] for face in result], [[], ["near"]])
