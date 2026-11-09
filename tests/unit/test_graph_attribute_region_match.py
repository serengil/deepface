from unittest import TestCase
from unittest.mock import patch

import numpy as np

from deepface.modules import datastore


class AttributeAssociationTests(TestCase):
    def test_single_face_from_a_different_region_is_not_assigned(self):
        detected = dict(x=0, y=0, w=10, h=10)
        other = dict(x=100, y=100, w=10, h=10)
        for backend, analysis_area, assigned in (
            ("opencv", other, False), ("opencv", detected, True), ("skip", other, True)
        ):
            with self.subTest(backend=backend, area=analysis_area):
                face = dict(img_index=0, facial_area=detected)
                result = [{"region": analysis_area, "age": 40}]
                with patch.object(datastore, "analyze", return_value=result):
                    getattr(datastore, "__assign_attributes")(
                        [face], [np.zeros((200, 200, 3))], ["age"], backend, True, True, 0, False
                    )
                self.assertEqual(face.get("age"), 40 if assigned else None)
