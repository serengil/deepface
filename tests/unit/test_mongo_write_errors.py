from unittest import TestCase
from unittest.mock import Mock

import numpy as np

from deepface.modules.database.mongo import MongoDbClient
from deepface.modules.exceptions import DuplicateEntryError


class BulkFailure(Exception):
    def __init__(self, codes, concern=False):
        super().__init__("bulk write failed")
        self.details = {
            "writeErrors": [{"code": code} for code in codes],
            "writeConcernErrors": [{"code": 64}] if concern else [],
        }


class WriteFailureTests(TestCase):
    def setUp(self):
        self.client = object.__new__(MongoDbClient)
        self.client.embeddings = Mock()
        self.client.counters = Mock()
        self.client.counters.find_one_and_update.return_value = {"seq": 1}
        self.client.BulkWriteError = BulkFailure
        self.client.DuplicateKeyError = KeyError
        self.client.Binary = bytes
        self.row = dict(img_name="face", face=np.zeros((2, 2)), embedding=[1.0],
                        model_name="Facenet", detector_backend="skip",
                        aligned=True, l2_normalized=False)

    def test_original_errors_propagate(self):
        for exc in (BulkFailure([121]), BulkFailure([11000, 121]), BulkFailure([11000], True)):
            for count in (1, 2):
                with self.subTest(exc=exc.details, count=count):
                    self.client.embeddings.insert_many.side_effect = exc
                    with self.assertRaises(BulkFailure) as caught:
                        self.client.insert_embeddings([self.row] * count)
                    self.assertIs(caught.exception, exc)

    def test_duplicate_only_behavior_is_preserved(self):
        self.client.embeddings.insert_many.side_effect = BulkFailure([11000])
        self.assertEqual(self.client.insert_embeddings([self.row]), 0)
        with self.assertRaises(DuplicateEntryError):
            self.client.insert_embeddings([self.row, self.row])
