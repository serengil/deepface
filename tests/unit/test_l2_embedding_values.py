"""L2 postprocessing checks that do not load recognition models."""

from copy import deepcopy
from unittest import TestCase

import numpy as np

from deepface.modules.normalization import normalize_embedding_l2


class L2EmbeddingValuesTests(TestCase):
    def test_single_embedding_keeps_values_shape_and_input(self):
        source = [3.0, -4.0, 0.0]
        original = source.copy()
        result = normalize_embedding_l2(source)
        self.assertIsInstance(result, list)
        np.testing.assert_array_equal(result, [0.6, -0.8, 0.0])
        self.assertEqual(source, original)

    def test_batch_keeps_each_embeddings_normalization_independent(self):
        source = [[3, 4], [5, 12], [-8, 15]]
        original = deepcopy(source)
        result = normalize_embedding_l2(source)
        self.assertEqual(len(result), 3)
        for actual, expected in zip(result, [[3 / 5, 4 / 5], [5 / 13, 12 / 13], [-8 / 17, 15 / 17]]):
            self.assertIsInstance(actual, list)
            np.testing.assert_array_equal(actual, expected)
        self.assertEqual(source, original)

    def test_zero_embedding_preserves_its_values(self):
        source = [0, -0.0, 0.0]
        result = normalize_embedding_l2(source)
        self.assertIs(result, source)
        self.assertTrue(np.signbit(result[1]))

    def test_zero_rows_do_not_change_other_rows(self):
        source = [[0, 0], [3.0, 4.0], [0.0, -0.0]]
        result = normalize_embedding_l2(source)
        self.assertIs(result[0], source[0])
        self.assertIs(result[2], source[2])
        np.testing.assert_array_equal(result[1], [0.6, 0.8])

    def test_empty_embedding_and_empty_batch_rows(self):
        source = []
        self.assertIs(normalize_embedding_l2(source), source)
        rows = [[], [0.0], []]
        result = normalize_embedding_l2(rows)
        self.assertEqual(result, rows)
        for actual, original in zip(result, rows):
            self.assertIs(actual, original)

    def test_unequal_rows_remain_supported(self):
        source = [[3.0, 4.0], [], [1.0], [1.0, 2.0, 2.0]]
        original = deepcopy(source)
        result = normalize_embedding_l2(source)
        self.assertEqual([len(row) for row in result], [2, 0, 1, 3])
        self.assertEqual(result, [[0.6, 0.8], [], [1.0], [1 / 3, 2 / 3, 2 / 3]])
        self.assertEqual(source, original)

    def test_model_sized_batches_match_rowwise_reference(self):
        rng = np.random.default_rng(2741)
        for dimension in (128, 512, 4096):
            with self.subTest(dimension=dimension):
                source = rng.normal(size=(16, dimension)).tolist()
                original = deepcopy(source)
                expected = [
                    (np.array(row) / float(np.linalg.norm(row))).tolist()
                    for row in source
                ]
                result = normalize_embedding_l2(source)
                np.testing.assert_array_equal(result, expected)
                np.testing.assert_allclose(np.linalg.norm(result, axis=1), np.ones(16))
                self.assertEqual(source, original)
