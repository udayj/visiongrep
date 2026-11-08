"""Checks for the COCO retrieval statistics without model or corpus files."""

import math
import unittest

import numpy as np

from benchmarks.model_contract.evaluate_int8 import coco_retrieval


class CocoRetrievalTests(unittest.TestCase):
    def test_recall_overlap_and_score_differences_across_chunks(self):
        # The unmodified vectors retrieve the correct image first. Swapping images
        # 0 and 11 moves image 0 behind image 11 and changes one top-ten member.
        # Zero-score ties exercise the filename-order tie break.
        images = np.eye(12, dtype=np.float32)
        quantized = images.copy()
        quantized[[0, 11]] = quantized[[11, 0]]
        captions = np.tile(images[:3], (100, 1))
        correct = np.tile(np.arange(3), 100)

        recalls, overlap, mean, stddev = coco_retrieval(captions, images, quantized, correct)

        np.testing.assert_allclose(recalls, [[1, 1, 1], [2 / 3, 1, 1]])
        self.assertAlmostEqual(overlap, 29 / 30)
        self.assertAlmostEqual(mean, 1 / 3)
        self.assertAlmostEqual(stddev, math.sqrt(2) / 3)


if __name__ == "__main__":
    unittest.main()
