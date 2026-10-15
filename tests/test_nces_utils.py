import json
import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from ontolearn.nces_utils import SimpleSolution, generate_training_data, sample_examples, try_get_embs

PATH_FAMILY = "KGs/Family/family-benchmark_rich_background.owl"


class TestSampleExamples(unittest.TestCase):

    def setUp(self):
        np.random.seed(0)

    def test_enough_of_both_balanced_split(self):
        pos = [f"p{i}" for i in range(10)]
        neg = [f"n{i}" for i in range(10)]
        p, n = sample_examples(pos, neg, 6)
        self.assertEqual((len(p), len(n)), (3, 3))
        self.assertTrue(set(p) <= set(pos) and set(n) <= set(neg))

    def test_more_positives_than_negatives(self):
        pos = [f"p{i}" for i in range(20)]
        neg = [f"n{i}" for i in range(10)]
        p, n = sample_examples(pos, neg, 10)
        self.assertEqual(len(p) + len(n), 10)
        self.assertEqual(len(n), 5)

    def test_few_negatives_fill_with_positives(self):
        pos = [f"p{i}" for i in range(20)]
        neg = ["n0", "n1"]
        p, n = sample_examples(pos, neg, 10)
        self.assertEqual(sorted(n), ["n0", "n1"])
        self.assertEqual(len(p), 8)

    def test_few_positives_fill_with_negatives(self):
        pos = ["p0", "p1"]
        neg = [f"n{i}" for i in range(20)]
        p, n = sample_examples(pos, neg, 10)
        self.assertEqual(sorted(p), ["p0", "p1"])
        self.assertEqual(len(n), 8)

    def test_not_enough_examples_returns_all(self):
        p, n = sample_examples(["p0", "p1"], ["n0"], 10)
        self.assertEqual((sorted(p), n), (["p0", "p1"], ["n0"]))

    def test_no_duplicates(self):
        pos = [f"p{i}" for i in range(10)]
        neg = [f"n{i}" for i in range(10)]
        p, n = sample_examples(pos, neg, 8)
        self.assertEqual(len(set(p)), len(p))
        self.assertEqual(len(set(n)), len(n))


class TestTryGetEmbs(unittest.TestCase):

    def setUp(self):
        np.random.seed(0)
        self.embeddings = pd.DataFrame(np.random.rand(6, 3), index=[f"e{i}" for i in range(6)])

    def test_all_present_unchanged(self):
        pos, neg = try_get_embs(["e0", "e1"], ["e2", "e3"], self.embeddings, 4)
        self.assertEqual((pos, neg), (["e0", "e1"], ["e2", "e3"]))

    def test_negatives_padded_when_too_few(self):
        pos, neg = try_get_embs(["e0", "e1"], ["e2", "e3"], self.embeddings, 5)
        self.assertEqual(len(pos) + len(neg), 5)
        self.assertEqual(neg, ["e2", "e3", "e2"])

    def test_negatives_truncated_when_too_many(self):
        pos, neg = try_get_embs(["e0"], ["e1", "e2", "e3", "e4"], self.embeddings, 3)
        self.assertEqual(len(pos) + len(neg), 3)

    def test_missing_positive_is_filtered_and_padded(self):
        pos, neg = try_get_embs(["e0", "e1", "missing"], ["e2", "e3"], self.embeddings, 5)
        self.assertEqual(len(pos), 3)
        self.assertTrue(all(p in self.embeddings.index for p in pos))
        self.assertTrue(all(n in self.embeddings.index for n in neg))

    def test_missing_negative_is_filtered_and_padded(self):
        pos, neg = try_get_embs(["e0", "e1"], ["e2", "e3", "missing"], self.embeddings, 5)
        self.assertEqual(len(neg), 3)
        self.assertTrue(all(n in self.embeddings.index for n in neg))

    def test_all_missing_falls_back_to_random_known_embeddings(self):
        pos, neg = try_get_embs(["x0", "x1"], ["y0", "y1"], self.embeddings, 4)
        self.assertTrue(all(p in self.embeddings.index for p in pos))
        self.assertTrue(all(n in self.embeddings.index for n in neg))


class TestSimpleSolution(unittest.TestCase):

    def test_predict_returns_known_atomic_classes(self):
        atoms = ["Male", "Female", "Person"]
        model = SimpleSolution(vocab=["Male Female Person ⊓ ⊔ ∃ hasChild"], atomic_concept_names=list(atoms))
        pred = model.predict("Male")
        self.assertEqual(model.name, "SimpleSolution")
        self.assertIn("Male", pred.split(" ⊔ "))
        self.assertTrue(set(pred.split(" ⊔ ")) <= set(atoms))

    def test_predict_falls_back_to_random_atoms_when_none_found(self):
        atoms = ["A", "B", "C", "D"]
        model = SimpleSolution(vocab=["A B C D ⊓"], atomic_concept_names=list(atoms))
        pred = model.predict("⊓ ⊓")
        parts = pred.split(" ⊔ ")
        self.assertEqual(len(parts), 3)
        self.assertTrue(set(parts) <= set(atoms))


class TestGenerateTrainingData(unittest.TestCase):

    def test_generate_and_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            storage = os.path.join(tmp, "data")
            lps = generate_training_data(PATH_FAMILY, None, max_num_lps=5, refs_sample_size=5,
                                         storage_path=storage)
            self.assertGreater(len(lps), 0)
            self.assertTrue(os.path.exists(os.path.join(storage, "LPs.json")))
            with open(os.path.join(storage, "LPs.json")) as f:
                self.assertIsInstance(json.load(f), (dict, list))


if __name__ == "__main__":
    unittest.main()
