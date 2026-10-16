import unittest

import numpy as np
import pandas as pd

from ontolearn.data_struct import CLIPDatasetInference


class TestCLIPDatasetInference(unittest.TestCase):

    def setUp(self):
        gen = np.random.RandomState(0)
        self.embeddings = pd.DataFrame(gen.rand(6, 3), index=[f"e{i}" for i in range(6)])

    def test_negative_embeddings_come_from_negative_examples(self):
        # Regression test for #638: the negative tensor was built from the positive examples.
        pos, neg = ["e0", "e1", "e2"], ["e3", "e4", "e5"]
        ds = CLIPDatasetInference([("", pos, neg)], self.embeddings, num_examples=6, shuffle_examples=False,
                                  sorted_examples=True)
        x_pos, x_neg = ds[0]
        self.assertEqual(tuple(x_pos.shape), (3, 3))
        self.assertEqual(tuple(x_neg.shape), (3, 3))
        np.testing.assert_allclose(x_pos.numpy(), self.embeddings.loc[pos].values, rtol=1e-6)
        np.testing.assert_allclose(x_neg.numpy(), self.embeddings.loc[neg].values, rtol=1e-6)
        self.assertFalse(np.allclose(x_pos.numpy(), x_neg.numpy()))


if __name__ == "__main__":
    unittest.main()
