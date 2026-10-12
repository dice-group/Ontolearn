import random
import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from ontolearn.data_struct import (CLIPDataset, CLIPDatasetInference, Experience, NCESBaseDataset, NCESDataset,
                                   NCESDatasetInference, PrepareBatchOfPrediction, PrepareBatchOfTraining,
                                   ROCESDataset, ROCESDatasetInference, TriplesData, TriplesDataset)

DIM = 4


def make_embeddings(n=12):
    gen = np.random.RandomState(0)
    return pd.DataFrame(gen.rand(n, DIM), index=[f"e{i}" for i in range(n)])


def make_data(n_lps=3):
    """[(concept_name, {'positive examples': [...], 'negative examples': [...], 'length': int})]"""
    return [(f"Male ⊓ ∃ hasChild.Person{i}", {"positive examples": [f"e{j}" for j in range(0, 6)],
                                             "negative examples": [f"e{j}" for j in range(6, 12)],
                                             "length": 5 + i}) for i in range(n_lps)]


def make_vocab(data):
    tokens = sorted({t for k, _ in data for t in NCESBaseDataset.decompose(k)})
    vocab = {t: i for i, t in enumerate(tokens)}
    vocab["PAD"] = len(vocab)
    return vocab, {v: k for k, v in vocab.items()}


class TestDecompose(unittest.TestCase):

    def test_splits_on_specials_and_keeps_them(self):
        self.assertEqual(NCESBaseDataset.decompose("A ⊓ B"), ["A", " ", "⊓", " ", "B"])
        self.assertEqual(NCESBaseDataset.decompose("∃ r.A"), ["∃", " ", "r", ".", "A"])
        self.assertEqual(NCESBaseDataset.decompose("¬(A ⊔ B)"), ["¬", "(", "A", " ", "⊔", " ", "B", ")"])

    def test_dot_inside_number_is_kept(self):
        self.assertEqual(NCESBaseDataset.decompose("≥ 2 r.A"), ["≥", " ", "2", " ", "r", ".", "A"])
        self.assertEqual(NCESBaseDataset.decompose("x1.5"), ["x1.5"])

    def test_empty_and_single_token(self):
        self.assertEqual(NCESBaseDataset.decompose(""), [])
        self.assertEqual(NCESBaseDataset.decompose("Person"), ["Person"])
        self.assertEqual(NCESBaseDataset.decompose("⊤"), ["⊤"])

    def test_get_labels(self):
        data = make_data(1)
        vocab, inv_vocab = make_vocab(data)
        ds = NCESBaseDataset(vocab, inv_vocab, max_length=20)
        labels, length = ds.get_labels(data[0][0])
        self.assertEqual(length, len(labels))
        self.assertEqual([inv_vocab[label] for label in labels], NCESBaseDataset.decompose(data[0][0]))


class TestNCESDatasets(unittest.TestCase):

    def setUp(self):
        random.seed(0)
        np.random.seed(0)
        self.data = make_data()
        self.embeddings = make_embeddings()
        self.vocab, self.inv_vocab = make_vocab(self.data)

    def test_training_dataset_item(self):
        ds = NCESDataset(self.data, self.embeddings, num_examples=8, vocab=self.vocab, inv_vocab=self.inv_vocab,
                         shuffle_examples=False, max_length=30)
        self.assertEqual(len(ds), 3)
        pos, neg, labels = ds[0]
        self.assertEqual(pos.shape[-1], DIM)
        self.assertEqual(neg.shape[-1], DIM)
        self.assertEqual(labels.shape, (30,))
        self.assertEqual(labels.dtype, torch.int64)
        self.assertEqual(labels[-1].item(), self.vocab["PAD"])

    def test_training_dataset_example_sizes(self):
        ds = NCESDataset(self.data, self.embeddings, num_examples=12, vocab=self.vocab, inv_vocab=self.inv_vocab,
                         shuffle_examples=True, max_length=30, example_sizes=[(2, 3)])
        pos, neg, _ = ds[1]
        self.assertEqual(pos.shape, (2, DIM))
        self.assertEqual(neg.shape, (3, DIM))

    def test_inference_dataset_sorted_and_shuffled(self):
        data = [(name, v["positive examples"], v["negative examples"]) for name, v in self.data]
        for sorted_examples, shuffle in ((True, False), (False, True), (False, False)):
            ds = NCESDatasetInference(data, self.embeddings, num_examples=8, vocab=self.vocab,
                                      inv_vocab=self.inv_vocab, shuffle_examples=shuffle, max_length=30,
                                      sorted_examples=sorted_examples)
            self.assertEqual(len(ds), 3)
            pos, neg = ds[0]
            self.assertEqual(pos.shape[-1], DIM)
            self.assertEqual(neg.shape[-1], DIM)

    def test_inference_dataset_tolerates_unknown_individuals(self):
        data = [("c", ["e0", "e1", "unknown"], ["e6", "e7"])]
        ds = NCESDatasetInference(data, self.embeddings, num_examples=5, vocab=self.vocab,
                                  inv_vocab=self.inv_vocab, shuffle_examples=False, max_length=30)
        pos, neg = ds[0]
        self.assertEqual(pos.shape[-1], DIM)
        self.assertEqual(neg.shape[-1], DIM)


class TestCLIPDatasets(unittest.TestCase):

    def setUp(self):
        random.seed(0)
        np.random.seed(0)
        self.data = make_data()
        self.embeddings = make_embeddings()

    def test_example_sizes(self):
        ds = CLIPDataset(self.data, self.embeddings, num_examples=8, shuffle_examples=True,
                         example_sizes=[(2, 2)], k=None)
        self.assertEqual(len(ds), 3)
        pos, neg, length = ds[0]
        self.assertEqual(pos.shape, (2, DIM))
        self.assertEqual(neg.shape, (2, DIM))
        self.assertEqual(length.item(), self.data[0][1]["length"])

    def test_k_sampling(self):
        ds = CLIPDataset(self.data, self.embeddings, num_examples=8, shuffle_examples=False, k=2)
        pos, neg, _ = ds[0]
        self.assertEqual(pos.shape[-1], DIM)
        self.assertEqual(neg.shape[-1], DIM)
        self.assertEqual(pos.shape[0] % 2, 0)

    def test_all_examples_when_no_sampling(self):
        ds = CLIPDataset(self.data, self.embeddings, num_examples=12, shuffle_examples=False, k=None)
        pos, neg, _ = ds[2]
        self.assertEqual(pos.shape, (6, DIM))
        self.assertEqual(neg.shape, (6, DIM))

    def test_inference(self):
        data = [(name, v["positive examples"], v["negative examples"]) for name, v in self.data]
        for sorted_examples in (True, False):
            ds = CLIPDatasetInference(data, self.embeddings, num_examples=12, shuffle_examples=True,
                                      sorted_examples=sorted_examples)
            self.assertEqual(len(ds), 3)
            pos, neg = ds[1]
            self.assertEqual(pos.shape[-1], DIM)
            self.assertEqual(neg.shape[-1], DIM)


class FakeEmbeddingModel:
    def __init__(self, n, dim):
        self.n, self.dim = n, dim

    def get_embeddings(self):
        return torch.arange(self.n * self.dim, dtype=torch.float32).view(self.n, self.dim), None


class TestROCESDatasets(unittest.TestCase):

    def setUp(self):
        random.seed(0)
        np.random.seed(0)
        self.data = make_data()
        names = [f"e{i}" for i in range(12)]
        self.triples = SimpleNamespace(entity2idx=pd.DataFrame(list(range(12)), index=names))
        self.vocab, self.inv_vocab = make_vocab(self.data)
        self.model = FakeEmbeddingModel(12, DIM)

    def test_training_sampling_strategies(self):
        for strategy in ("p", "nces2", "other"):
            ds = ROCESDataset(self.data, self.triples, num_examples=8, k=2, vocab=self.vocab,
                              inv_vocab=self.inv_vocab, max_length=30, sampling_strategy=strategy)
            ds.load_embeddings(self.model)
            self.assertEqual(len(ds), 3)
            pos, neg, labels = ds[0]
            self.assertEqual(pos.shape[-1], DIM, strategy)
            self.assertEqual(neg.shape[-1], DIM, strategy)
            self.assertEqual(labels.shape, (30,))

    def test_set_k(self):
        ds = ROCESDataset(self.data, self.triples, 8, 2, self.vocab, self.inv_vocab, 30)
        ds.set_k(3)
        self.assertEqual(ds.k, 3)
        inf = ROCESDatasetInference([], self.triples, 8, 2, self.vocab, self.inv_vocab, 30)
        inf.set_k(4)
        self.assertEqual(inf.k, 4)

    def test_inference_sampling_strategies(self):
        data = [(name, v["positive examples"], v["negative examples"]) for name, v in self.data]
        for strategy in ("p", "nces2", "other"):
            ds = ROCESDatasetInference(data, self.triples, num_examples=8, k=2, vocab=self.vocab,
                                       inv_vocab=self.inv_vocab, max_length=30, sampling_strategy=strategy,
                                       num_pred_per_lp=2)
            ds.load_embeddings(self.model)
            self.assertEqual(len(ds), 3)
            pos, neg = ds[0]
            self.assertEqual(pos.shape[0], 2, strategy)
            self.assertEqual(pos.shape[-1], DIM, strategy)
            self.assertEqual(neg.shape[0], 2, strategy)
            self.assertEqual(neg.shape[-1], DIM, strategy)


class TestTriples(unittest.TestCase):

    def test_triples_data_from_owl(self):
        td = TriplesData("KGs/Family/father.owl")
        self.assertGreater(len(td.triples), 0)
        self.assertTrue(all(len(t) == 3 for t in td.triples))
        self.assertEqual(td.entities, sorted(set(td.entities)))
        self.assertEqual(td.relations, sorted(set(td.relations)))
        self.assertEqual(len(td.entity2idx), len(td.entities))
        self.assertEqual(len(td.relation2idx), len(td.relations))
        self.assertEqual(td.entity2idx.loc[td.entities[0]].item(), 0)

    def test_static_helpers(self):
        data = [("a", "r", "b"), ("c", "s", "a")]
        self.assertEqual(TriplesData.get_entities(data), ["a", "b", "c"])
        self.assertEqual(TriplesData.get_relations(data), ["r", "s"])

    def test_triples_dataset(self):
        ds = TriplesDataset({(0, 1): [0, 2], (1, 0): [1]}, num_e=3)
        self.assertEqual(len(ds), 2)
        h, r, y = ds[0]
        self.assertEqual((h.item(), r.item()), (0, 1))
        self.assertEqual(y.tolist(), [1.0, 0.0, 1.0])
        h, r, y = ds[1]
        self.assertEqual((h.item(), r.item()), (1, 0))
        self.assertEqual(y.tolist(), [0.0, 1.0, 0.0])


class TestRLStructures(unittest.TestCase):

    def test_experience(self):
        exp = Experience(maxlen=2)
        state = lambda: SimpleNamespace(embeddings=torch.zeros(1, DIM))  # noqa: E731
        for r in (0.1, 0.2, 0.3):
            exp.append((state(), state(), r))
        self.assertEqual(len(exp), 2)  # bounded by maxlen
        cur, nxt, rewards = exp.retrieve()
        self.assertEqual((len(cur), len(nxt), rewards), (2, 2, [0.2, 0.3]))
        exp.clear()
        self.assertEqual(len(exp), 0)

    def test_experience_rejects_mismatched_shapes(self):
        exp = Experience(maxlen=2)
        with self.assertRaises(AssertionError):
            exp.append((SimpleNamespace(embeddings=torch.zeros(1, DIM)),
                        SimpleNamespace(embeddings=torch.zeros(1, DIM + 1)), 0.0))

    def test_prepare_batch_of_training(self):
        n = 5
        s, s_prime = torch.rand(n, 1, DIM), torch.rand(n, 1, DIM)
        p, neg = torch.rand(1, 1, DIM), torch.rand(1, 1, DIM)
        q = torch.rand(n)
        ds = PrepareBatchOfTraining(s, s_prime, p, neg, q)
        self.assertEqual(len(ds), n)
        x, y = ds[0]
        self.assertEqual(x.shape, (4, DIM))
        self.assertEqual(y.shape, (1,))

    def test_prepare_batch_of_training_rejects_invalid_values(self):
        n = 3
        good = lambda: torch.rand(n, 1, DIM)  # noqa: E731
        p, neg, q = torch.rand(1, 1, DIM), torch.rand(1, 1, DIM), torch.rand(n)
        bad = good()
        bad[0, 0, 0] = float("nan")
        with self.assertRaises(ValueError):
            PrepareBatchOfTraining(bad, good(), p, neg, q)
        with self.assertRaises(ValueError):
            PrepareBatchOfTraining(good(), bad, p, neg, q)
        with self.assertRaises(ValueError):
            PrepareBatchOfTraining(good(), good(), torch.full((1, 1, DIM), float("inf")), neg, q)
        with self.assertRaises(ValueError):
            PrepareBatchOfTraining(good(), good(), p, torch.full((1, 1, DIM), float("nan")), q)
        with self.assertRaises(ValueError):
            PrepareBatchOfTraining(good(), good(), p, neg, torch.full((n,), float("nan")))

    def test_prepare_batch_of_prediction(self):
        num_next = 4
        current = torch.rand(1, DIM)
        nxt = torch.rand(num_next, 1, DIM)
        p, n = torch.rand(1, DIM), torch.rand(1, DIM)
        ds = PrepareBatchOfPrediction(current, nxt, p, n)
        self.assertEqual(len(ds), num_next)
        self.assertEqual(ds[0].shape, (4, DIM))
        self.assertEqual(ds.get_all().shape, (num_next, 4, DIM))


if __name__ == "__main__":
    unittest.main()
