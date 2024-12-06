# -----------------------------------------------------------------------------
# MIT License
#
# Copyright (c) 2024 Ontolearn Team
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# -----------------------------------------------------------------------------

"""The base class of NCES."""

from ontolearn.knowledge_base import KnowledgeBase
from owlapy.render import DLSyntaxObjectRenderer
import numpy as np
from torch.functional import F
from torch.nn.utils.rnn import pad_sequence
from abc import abstractmethod


class BaseNCES:

    def __init__(self, knowledge_base_path, quality_func, num_predictions, proj_dim=128, drop_prob=0.1,
                 num_heads=4, num_seeds=1, m=32, ln=False, learning_rate=1e-4, decay_rate=0.0, clip_value=5.0,
                 batch_size=256, num_workers=4, max_length=48, load_pretrained=True, sorted_examples=False, verbose: int = 0):
        kb = KnowledgeBase(path=knowledge_base_path)
        self.kb_namespace = list(kb.ontology.classes_in_signature())[0].iri.get_namespace()
        self.renderer = DLSyntaxObjectRenderer()
        atomic_concepts = list(kb.ontology.classes_in_signature())
        atomic_concept_names = [self.renderer.render(a) for a in atomic_concepts]
        self.atomic_concept_names = atomic_concept_names
        role_names = [rel.iri.get_remainder() for rel in kb.ontology.object_properties_in_signature()]
        vocab = atomic_concept_names + role_names + ['⊔', '⊓', '∃', '∀', '¬', '⊤', '⊥', '.', ' ', '(', ')']
        vocab = sorted(vocab) + ['PAD']
        self.knowledge_base_path = knowledge_base_path
        self.kb = kb
        self.all_individuals = set([ind.str.split("/")[-1] for ind in kb.individuals()])
        self.inv_vocab = np.array(vocab, dtype='object')
        self.vocab = {vocab[i]: i for i in range(len(vocab))}
        self.num_predictions = num_predictions
        self.proj_dim = proj_dim
        self.drop_prob = drop_prob
        self.num_heads = num_heads
        self.num_seeds = num_seeds
        self.m = m
        self.ln = ln
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.clip_value = clip_value
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_length = max_length
        self.load_pretrained = load_pretrained
        self.sorted_examples = sorted_examples
        self.verbose = verbose
        self.num_examples = self.find_optimal_number_of_examples(kb)
        self.best_predictions = None
        

    @staticmethod
    def find_optimal_number_of_examples(kb):
        if kb.individuals_count() >= 600:
            return min(kb.individuals_count()//2, 1000)
        return kb.individuals_count()

    def collate_batch(self, batch):  # pragma: no cover
        pos_emb_list = []
        neg_emb_list = []
        target_labels = []
        for pos_emb, neg_emb, label in batch:
            if pos_emb.ndim != 2:
                pos_emb = pos_emb.reshape(1, -1)
            if neg_emb.ndim != 2:
                neg_emb = neg_emb.reshape(1, -1)
            pos_emb_list.append(pos_emb)
            neg_emb_list.append(neg_emb)
            target_labels.append(label)
        pos_emb_list[0] = F.pad(pos_emb_list[0], (0, 0, 0, self.num_examples - pos_emb_list[0].shape[0]), "constant", 0)
        pos_emb_list = pad_sequence(pos_emb_list, batch_first=True, padding_value=0)
        neg_emb_list[0] = F.pad(neg_emb_list[0], (0, 0, 0, self.num_examples - neg_emb_list[0].shape[0]), "constant", 0)
        neg_emb_list = pad_sequence(neg_emb_list, batch_first=True, padding_value=0)
        target_labels = pad_sequence(target_labels, batch_first=True, padding_value=-100)
        return pos_emb_list, neg_emb_list, target_labels

    def collate_batch_inference(self, batch):  # pragma: no cover
        pos_emb_list = []
        neg_emb_list = []
        for pos_emb, neg_emb in batch:
            if pos_emb.ndim != 2:
                pos_emb = pos_emb.reshape(1, -1)
            if neg_emb.ndim != 2:
                neg_emb = neg_emb.reshape(1, -1)
            pos_emb_list.append(pos_emb)
            neg_emb_list.append(neg_emb)
        pos_emb_list[0] = F.pad(pos_emb_list[0], (0, 0, 0, self.num_examples - pos_emb_list[0].shape[0]), "constant", 0)
        pos_emb_list = pad_sequence(pos_emb_list, batch_first=True, padding_value=0)
        neg_emb_list[0] = F.pad(neg_emb_list[0], (0, 0, 0, self.num_examples - neg_emb_list[0].shape[0]), "constant", 0)
        neg_emb_list = pad_sequence(neg_emb_list, batch_first=True, padding_value=0)
        return pos_emb_list, neg_emb_list

    @abstractmethod
    def get_synthesizer(self):
        pass

    @abstractmethod
    def load_pretrained(self):
        pass
