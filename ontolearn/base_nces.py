"""The base class of NCES."""

from ontolearn.knowledge_base import KnowledgeBase
from owlapy.render import DLSyntaxObjectRenderer
import numpy as np
from torch.functional import F
from torch.nn.utils.rnn import pad_sequence
from .utils import read_csv
from abc import abstractmethod


class BaseNCES:

    def __init__(self, knowledge_base_path, learner_name, path_of_embeddings, batch_size=256, learning_rate=1e-4,
                 decay_rate=0.0, clip_value=5.0, num_workers=8):
        self.name = "NCES"
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
        self.learner_name = learner_name
        self.num_examples = self.find_optimal_number_of_examples(kb)
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.clip_value = clip_value
        self.num_workers = num_workers
        self.instance_embeddings = read_csv(path_of_embeddings)
        self.input_size = self.instance_embeddings.shape[1]

    @staticmethod
    def find_optimal_number_of_examples(kb):
        if kb.individuals_count() >= 600:
            return min(kb.individuals_count()//2, 1000)
        return kb.individuals_count()

    def collate_batch(self, batch):
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

    def collate_batch_inference(self, batch):
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
