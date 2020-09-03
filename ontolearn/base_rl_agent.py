from abc import ABCMeta, abstractmethod
from .base import KnowledgeBase
from .abstracts import *


class BaseRLTrainer(metaclass=ABCMeta):

    @abstractmethod
    def __init__(self, *,
                 kb,
                 rho,
                 search_tree,
                 reward_func,
                 train_data, iter_bound=1000):
        assert isinstance(kb, KnowledgeBase)
        assert isinstance(rho, BaseRefinement)
        assert isinstance(search_tree, AbstractTree)
        assert isinstance(iter_bound, int)
        assert isinstance(reward_func, AbstractScorer)

        self.kb = kb
        self.reward_func = reward_func
        self.rho = rho
        self.search_tree = search_tree
        self.train_data = train_data

        self.start_class = self.kb.thing
        self.concepts_to_nodes = dict()
        self.rho.set_concepts_node_mapping(self.concepts_to_nodes)

        if train_data:
            assert isinstance(train_data, Dict)
            example_types = {'positive_examples', 'negative_examples'}
            for k, v in train_data.items():
                assert isinstance(k, str)
                try:
                    assert example_types.issubset(set(v.keys()))
                except AssertionError:
                    print(example_types, 'not found in v.keys()')
                    exit(1)

    @abstractmethod
    def start(self, *args, **kwargs):
        pass

    @abstractmethod
    def next_node_to_expand(self, *args, **kwargs):
        pass

    @abstractmethod
    def apply_rho(self, *args, **kwargs):
        pass
