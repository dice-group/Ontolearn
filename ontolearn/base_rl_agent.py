from abc import ABCMeta, abstractmethod
from .base import KnowledgeBase
from .abstracts import *
from .util import create_experiment_folder


class BaseRLTrainer(metaclass=ABCMeta):

    @abstractmethod
    def __init__(self, *,
                 kb,
                 rho,
                 search_tree,
                 quality_func,
                 reward_func,
                 num_epochs_per_replay,
                 num_episode,
                 epsilon, epsilon_decay,
                 max_len_replay_memory):
        assert isinstance(kb, KnowledgeBase)
        assert isinstance(rho, BaseRefinement)
        assert isinstance(search_tree, AbstractTree)
        assert isinstance(num_epochs_per_replay, int)
        assert isinstance(num_episode, int)
        assert isinstance(reward_func, AbstractScorer)

        self.kb = kb
        self.reward_func = reward_func
        self.rho = rho
        self.search_tree = search_tree
        self.search_tree.quality_func = quality_func
        self.num_episode = num_episode
        self.num_epochs_per_replay = num_epochs_per_replay
        self.epsilon, self.epsilon_decay, self.epsilon_min = epsilon, epsilon_decay, 0

        self.max_len_replay_memory = max_len_replay_memory

        self.start_class = self.kb.thing
        self.concepts_to_nodes = dict()
        self.rho.set_concepts_node_mapping(self.concepts_to_nodes)
        self.storage_path, _ = create_experiment_folder()

    @abstractmethod
    def start(self, *args, **kwargs):
        pass

    @abstractmethod
    def next_node_to_expand(self, *args, **kwargs):
        pass

    @abstractmethod
    def apply_rho(self, *args, **kwargs):
        pass
