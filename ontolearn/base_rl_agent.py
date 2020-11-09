from abc import ABCMeta, abstractmethod
from .base import KnowledgeBase
from .abstracts import *
from .util import create_experiment_folder
from .search import SearchTreePriorityQueue
from .metrics import F1
from .heuristics import Reward
from collections import deque

import torch.optim as optim

"""
import random
import torch
import numpy as np
import functools
from torch.functional import F
from typing import List, Any, Set, AnyStr, Tuple
from collections import namedtuple, deque
from .abstracts import AbstractScorer
from torch.nn.init import xavier_normal_
from itertools import chain
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ExponentialLR
import json
import pandas as pd
from .refinement_operators import LengthBasedRefinement
from .metrics import F1
from .heuristics import Reward
from .base_rl_agent import BaseRLAgent
from concurrent.futures import ThreadPoolExecutor
import time
"""


class BaseRLTrainer(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self,
                 kb,
                 rho,
                 search_tree=SearchTreePriorityQueue(),
                 quality_func=F1(),
                 reward_func=Reward(),
                 num_epochs_per_replay=10,
                 num_episode=759,
                 epsilon=1.0, epsilon_decay=.001,
                 max_len_replay_memory=1024,
                 verbose=0):
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
        self.epsilon, self.epsilon_decay, self.epsilon_min = epsilon, epsilon_decay, 0.0
        self.verbose = verbose
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


class BaseRLAgent(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, reward_func=Reward(),
                 num_epochs_per_replay=10,
                 num_episode=759,
                 epsilon=1.0, epsilon_decay=.001,num_of_sequential_actions=10,
                 max_len_replay_memory=1024):
        # TODO investigate params of agents used in the literature
        self.reward_func = reward_func
        self.num_episode = num_episode
        self.num_epochs_per_replay = num_epochs_per_replay
        self.epsilon, self.epsilon_decay, self.epsilon_min = epsilon, epsilon_decay, 0.0
        self.num_of_sequential_actions = num_of_sequential_actions  # i.e. number of times a concept is refined.
        self.max_len_replay_memory = max_len_replay_memory
        self.seen_examples = dict()

        # DQL related.
        self.replay_modulo_per_episode = 10
        self.experiences = deque(maxlen=self.max_len_replay_memory)

        # self.pos, self.neg = None, None
        # self.emb_pos, self.emb_neg = None, None

        """
        if path_pretrained_agent:
            self.model = Drill()
            self.model.load_state_dict(torch.load(path_pretrained_agent))
        else:
            self.model = Drill()
            self.model.init()
        """
