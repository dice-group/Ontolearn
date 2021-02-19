from abc import ABCMeta
from .concept_learner import BaseConceptLearner
from .abstracts import AbstractDrill, AbstractScorer
from .util import *
from .search import Node, SearchTreePriorityQueue
from .data_struct import PrepareBatchOfTraining, PrepareBatchOfPrediction, Experience
from .refinement_operators import LengthBasedRefinement
from .metrics import F1
from .heuristics import Reward
import time
import json
import pandas as pd
import random
import torch
from torch import nn
import numpy as np
import functools
from torch.functional import F
from typing import List, Any, Set, Tuple
from collections import namedtuple, deque
from torch.nn.init import xavier_normal_
from itertools import chain
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ExponentialLR
import random

random.seed(1)


class DrillAverage(AbstractDrill, BaseConceptLearner):
    """
    Convolutional DQL concept learning agent based on input averaging
    """

    def __init__(self, knowledge_base,
                 path_of_embeddings=None,
                 refinement_operator=None, quality_func=F1(),
                 pretrained_model_path=None,
                 iter_bound=None, max_num_of_concepts_tested=None, verbose=None,
                 terminate_on_goal=True, ignored_concepts=None,
                 max_len_replay_memory=None, batch_size=None, epsilon_decay=None, epsilon_min=None,
                 num_epochs_per_replay=None, learning_rate=None,
                 max_runtime=5, num_of_sequential_actions=None, num_episode=None, num_workers=32):

        if refinement_operator is None:
            refinement_operator = LengthBasedRefinement(kb=knowledge_base)
        AbstractDrill.__init__(self,
                               path_of_embeddings=path_of_embeddings,
                               reward_func=Reward(),
                               max_len_replay_memory=max_len_replay_memory,
                               batch_size=batch_size, epsilon_min=epsilon_min,
                               num_epochs_per_replay=num_epochs_per_replay,
                               representation_mode='averaging',
                               epsilon_decay=epsilon_decay,
                               num_of_sequential_actions=num_of_sequential_actions, num_episode=num_episode,
                               learning_rate=learning_rate,
                               num_workers=num_workers)

        self.sample_size = 1
        arg_net = {'input_shape': (4 * self.sample_size, self.embedding_dim),
                   'first_out_channels': 32, 'second_out_channels': 16, 'third_out_channels': 8,
                   'kernel_size': 3}

        self.heuristic_func = DrillHeuristic(mode='averaging', model_args=arg_net)
        self.optimizer = torch.optim.Adam(self.heuristic_func.net.parameters(), lr=self.learning_rate)

        if pretrained_model_path:
            m = torch.load(pretrained_model_path, torch.device('cpu'))
            self.heuristic_func.net.load_state_dict(m)

        BaseConceptLearner.__init__(self, knowledge_base=knowledge_base,
                                    refinement_operator=refinement_operator,
                                    search_tree=SearchTreePriorityQueue(),
                                    quality_func=quality_func,
                                    heuristic_func=self.heuristic_func,
                                    ignored_concepts=ignored_concepts,
                                    terminate_on_goal=terminate_on_goal,
                                    iter_bound=iter_bound,
                                    max_num_of_concepts_tested=max_num_of_concepts_tested,
                                    max_runtime=max_runtime,
                                    verbose=verbose, name='DrillAverage')

    def represent_examples(self, *, pos: Set[str], neg: Set[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Represent E+ and E- by using embeddings of individuals.
        Here, we take the average of embeddings of individuals.
        @param pos:
        @param neg:
        @return:
        """
        assert isinstance(pos, set) and isinstance(neg, set)

        emb_pos = torch.tensor(self.instance_embeddings.loc[list(pos)].values,
                               dtype=torch.float32)
        emb_neg = torch.tensor(self.instance_embeddings.loc[list(neg)].values,
                               dtype=torch.float32)
        assert emb_pos.shape[0] == len(pos)
        assert emb_neg.shape[0] == len(neg)

        # Take the mean of embeddings.
        emb_pos = torch.mean(emb_pos, dim=0)
        emb_pos = emb_pos.view(1, 1, emb_pos.shape[0])
        emb_neg = torch.mean(emb_neg, dim=0)
        emb_neg = emb_neg.view(1, 1, emb_neg.shape[0])
        return emb_pos, emb_neg

    def init_training(self, pos_uri: Set[str], neg_uri: Set[str]) -> None:
        """

        @param pos_uri: A set of positive examples where each example corresponds to a string representation of an individual/instance.
        @param neg_uri: A set of negative examples where each example corresponds to a string representation of an individual/instance.
        @return:
        """
        # (1) String to Owlready2 conversion of examples.
        self.reward_func.pos = set(self.kb.convert_uri_instance_to_obj_from_iterable(pos_uri))
        self.reward_func.neg = set(self.kb.convert_uri_instance_to_obj_from_iterable(neg_uri))

        # (3) Obtain embeddings of positive and negative examples.
        self.emb_pos = torch.tensor(self.instance_embeddings.loc[list(pos_uri)].values, dtype=torch.float32)
        self.emb_neg = torch.tensor(self.instance_embeddings.loc[list(neg_uri)].values, dtype=torch.float32)

        # (3) Take the mean of positive and negative examples and reshape it into (1,1,embedding_dim) for mini batching.
        self.emb_pos = torch.mean(self.emb_pos, dim=0)
        self.emb_pos = self.emb_pos.view(1, 1, self.emb_pos.shape[0])
        self.emb_neg = torch.mean(self.emb_neg, dim=0)
        self.emb_neg = self.emb_neg.view(1, 1, self.emb_neg.shape[0])
        # Sanity checking
        if torch.isnan(self.emb_pos).any() or torch.isinf(self.emb_pos).any():
            print(string_balanced_pos)
            raise ValueError('invalid value detected in E+,\n{0}'.format(self.emb_pos))
        if torch.isnan(self.emb_neg).any() or torch.isinf(self.emb_neg).any():
            raise ValueError('invalid value detected in E-,\n{0}'.format(self.emb_neg))

        # Default exploration exploitation tradeoff.
        self.epsilon = 1

    def terminate_training(self):
        self.save_weights()
        # json serialize
        with open(self.storage_path + '/seen_lp.json', 'w') as file_descriptor:
            json.dump(self.seen_examples, file_descriptor, indent=3)
        self.kb.clean()
        return self

    def fit(self, pos: Set[str], neg: Set[str], ignore: Set[str] = None):
        """
        Find hypotheses that explain pos and neg.
        """
        self.default_state_rl()
        self.initialize_learning_problem(pos=pos, neg=neg, all_instances=self.kb.thing.instances, ignore=ignore)
        self.emb_pos, self.emb_neg = self.represent_examples(pos=pos, neg=neg)
        self.start_time = time.time()
        for i in range(1, self.iter_bound):
            most_promising = self.next_node_to_expand(i)
            refinements = [ref for ref in self.apply_rho(most_promising)]
            if len(refinements) == 0:
                most_promising.heuristic = -1
                self.search_tree.add(most_promising)
                continue
            predicted_Q_values = self.predict_Q(current_state=most_promising, next_states=refinements)
            self.goal_found = self.update_search(refinements, predicted_Q_values)
            if self.goal_found:
                if self.terminate_on_goal:
                    return self.terminate()
            if time.time() - self.start_time > self.max_runtime:
                return self.terminate()
            if self.number_of_tested_concepts >= self.max_num_of_concepts_tested:
                return self.terminate()
        return self.terminate()


class DrillSample(AbstractDrill, BaseConceptLearner):
    """
    Convolutional DQL concept learning agent based on input sampling
    """

    def __init__(self, knowledge_base,
                 path_of_embeddings=None,
                 refinement_operator=None, quality_func=F1(),
                 pretrained_model_path=None,
                 iter_bound=None, max_num_of_concepts_tested=None, verbose=None,
                 terminate_on_goal=True, ignored_concepts=None,
                 max_len_replay_memory=None, batch_size=None, epsilon_decay=None, epsilon_min=None,
                 num_epochs_per_replay=None, learning_rate=None,
                 max_runtime=5, num_of_sequential_actions=None, num_episode=None, num_workers=32):

        if refinement_operator is None:
            refinement_operator = LengthBasedRefinement(kb=knowledge_base)
        AbstractDrill.__init__(self,
                               path_of_embeddings=path_of_embeddings,
                               reward_func=Reward(),
                               max_len_replay_memory=max_len_replay_memory,
                               batch_size=batch_size, epsilon_min=epsilon_min,
                               num_epochs_per_replay=num_epochs_per_replay,
                               representation_mode='sampling',
                               epsilon_decay=epsilon_decay,
                               num_of_sequential_actions=num_of_sequential_actions, num_episode=num_episode,
                               learning_rate=learning_rate,
                               num_workers=num_workers)
        self.sample_size = 1
        arg_net = {'input_shape': (4 * self.sample_size, self.embedding_dim),
                   'first_out_channels': 32, 'second_out_channels': 16, 'third_out_channels': 8,
                   'kernel_size': 3}

        self.heuristic_func = DrillHeuristic(mode='sampling', model_args=arg_net)
        self.optimizer = torch.optim.Adam(self.heuristic_func.net.parameters(), lr=self.learning_rate)

        if pretrained_model_path:
            m = torch.load(pretrained_model_path, torch.device('cpu'))
            self.heuristic_func.net.load_state_dict(m)

        BaseConceptLearner.__init__(self, knowledge_base=knowledge_base,
                                    refinement_operator=refinement_operator,
                                    search_tree=SearchTreePriorityQueue(),
                                    quality_func=quality_func,
                                    heuristic_func=self.heuristic_func,
                                    ignored_concepts=ignored_concepts,
                                    terminate_on_goal=terminate_on_goal,
                                    iter_bound=iter_bound,
                                    max_num_of_concepts_tested=max_num_of_concepts_tested,
                                    max_runtime=max_runtime,
                                    verbose=verbose, name='DrillSample')

    def represent_examples(self, *, pos, neg) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        @param pos:
        @param neg:
        @return:
        """
        assert isinstance(pos, set) and isinstance(neg, set)
        try:
            assert len(pos) >= self.sample_size and len(neg) >= self.sample_size
        except AssertionError:
            print(len(pos))
            print(len(neg))
            exit(1)
        sampled_pos = random.sample(pos, self.sample_size)
        sampled_neg = random.sample(neg, self.sample_size)

        emb_pos = torch.tensor(self.instance_embeddings.loc[sampled_pos].values,
                               dtype=torch.float32)
        emb_neg = torch.tensor(self.instance_embeddings.loc[list(sampled_neg)].values,
                               dtype=torch.float32)
        emb_pos = emb_pos.view(1, self.sample_size, self.instance_embeddings.shape[1])
        emb_neg = emb_neg.view(1, self.sample_size, self.instance_embeddings.shape[1])
        return emb_pos, emb_neg

    def init_training(self, pos_uri: Set[str], neg_uri: Set[str]) -> None:
        """
        Initialize training for DrillSample.

        @param pos_uri: A set of positive examples where each example corresponds to a string representation of an individual/instance.
        @param neg_uri: A set of negative examples where each example corresponds to a string representation of an individual/instance.
        @return:
        """
        # (1) Sample from positive and negative examples without replacement.
        if self.sample_size > len(pos_uri):
            print('positive examples less than ', self.sample_size)
            pos_uri = list(pos_uri)
        else:
            pos_uri = random.sample(pos_uri, self.sample_size)

        if self.sample_size > len(neg_uri):
            print('negative examples less than ', self.sample_size)
            neg_uri = list(neg_uri)
        else:
            neg_uri = random.sample(neg_uri, self.sample_size)

        # self.logger.info('Sampled E^+:[{0}] \t Sampled E^-:[{1}]'.format(len(pos_uri), len(neg_uri)))

        # (2) String to Owlready2 conversion of SAMPLED examples
        self.reward_func.pos = set(self.kb.convert_uri_instance_to_obj_from_iterable(pos_uri))
        self.reward_func.neg = set(self.kb.convert_uri_instance_to_obj_from_iterable(neg_uri))

        # (3) Assign embeddings of sampled examples.
        self.emb_pos = torch.tensor(self.instance_embeddings.loc[pos_uri].values, dtype=torch.float32)
        self.emb_neg = torch.tensor(self.instance_embeddings.loc[neg_uri].values, dtype=torch.float32)
        # (3.1) ADD ZEROS if lengths of the provided positive or negative examples are less than the required sample size
        if len(self.emb_pos) < self.sample_size:
            num_rows_to_fill = self.sample_size - len(self.emb_pos)
            self.emb_pos = torch.cat((torch.zeros(num_rows_to_fill, self.instance_embeddings.shape[1]), self.emb_pos))
        if len(self.emb_neg) < self.sample_size:
            num_rows_to_fill = self.sample_size - len(self.emb_neg)
            self.emb_neg = torch.cat((torch.zeros(num_rows_to_fill, self.instance_embeddings.shape[1]), self.emb_neg))

        self.emb_pos = self.emb_pos.view(1, self.emb_pos.shape[0], self.emb_pos.shape[1])
        self.emb_neg = self.emb_neg.view(1, self.emb_neg.shape[0], self.emb_neg.shape[1])

        # Sanity checking
        if torch.isnan(self.emb_pos).any() or torch.isinf(self.emb_pos).any():
            print(string_balanced_pos)
            raise ValueError('invalid value detected in E+,\n{0}'.format(self.emb_pos))
        if torch.isnan(self.emb_neg).any() or torch.isinf(self.emb_neg).any():
            raise ValueError('invalid value detected in E-,\n{0}'.format(self.emb_neg))
        # Default exploration exploitation tradeoff.
        self.epsilon = 1

    def fit(self, pos: Set[str], neg: Set[str], ignore: Set[str] = None, max_runtime=None):
        """
        Find hypotheses that explain pos and neg.
        """
        self.default_state_rl()
        self.initialize_learning_problem(pos=pos, neg=neg, all_instances=self.kb.thing.instances, ignore=ignore)
        self.emb_pos, self.emb_neg = self.represent_examples(pos=pos, neg=neg)
        self.start_time = time.time()
        if max_runtime:
            self.max_runtime = max_runtime
        for i in range(1, self.iter_bound):
            most_promising = self.next_node_to_expand(i)
            refinements = [ref for ref in self.apply_rho(most_promising)]
            if len(refinements) == 0:
                most_promising.heuristic = -1
                self.search_tree.add(most_promising)
                continue
            predicted_Q_values = self.predict_Q(current_state=most_promising, next_states=refinements)
            self.goal_found = self.update_search(refinements, predicted_Q_values)
            if self.goal_found:
                if self.terminate_on_goal:
                    return self.terminate()
            if time.time() - self.start_time > self.max_runtime:
                return self.terminate()
            if self.number_of_tested_concepts >= self.max_num_of_concepts_tested:
                return self.terminate()
        return self.terminate()

    def terminate_training(self):
        self.save_weights()
        # json serialize
        with open(self.storage_path + '/seen_lp.json', 'w') as file_descriptor:
            json.dump(self.seen_examples, file_descriptor, indent=3)
        return self


class DrillHeuristic(AbstractScorer):
    """
    Heuristic in Convolutional DQL concept learning.
    Heuristic implements a convolutional neural network.
    """

    def __init__(self, pos=None, neg=None, model=None, mode=None, model_args=None):
        super().__init__(pos, neg, unlabelled=None)
        if model:
            self.net = model
        elif mode in ['averaging', 'sampling']:
            self.net = Drill(model_args)
            self.mode = mode
            self.name = 'DrillHeuristic_' + self.mode
        else:
            raise ValueError
        self.net.eval()

    def score(self, node, parent_node=None):
        """ Compute heuristic value of root node only"""
        if parent_node is None and node.is_root:
            return torch.FloatTensor([.0001]).squeeze()
        raise ValueError

    def apply(self, node, parent_node=None):
        """ Assign predicted Q-value to node object."""
        predicted_q_val = self.score(node, parent_node)
        node.heuristic = predicted_q_val


class Drill(nn.Module):
    """
    A neural model for Deep Q-Learning.

    An input Drill has the following form
            1. indexes of individuals belonging to current state (s).
            2. indexes of individuals belonging to next state state (s_prime).
            3. indexes of individuals provided as positive examples.
            4. indexes of individuals provided as negative examples.

    Given such input, we from a sparse 3D Tensor where  each slice is a **** N *** by ***D***
    where N is the number of individuals and D is the number of dimension of embeddings.
    Given that N on the current benchmark datasets < 10^3, we can get away with this computation. By doing so
    we do not need to subsample from given inputs.

    """

    def __init__(self, args):
        super(Drill, self).__init__()
        self.in_channels, self.embedding_dim = args['input_shape']
        self.loss = nn.MSELoss()

        self.conv1 = nn.Conv2d(in_channels=self.in_channels,
                               out_channels=args['first_out_channels'],
                               kernel_size=args['kernel_size'],
                               padding=1, stride=1, bias=True)
        self.bn1 = nn.BatchNorm1d(args['first_out_channels'])

        self.conv2 = nn.Conv2d(in_channels=args['first_out_channels'],
                               out_channels=args['second_out_channels'],
                               kernel_size=args['kernel_size'], padding=1, stride=1, bias=True)
        self.bn2 = nn.BatchNorm1d(args['second_out_channels'])

        self.conv3 = nn.Conv2d(in_channels=args['second_out_channels'],
                               out_channels=args['third_out_channels'],
                               kernel_size=args['kernel_size'], padding=1, stride=1, bias=True)
        self.bn3 = nn.BatchNorm1d(args['third_out_channels'])

        # Fully connected layers.

        self.size_of_fc1 = int(args['third_out_channels'] * self.embedding_dim)
        self.fc1 = nn.Linear(in_features=self.size_of_fc1, out_features=self.size_of_fc1 // 2)

        self.bn4 = nn.BatchNorm2d(self.size_of_fc1 // 2)

        self.fc2 = nn.Linear(in_features=self.size_of_fc1 // 2, out_features=self.size_of_fc1 // 4)
        self.bn5 = nn.BatchNorm1d(self.size_of_fc1 // 4)

        self.fc3 = nn.Linear(in_features=self.size_of_fc1 // 4, out_features=1)

        assert self.__sanity_checking(torch.rand(32, 4, 1, self.embedding_dim)).shape == (32, 1)

    def init(self):
        xavier_normal_(self.fc1.weight.data)
        xavier_normal_(self.fc2.weight.data)
        xavier_normal_(self.fc3.weight.data)

        xavier_normal_(self.conv1.weight.data)
        xavier_normal_(self.conv2.weight.data)
        xavier_normal_(self.conv3.weight.data)

    def __sanity_checking(self, X):
        return self.forward(X)

    def forward(self, X: torch.FloatTensor):
        # X denotes a batch of tensors where each tensor has the shape of (4, 1, embedding_dim)
        # 4 => S, S', E^+, E^- \in R^embedding_dim

        # @TODO Later Add Residual learning

        # In => torch.Size([batchsize, 4, 1, self.embedding_dim])
        # Out => torch.Size([batchsize, 32, 1, self.embedding_dim]) # 32 number of input channels
        X = F.relu(self.conv1(X))

        # Out => torch.Size([batchsize, 16, 1, self.embedding_dim]) # 16 number of input channels
        X = F.relu(self.conv2(X))

        # Out => torch.Size([batchsize, 8, 1, self.embedding_dim]) # 8 number of input channels
        X = F.relu(self.conv3(X))

        # Out => torch.Size([batchsize, 8, * self.embedding_dim]) # 8 number of input channels
        X = X.view(X.shape[0], X.shape[1] * X.shape[2] * X.shape[3])

        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        return F.relu(self.fc3(X))
