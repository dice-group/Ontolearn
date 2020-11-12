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
from typing import List, Any, Set, AnyStr, Tuple
from collections import namedtuple, deque
from torch.nn.init import xavier_normal_
from itertools import chain
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ExponentialLR
import random

random.seed(1)


class DrillAverage(AbstractDrill, BaseConceptLearner):
    def __init__(self, knowledge_base, refinement_operator=None, quality_func=F1(),
                 iter_bound=None, max_num_of_concepts_tested=None, verbose=None,
                 terminate_on_goal=True, instance_embeddings=None, ignored_concepts=None,
                 max_len_replay_memory=None, batch_size=None, epsilon_decay=None, epsilon_min=None,
                 num_epochs_per_replay=None, learning_rate=None,
                 num_of_sequential_actions=None, num_episode=None):
        self.sample_size = 1
        self.heuristic_func = DrillHeuristic(mode='average',
                                             input_shape=(4 * self.sample_size, instance_embeddings.shape[1]))
        AbstractDrill.__init__(self, model=self.heuristic_func.model, reward_func=Reward(),
                               max_len_replay_memory=max_len_replay_memory,
                               batch_size=batch_size, epsilon_min=epsilon_min,
                               num_epochs_per_replay=num_epochs_per_replay,
                               representation_mode='averaging',
                               epsilon_decay=epsilon_decay,
                               num_of_sequential_actions=num_of_sequential_actions, num_episode=num_episode,
                               instance_embeddings=instance_embeddings, learning_rate=learning_rate)
        BaseConceptLearner.__init__(self, knowledge_base=knowledge_base,
                                    refinement_operator=refinement_operator,
                                    search_tree=SearchTreePriorityQueue(),
                                    quality_func=quality_func,
                                    heuristic_func=self.heuristic_func,
                                    ignored_concepts=ignored_concepts,
                                    terminate_on_goal=terminate_on_goal,
                                    iter_bound=iter_bound,
                                    max_num_of_concepts_tested=max_num_of_concepts_tested,
                                    verbose=verbose, name='DrillAverage')

        self.experiences = Experience(maxlen=self.max_len_replay_memory)
        self.emb_pos, self.emb_neg = None, None

    def represent_examples(self, *, pos: Set[AnyStr], neg: Set[AnyStr]) -> Tuple[torch.Tensor, torch.Tensor]:
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

    def fit(self, pos: Set[AnyStr], neg: Set[AnyStr], ignore: Set[AnyStr] = None):
        """
        Find hypotheses that explain pos and neg.
        """
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
            if self.number_of_tested_concepts >= self.max_num_of_concepts_tested:
                return self.terminate()
        return self.terminate()

    def train(self, learning_problems, n: int = 1):
        """
        Train RL agent on input learning problems n times.
        For each learning problem:
        (1) Obtain balanced E+ and E- from a syntactic goal node.
        (2) Input E+ and E- into rl agent learning loop.
        (3) Store parameterized model/net/Drill/Pytorch model.
        (4) Store E+ and E-.

        @param learning_problems:
        @param n:
        @return:
        """
        for _ in range(n):  # repeat training over learning problems.
            for (balanced_pos, balanced_neg, goal_concept) in self.preprocess_lp(learning_problems):
                self.rl_learning_loop(pos_uri=balanced_pos, neg_uri=balanced_neg)
                # Save model.
                torch.save(self.model.state_dict(), self.storage_path + '/model.pth')
                self.seen_examples.setdefault(str(n) + '_' + goal_concept.concept.str, dict()).update(
                    {'Positives': list(balanced_pos), 'Negatives': list(balanced_neg)})
        # json serialize
        with open(self.storage_path + '/seen_lp.json', 'w') as file_descriptor:
            json.dump(self.seen_examples, file_descriptor, indent=3)
        self.reset_state()
        return self

    def predict_Q(self, current_state: Node, next_states: List[Node]) -> torch.Tensor:
        """
        Predict promise of next states given current state.
        @param current_state:
        @param next_states:
        @return: predicted Q values.
        """
        self.assign_embeddings(current_state)
        assert len(next_states) > 0
        with torch.no_grad():
            self.model.eval()
            # create batch batch.
            next_state_batch = []
            for _ in next_states:
                self.assign_embeddings(_)
                next_state_batch.append(_.concept.embeddings)
            next_state_batch = torch.cat(next_state_batch, dim=0)
            ds = PrepareBatchOfPrediction(current_state.concept.embeddings,
                                          next_state_batch,
                                          self.emb_pos,
                                          self.emb_neg)
            predictions = self.model.forward(ds.get_all())
        return predictions

    def init_training(self, pos_uri: Set[AnyStr], neg_uri: Set[AnyStr]):
        """

        @param pos_uri:
        @param neg_uri:
        @return:
        """
        self.reset_state()
        # string to owlready2 object conversion
        self.reward_func.pos = set(self.kb.convert_uri_instance_to_obj_from_iterable(pos_uri))
        self.reward_func.neg = set(self.kb.convert_uri_instance_to_obj_from_iterable(neg_uri))

        self.emb_pos = torch.tensor(self.instance_embeddings.loc[list(pos_uri)].values,
                                    dtype=torch.float32)
        self.emb_neg = torch.tensor(self.instance_embeddings.loc[list(neg_uri)].values,
                                    dtype=torch.float32)
        # take mean and reshape it into (1,1,embedding_dim) for mini batching.
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
        self.epsilon = 1

class DrillSample(AbstractDrill, BaseConceptLearner):
    def __init__(self, knowledge_base, refinement_operator,
                 quality_func=F1(), iter_bound=None, num_episode=None, max_num_of_concepts_tested=None, verbose=None,
                 sample_size=10, terminate_on_goal=True, instance_embeddings=None,
                 ignored_concepts=None, num_of_sequential_actions=None):
        self.sample_size = sample_size
        self.heuristic_func = DrillHeuristic(mode='sample',
                                             input_shape=(4 * self.sample_size, instance_embeddings.shape[1]))
        AbstractDrill.__init__(self, model=self.heuristic_func.model,
                               instance_embeddings=instance_embeddings,
                               reward_func=Reward(), num_episode=num_episode,
                               representation_mode='sampling',
                               num_of_sequential_actions=num_of_sequential_actions)
        BaseConceptLearner.__init__(self, knowledge_base=knowledge_base,
                                    refinement_operator=refinement_operator,
                                    search_tree=SearchTreePriorityQueue(),
                                    quality_func=quality_func,
                                    heuristic_func=self.heuristic_func,
                                    ignored_concepts=ignored_concepts,
                                    terminate_on_goal=terminate_on_goal,
                                    iter_bound=iter_bound,
                                    max_num_of_concepts_tested=max_num_of_concepts_tested,
                                    verbose=verbose,
                                    name='DrillSample')
        self.experiences = Experience(maxlen=self.max_len_replay_memory)
        self.emb_pos, self.emb_neg = None, None

    def represent_examples(self, *, pos, neg) -> None:
        """

        @param pos:
        @param neg:
        @return:
        """
        assert isinstance(pos, set) and isinstance(neg, set)

        assert len(pos) >= self.sample_size and len(neg) >= self.sample_size
        sampled_pos = random.sample(pos, self.sample_size)
        sampled_neg = random.sample(neg, self.sample_size)

        self.emb_pos = torch.tensor(self.instance_embeddings.loc[sampled_pos].values,
                                    dtype=torch.float32)
        self.emb_neg = torch.tensor(self.instance_embeddings.loc[list(sampled_neg)].values,
                                    dtype=torch.float32)
        self.emb_pos = self.emb_pos.view(1, self.sample_size, self.instance_embeddings.shape[1])
        self.emb_neg = self.emb_neg.view(1, self.sample_size, self.instance_embeddings.shape[1])

    def fit(self, pos: Set[AnyStr], neg: Set[AnyStr], ignore: Set[AnyStr]):
        """
        Find hypotheses that explain pos and neg.
        """
        self.initialize_learning_problem(pos=pos, neg=neg, all_instances=self.kb.thing.instances, ignore=ignore)
        self.represent_examples(pos=pos, neg=neg)
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
            if self.number_of_tested_concepts >= self.max_num_of_concepts_tested:
                return self.terminate()
        return self.terminate()

    def train(self, learning_problems, n: int = 1):
        """
        Train RL agent on input learning problems n times.
        For each learning problem:
        (1) Obtain balanced E+ and E- from a syntactic goal node.
        (2) Input E+ and E- into rl agent learning loop.
        (3) Store parameterized model/net/Drill/Pytorch model.
        (4) Store E+ and E-.

        @param learning_problems:
        @param n:
        @return:
        """
        for _ in range(n):  # repeat training over learning problems.
            for (balanced_pos, balanced_neg, goal_concept) in self.preprocess_lp(learning_problems):
                self.rl_learning_loop(pos_uri=balanced_pos, neg_uri=balanced_neg)
                # Save model.
                torch.save(self.model.state_dict(), self.storage_path + '/model.pth')
                self.seen_examples.setdefault(str(n) + '_' + goal_concept.concept.str, dict()).update(
                    {'Positives': list(balanced_pos), 'Negatives': list(balanced_neg)})
        # json serialize
        with open(self.storage_path + '/seen_lp.json', 'w') as file_descriptor:
            json.dump(self.seen_examples, file_descriptor, indent=3)
        self.reset_state()
        return self

    def init_training(self, pos_uri: Set[AnyStr], neg_uri: Set[AnyStr]):
        """

        @param pos_uri:
        @param neg_uri:
        @return:
        """
        self.reset_state()

        # (1) Sample from positive and negative examples without replacement.
        pos_uri_sub = random.sample(pos_uri, self.sample_size)
        neg_uri_sub = random.sample(neg_uri, self.sample_size)

        # (2) Convert (1) into owlready2 objects.
        pos = set(self.kb.convert_uri_instance_to_obj_from_iterable(pos_uri_sub))
        neg = set(self.kb.convert_uri_instance_to_obj_from_iterable(neg_uri_sub))

        # (3) Assign (2) to reward function to calculate rewards.
        self.reward_func.pos = pos
        self.reward_func.neg = neg

        # (4) Assign embeddings from (1).
        self.emb_pos = torch.tensor(self.instance_embeddings.loc[pos_uri_sub].values, dtype=torch.float32)
        self.emb_neg = torch.tensor(self.instance_embeddings.loc[neg_uri_sub].values, dtype=torch.float32)

        self.emb_pos = self.emb_pos.view(1, self.emb_pos.shape[0], self.emb_pos.shape[1])
        self.emb_neg = self.emb_neg.view(1, self.emb_neg.shape[0], self.emb_neg.shape[1])
        # Sanity checking
        if torch.isnan(self.emb_pos).any() or torch.isinf(self.emb_pos).any():
            print(string_balanced_pos)
            raise ValueError('invalid value detected in E+,\n{0}'.format(self.emb_pos))
        if torch.isnan(self.emb_neg).any() or torch.isinf(self.emb_neg).any():
            raise ValueError('invalid value detected in E-,\n{0}'.format(self.emb_neg))
        self.epsilon = 1

    def predict_Q(self, current_state: Node, next_states: List[Node]) -> torch.Tensor:
        """
        Predict promise of next states given current state.
        @param current_state:
        @param next_states:
        @return: predicted Q values.
        """
        self.assign_embeddings(current_state)
        assert len(next_states) > 0
        with torch.no_grad():
            self.model.eval()
            # create batch batch.
            next_state_batch = []
            for _ in next_states:
                self.assign_embeddings(_)
                next_state_batch.append(_.concept.embeddings)
            next_state_batch = torch.cat(next_state_batch, dim=0)
            ds = PrepareBatchOfPrediction(current_state.concept.embeddings,
                                          next_state_batch,
                                          self.emb_pos,
                                          self.emb_neg)
            predictions = self.model.forward(ds.get_all())
        return predictions

    """
    def sequence_of_actions(self, root: Node) -> Tuple[List[Tuple[Node, Node]], List]:
        current_state = root
        path_of_concepts = []
        rewards = []
        for _ in range(self.num_of_sequential_actions):
            next_states = list(self.apply_rho(current_state))
            if len(next_states) == 0:  # DEAD END
                break
            next_state = self.exploration_exploitation_tradeoff(current_state, next_states)
            assert next_state
            assert current_state
            if next_state.concept.str == 'Nothing':  # Dead END
                break
            path_of_concepts.append((current_state, next_state))
            rewards.append(self.reward_func.calculate(current_state, next_state))
            current_state = next_state
        return path_of_concepts, rewards
    """


class DrillHeuristic(AbstractScorer):
    def __init__(self, pos=None, neg=None, model_path=None, model=None, mode=None, input_shape=None):
        super().__init__(pos, neg, unlabelled=None)
        self.name = 'DrillHeuristic'
        self.mode = mode
        self.length, self.width = input_shape
        if model:
            self.model = model
        elif self.mode == 'average':
            self.model = Drill(input_shape=(self.length, self.width))
        elif self.mode == 'sample':

            self.model = Drill(input_shape=(self.length, self.width))
        elif model_path:
            raise ValueError
        else:
            raise ValueError
        """
        if model_path:
            self.model = Drill()
            try:
                self.model.load_state_dict(torch.load(model_path))
            except (FileNotFoundError, KeyError) as e:
                print(e)
                if model:
                    self.model = model
                else:
                    self.model = Drill()
        else:
            if model:
                self.model = model
            else:
                self.model = Drill()
        """
        self.model.eval()

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

    def __init__(self, input_shape):
        super(Drill, self).__init__()
        self.l, self.w = input_shape
        self.loss = nn.MSELoss()

        self.conv1 = nn.Conv1d(in_channels=self.l,
                               out_channels=32,
                               kernel_size=3,
                               padding=1,
                               stride=1,
                               bias=True)
        self.pool = nn.MaxPool2d(kernel_size=3,
                                 padding=1,
                                 stride=1)
        self.bn1 = nn.BatchNorm1d(32)

        self.conv2 = nn.Conv1d(in_channels=32,
                               out_channels=16,
                               kernel_size=3,
                               padding=1,
                               stride=1,
                               bias=True)
        self.bn2 = nn.BatchNorm1d(16)

        self.conv3 = nn.Conv1d(in_channels=16,
                               out_channels=8,
                               kernel_size=3,
                               padding=1,
                               stride=1,
                               bias=True)

        self.fc1 = nn.Linear(in_features=800, out_features=200)
        self.bn3 = nn.BatchNorm1d(8)
        self.fc2 = nn.Linear(in_features=200, out_features=50)
        self.bn4 = nn.BatchNorm1d(200)
        self.fc3 = nn.Linear(in_features=50, out_features=1)
        self.bn5 = nn.BatchNorm1d(50)

    def init(self):
        xavier_normal_(self.fc1.weight.data)
        xavier_normal_(self.fc2.weight.data)
        xavier_normal_(self.fc3.weight.data)

        xavier_normal_(self.conv1.weight.data)
        xavier_normal_(self.conv2.weight.data)
        xavier_normal_(self.conv3.weight.data)

    def forward(self, X: torch.FloatTensor):
        # X => torch.Size([batchsize, 4, dim])
        X = self.bn1(self.pool(F.relu(self.conv1(X))))

        # X => torch.Size([batchsize, 32, dim]) # 32 kernels.
        X = self.bn2(self.pool(F.relu(self.conv2(X))))
        # X => torch.Size([batchsize, 16, dim]) # 16 kernels.
        X = self.bn3(self.pool(F.relu(self.conv3(X))))
        # X => torch.Size([batchsize, 8, dim]) # 16 kernels.
        X = X.view(-1, X.shape[1] * X.shape[2])

        X = self.bn4(F.relu(self.fc1(X)))
        X = self.bn5(F.relu(self.fc2(X)))
        return F.relu(self.fc3(X))
