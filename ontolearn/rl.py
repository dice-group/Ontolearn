from abc import ABCMeta
from .concept_learner import BaseConceptLearner
from .abstracts import AbstractDrill, AbstractScorer
from .util import *
from .search import Node, SearchTreePriorityQueue
from .data_struct import PrepareBatchOfTraining, PrepareBatchOfPrediction, Experience
from .refinement_operators import LengthBasedRefinement
from .metrics import F1
from .heuristics import Reward
from concurrent.futures import ThreadPoolExecutor
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
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ExponentialLR
import random

random.seed(1)


class DrillAverage(AbstractDrill, BaseConceptLearner):
    def __init__(self, knowledge_base, refinement_operator=None, quality_func=F1(),
                 heuristic_func=None, iter_bound=None, max_num_of_concepts_tested=None, verbose=None,
                 terminate_on_goal=True, instance_embeddings=None, ignored_concepts=None,
                 max_len_replay_memory=None, batch_size=None, epsilon_decay=None, epsilon_min=None,
                 num_epochs_per_replay=None, learning_rate=None,
                 num_of_sequential_actions=None, num_episode=None):

        AbstractDrill.__init__(self, model=heuristic_func.model, reward_func=Reward(),
                               max_len_replay_memory=max_len_replay_memory,
                               batch_size=batch_size, epsilon_min=epsilon_min,
                               num_epochs_per_replay=num_epochs_per_replay,
                               epsilon_decay=epsilon_decay,
                               num_of_sequential_actions=num_of_sequential_actions, num_episode=num_episode,
                               instance_embeddings=instance_embeddings, learning_rate=learning_rate)
        BaseConceptLearner.__init__(self, knowledge_base=knowledge_base,
                                    refinement_operator=refinement_operator,
                                    search_tree=SearchTreePriorityQueue(),
                                    quality_func=quality_func,
                                    heuristic_func=heuristic_func,
                                    ignored_concepts=ignored_concepts,
                                    terminate_on_goal=terminate_on_goal,
                                    iter_bound=iter_bound,
                                    max_num_of_concepts_tested=max_num_of_concepts_tested,
                                    verbose=verbose)

        self.experiences = Experience(maxlen=self.max_len_replay_memory)
        self.emb_pos, self.emb_neg = None, None
        self.storage_path, _ = create_experiment_folder()
        self.logger = create_logger(name='DrillAverage', p=self.storage_path)

    def represent_examples(self, *, pos, neg) -> None:
        """

        @param pos:
        @param neg:
        @return:
        """
        assert isinstance(pos, set) and isinstance(neg, set)

        self.emb_pos = torch.tensor(self.instance_embeddings.loc[list(pos)].values,
                                    dtype=torch.float32)
        self.emb_neg = torch.tensor(self.instance_embeddings.loc[list(neg)].values,
                                    dtype=torch.float32)
        assert self.emb_pos.shape[0] == len(pos)
        assert self.emb_neg.shape[0] == len(neg)

        self.emb_pos = torch.mean(self.emb_pos, dim=0)
        self.emb_pos = self.emb_pos.view(1, 1, self.emb_pos.shape[0])
        self.emb_neg = torch.mean(self.emb_neg, dim=0)
        self.emb_neg = self.emb_neg.view(1, 1, self.emb_neg.shape[0])

    def fit(self, pos: Set[AnyStr], neg: Set[AnyStr]):
        """
        @param pos: A set of str representations of given positive examples/individuals.
        @param neg: A set of str representations of given negative examples/individuals.
        @return:
        """
        self.initialize_learning_problem(pos=pos, neg=neg, all_instances=self.kb.thing.instances)
        self.represent_examples(pos=pos, neg=neg)
        for i in range(1, self.iter_bound):
            most_promising = self.next_node_to_expand(i)
            refinements = [ref for ref in self.apply_rho(most_promising)]
            if len(refinements) == 0:
                #print('Dead end')
                #print(most_promising)
                break
            predicted_Q_values = self.predict_Q(current_state=most_promising, next_states=refinements)
            self.goal_found = self.update_search(refinements, predicted_Q_values)
            if self.goal_found:
                if self.terminate_on_goal:
                    return self.terminate()
            if self.number_of_tested_concepts >= self.max_num_of_concepts_tested:
                return self.terminate()
        return self.terminate()

    # RL starts
    def form_experiences(self, state_pairs: List, rewards: List) -> None:
        """
        Form experiences from a sequence of concepts and corresponding rewards.

        state_pairs - a list of tuples containing two consecutive states
        reward      - a list of reward.

        Return
        X - a list of embeddings of current concept, next concept, positive examples, negative examples
        y - argmax Q value.
        """

        for th, consecutive_states in enumerate(state_pairs):
            e, e_next = consecutive_states
            self.experiences.append(
                (e, e_next, max(rewards[th:])))  # given e, e_next, Q val is the max Q value reachable.

    def learn_from_replay_memory(self) -> None:
        current_state_batch, next_state_batch, q_values = self.experiences.retrieve()
        current_state_batch = torch.cat(current_state_batch, dim=0)
        next_state_batch = torch.cat(next_state_batch, dim=0)
        q_values = torch.Tensor(q_values)

        ds = PrepareBatchOfTraining(current_state_batch=current_state_batch,
                                    next_state_batch=next_state_batch,
                                    p=self.emb_pos, n=self.emb_neg, q=q_values)
        self.model.train()
        for m in range(self.num_epochs_per_replay):
            total_loss = 0
            for X, y in DataLoader(ds, batch_size=self.batch_size, shuffle=True, num_workers=4):
                if len(X) == 1:
                    continue
                self.optimizer.zero_grad()  # zero the gradient buffers
                # forward
                predicted_q = self.model.forward(X)
                # loss
                loss = self.model.loss(predicted_q, y)
                total_loss += loss.item()
                # compute the derivative of the loss w.r.t. the parameters using backpropagation
                loss.backward()
                # clip gradients if gradients are killed. =>torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
        self.model.eval()

    def __preprocess_examples(self, example_node):
        # Instances of example concept conversion to URIs in string format.
        # All concept learners must be able to perform on string representations of instances.
        string_all_pos = set(
            self.kb.convert_owlready2_individuals_to_uri_from_iterable(example_node.concept.instances))
        string_all_neg = set(self.kb.convert_owlready2_individuals_to_uri_from_iterable(
            self.kb.individuals.difference(example_node.concept.instances)))
        data_set_info = 'Target Concept:{0}\t |E+|:{1}\t |E-|:{2}'.format(example_node.concept.str,
                                                                          len(string_all_pos),
                                                                          len(string_all_neg))
        # create balanced setting
        string_balanced_pos, string_balanced_neg = balanced_sets(string_all_pos, string_all_neg)
        data_set_info += '\tBalanced |E+|:{0}\t|E-|:{1}:'.format(len(string_balanced_pos), len(string_balanced_neg))
        self.logger.info(data_set_info)
        try:
            assert len(string_balanced_pos) > 0 and len(string_balanced_neg)
        except AssertionError:
            self.logger.info('Balancing is not possible. Example will be skipped.')
            return False, None, None
        return True, string_balanced_pos, string_balanced_neg

    def exploitation(self, current_state: Node, next_states: List[Node]) -> Node:
        self.assign_embeddings(current_state)
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
            argmax_id = int(torch.argmax(predictions))
            next_state = next_states[argmax_id]
        return next_state

    def exploration_exploitation_tradeoff(self, current_state: Node, next_states: List[Node]) -> Node:
        """
        Exploration vs Exploitation tradeoff at finding next state.
        """
        if np.random.random() < self.epsilon:  # Exploration
            next_state = random.choice(next_states)
            self.assign_embeddings(next_state)
        else:  # Exploitation
            next_state = self.exploitation(current_state, next_states)
        return next_state

    def sequence_of_actions(self, root):
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

    def rl_learning_loop(self, pos_uri: Set[AnyStr], neg_uri: Set[AnyStr]):

        # string to owlready2 object conversion
        pos = set(self.kb.convert_uri_instance_to_obj_from_iterable(pos_uri))
        neg = set(self.kb.convert_uri_instance_to_obj_from_iterable(neg_uri))
        self.reward_func.pos = pos
        self.reward_func.neg = neg

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
        root = self.rho.getNode(self.start_class, root=True)
        self.assign_embeddings(root)
        sum_of_rewards_per_actions = []
        self.epsilon = 1
        for th in range(self.num_episode):  # Inner training loop
            path_of_concepts, rewards = self.sequence_of_actions(root)
            if th % 10 == 0:
                self.logger.info(
                    '{0}.th iter. SumOfRewards: {1:.2f}\tEpsilon:{2:.2f}\t|ReplayMem.|:{3}'.format(th, sum(rewards),
                                                                                                   self.epsilon,
                                                                                                   len(
                                                                                                       self.experiences)))
            self.epsilon -= self.epsilon_decay
            if self.epsilon < self.epsilon_min:
                break
            self.form_experiences(path_of_concepts, rewards)
            if th % self.num_epochs_per_replay == 0 and len(self.experiences) > 1:
                self.learn_from_replay_memory()

            sum_of_rewards_per_actions.append(sum(rewards))
        return sum_of_rewards_per_actions

    def apply_demonstration(self, goal_node):
        goal_path = retrieve_concept_hierarchy(goal_node)
        rewards = None
        self.form_experiences(goal_path, rewards)
        raise NotImplementedError

    def train(self, learning_problems):
        for example_node in learning_problems:
            flag, pos, neg = self.__preprocess_examples(example_node)
            if flag is False:
                continue
            # self.apply_demonstration(example_node)
            self.rl_learning_loop(pos_uri=pos, neg_uri=neg)
            # Save model.
            torch.save(self.model.state_dict(), self.storage_path + '/model.pth')
            self.seen_examples.setdefault(example_node.concept.str, dict()).update(
                {'Positives': list(pos), 'Negatives': list(neg)})
        # json serialize
        with open(self.storage_path + '/seen_lp.json', 'w') as file_descriptor:
            json.dump(self.seen_examples, file_descriptor, indent=3)


class DrillSample(AbstractDrill, BaseConceptLearner):
    def __init__(self, knowledge_base, refinement_operator,
                 quality_func=F1(),
                 heuristic_func=None, iter_bound=None, num_episode=None, max_num_of_concepts_tested=None, verbose=None,
                 sample_size=10, terminate_on_goal=True, instance_embeddings=None,
                 ignored_concepts=None, num_of_sequential_actions=None):
        AbstractDrill.__init__(self, model=heuristic_func.model,
                               instance_embeddings=instance_embeddings,
                               reward_func=Reward(), num_episode=num_episode,
                               num_of_sequential_actions=num_of_sequential_actions)
        self.sample_size = sample_size

        BaseConceptLearner.__init__(self, knowledge_base=knowledge_base,
                                    refinement_operator=refinement_operator,
                                    search_tree=SearchTreePriorityQueue(),
                                    quality_func=quality_func,
                                    heuristic_func=heuristic_func,
                                    ignored_concepts=ignored_concepts,
                                    terminate_on_goal=terminate_on_goal,
                                    iter_bound=iter_bound,
                                    max_num_of_concepts_tested=max_num_of_concepts_tested,
                                    verbose=verbose)
        self.experiences = Experience(maxlen=self.max_len_replay_memory)
        self.emb_pos, self.emb_neg = None, None
        self.storage_path, _ = create_experiment_folder()
        self.logger = create_logger(name='DrillSample', p=self.storage_path)

    def represent_examples(self, *, pos, neg) -> None:
        """

        @param pos:
        @param neg:
        @return:
        """
        assert isinstance(pos, set) and isinstance(neg, set)

        self.emb_pos = torch.tensor(self.instance_embeddings.loc[list(pos)].values,
                                    dtype=torch.float32)
        self.emb_neg = torch.tensor(self.instance_embeddings.loc[list(neg)].values,
                                    dtype=torch.float32)
        assert self.emb_pos.shape[0] == len(pos)
        assert self.emb_neg.shape[0] == len(neg)

        self.emb_pos = torch.mean(self.emb_pos, dim=0)
        self.emb_pos = self.emb_pos.view(1, 1, self.emb_pos.shape[0])
        self.emb_neg = torch.mean(self.emb_neg, dim=0)
        self.emb_neg = self.emb_neg.view(1, 1, self.emb_neg.shape[0])

    def fit(self, pos: Set[AnyStr], neg: Set[AnyStr]):
        """
        @param pos: A set of str representations of given positive examples/individuals.
        @param neg: A set of str representations of given negative examples/individuals.
        @return:
        """
        self.initialize_learning_problem(pos=pos, neg=neg, all_instances=self.kb.thing.instances)
        self.represent_examples(pos=pos, neg=neg)
        for i in range(1, self.iter_bound):
            most_promising = self.next_node_to_expand(i)
            refinements = [ref for ref in self.apply_rho(most_promising)]
            if len(refinements) == 0:
                print('Dead end')
                print(most_promising)
                continue
            predicted_Q_values = self.predict_Q(current_state=most_promising, next_states=refinements)
            self.goal_found = self.update_search(refinements, predicted_Q_values)
            if self.goal_found:
                if self.terminate_on_goal:
                    return self.terminate()
            if self.number_of_tested_concepts >= self.max_num_of_concepts_tested:
                return self.terminate()
        return self.terminate()

    # RL starts
    def form_experiences(self, state_pairs: List, rewards: List) -> None:
        """
        Form experiences from a sequence of concepts and corresponding rewards.

        state_pairs - a list of tuples containing two consecutive states
        reward      - a list of reward.

        Return
        X - a list of embeddings of current concept, next concept, positive examples, negative examples
        y - argmax Q value.
        """

        for th, consecutive_states in enumerate(state_pairs):
            e, e_next = consecutive_states
            self.experiences.append(
                (e, e_next, max(rewards[th:])))  # given e, e_next, Q val is the max Q value reachable.

    def learn_from_replay_memory(self) -> None:
        current_state_batch, next_state_batch, q_values = self.experiences.retrieve()

        current_state_batch = torch.cat(current_state_batch, dim=0)
        next_state_batch = torch.cat(next_state_batch, dim=0)
        q_values = torch.Tensor(q_values)

        ds = PrepareBatchOfTraining(current_state_batch=current_state_batch,
                                    next_state_batch=next_state_batch,
                                    p=self.emb_pos, n=self.emb_neg, q=q_values)
        self.model.train()
        for m in range(self.num_epochs_per_replay):
            total_loss = 0
            for X, y in DataLoader(ds, batch_size=self.batch_size, shuffle=True, num_workers=4):
                if len(X) == 1:
                    continue
                self.optimizer.zero_grad()  # zero the gradient buffers
                # forward
                predicted_q = self.model.forward(X)
                # loss
                loss = self.model.loss(predicted_q, y)
                total_loss += loss.item()
                # compute the derivative of the loss w.r.t. the parameters using backpropagation
                loss.backward()
                # clip gradients if gradients are killed. =>torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
        self.model.eval()

    def __preprocess_examples(self, example_node):
        # Instances of example concept conversion to URIs in string format.
        # All concept learners must be able to perform on string representations of instances.
        string_all_pos = set(
            self.kb.convert_owlready2_individuals_to_uri_from_iterable(example_node.concept.instances))
        string_all_neg = set(self.kb.convert_owlready2_individuals_to_uri_from_iterable(
            self.kb.individuals.difference(example_node.concept.instances)))
        data_set_info = 'Target Concept:{0}\t |E+|:{1}\t |E-|:{2}'.format(example_node.concept.str,
                                                                          len(string_all_pos),
                                                                          len(string_all_neg))
        # create balanced setting
        string_balanced_pos, string_balanced_neg = balanced_sets(string_all_pos, string_all_neg)
        data_set_info += '\tBalanced |E+|:{0}\t|E-|:{1}:'.format(len(string_balanced_pos), len(string_balanced_neg))
        self.logger.info(data_set_info)
        try:
            assert len(string_balanced_pos) > 0 and len(string_balanced_neg)
        except AssertionError:
            self.logger.info('Balancing is not possible. Example will be skipped.')
            return False, None, None
        return True, string_balanced_pos, string_balanced_neg

    def exploitation(self, current_state: Node, next_states: List[Node]) -> Node:
        self.assign_embeddings(current_state)
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
            argmax_id = int(torch.argmax(predictions))
            next_state = next_states[argmax_id]
        return next_state

    def exploration_exploitation_tradeoff(self, current_state: Node, next_states: List[Node]) -> Node:
        """
        Exploration vs Exploitation tradeoff at finding next state.
        """
        if np.random.random() < self.epsilon:  # Exploration
            next_state = random.choice(next_states)
            self.assign_embeddings(next_state)
        else:  # Exploitation
            next_state = self.exploitation(current_state, next_states)
        return next_state

    def sequence_of_actions(self, root: Node) -> Tuple[List[Tuple[Node, Node]], List]:
        """
        Sequential decision making in concept space.

        @param root:
        @return: Sequential decisions and respective rewards.
        """
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

    def rl_learning_loop(self, pos_uri: Set[AnyStr], neg_uri: Set[AnyStr]) -> None:
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
        root = self.rho.getNode(self.start_class, root=True)
        self.assign_embeddings(root, mode='sampling', sample_size=self.sample_size)
        sum_of_rewards_per_actions = []
        self.epsilon = 1
        for th in range(self.num_episode):  # Inner training loop
            path_of_concepts, rewards = self.sequence_of_actions(root)
            if th % 10 == 0:
                self.logger.info(
                    '{0}.th iter. SumOfRewards: {1:.2f}\tEpsilon:{2:.2f}\t|ReplayMem.|:{3}'.format(th, sum(rewards),
                                                                                                   self.epsilon,
                                                                                                   len(
                                                                                                       self.experiences)))
            self.epsilon -= self.epsilon_decay
            if self.epsilon < self.epsilon_min:
                break
            self.form_experiences(path_of_concepts, rewards)
            if th % self.num_epochs_per_replay == 0 and len(self.experiences) > 1:
                self.learn_from_replay_memory()

            sum_of_rewards_per_actions.append(sum(rewards))
        return sum_of_rewards_per_actions

    def train(self, learning_problems):
        for example_node in learning_problems:
            valid_example, pos, neg = self.__preprocess_examples(example_node)
            if valid_example:
                if len(pos) <= self.sample_size or len(neg) <= self.sample_size:
                    self.logger.info('|E+| or |E-| < {0}'.format(self.sample_size))
                    continue
                # RL learning loop, Given E+ and E-, find adequate hypotheses.
                self.rl_learning_loop(pos_uri=pos, neg_uri=neg)
                # Save model.
                torch.save(self.model.state_dict(),
                           self.storage_path + '/{0}_model.pth'.format(example_node.concept.str))
                self.seen_examples.setdefault(example_node.concept.str, dict()).update(
                    {'Positives': list(pos), 'Negatives': list(neg)})
        # json serialize
        with open(self.storage_path + '/seen_lp.json', 'w') as file_descriptor:
            json.dump(self.seen_examples, file_descriptor, indent=3)


class DrillHeuristic(AbstractScorer):
    def __init__(self, pos=None, neg=None, model_path=None, model=None):
        super().__init__(pos, neg, unlabelled=None)
        self.name = 'DrillHeuristic'
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


class Drill(nn.Module, metaclass=ABCMeta):
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

    def __init__(self):
        super(Drill, self).__init__()
        self.loss = nn.MSELoss()

        self.conv1 = nn.Conv1d(in_channels=4,
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

