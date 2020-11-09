from abc import ABCMeta
from .concept_learner import BaseConceptLearner
from .base_rl_agent import BaseRLTrainer
from .abstracts import AbstractDrill
from .util import *
from .search import Node, SearchTreePriorityQueue
import random
import torch
from torch import nn
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


class DrillAverage(AbstractDrill, BaseConceptLearner, BaseRLAgent):
    def __init__(self, knowledge_base, refinement_operator=None, quality_func=F1(),
                 heuristic_func=None, iter_bound=None, num_episode=759, max_num_of_concepts_tested=None, verbose=None,
                 terminate_on_goal=True, instance_embeddings=None, ignored_concepts=None,
                 num_of_sequential_actions=None):
        AbstractDrill.__init__(self, model=heuristic_func.model, instance_embeddings=instance_embeddings)
        if refinement_operator is None:
            refinement_operator = LengthBasedRefinement(kb=knowledge_base)
        if num_of_sequential_actions is None:
            num_of_sequential_actions = 10

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
        BaseRLAgent.__init__(self, reward_func=Reward(), num_episode=num_episode,
                             num_of_sequential_actions=num_of_sequential_actions)

        self.emb_pos, self.emb_neg = None, None
        self.storage_path, _ = create_experiment_folder()
        self.logger = create_logger(name='Drill', p=self.storage_path)


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
        current_state_batch = []
        next_state_batch = []
        q_values = []
        for experience in self.experiences:
            s, s_prime, q = experience
            current_state_batch.append(s.concept.embeddings)
            next_state_batch.append(s_prime.concept.embeddings)
            q_values.append(q)

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
            self.apply_demonstration(example_node)
            self.rl_learning_loop(pos_uri=pos, neg_uri=neg)
            # Save model.
            torch.save(self.model.state_dict(), self.storage_path + '/model.pth')
            self.seen_examples.setdefault(example_node.concept.str, dict()).update(
                {'Positives': list(pos), 'Negatives': list(neg)})
        # json serialize
        with open(self.storage_path + '/seen_lp.json', 'w') as file_descriptor:
            json.dump(self.seen_examples, file_descriptor, indent=3)


class DrillHeuristic(AbstractScorer):
    """
    Use pretrained agent as heuristic func.
    """

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
                    print('Default init.')
                    self.model = Drill()
        else:
            if model:
                self.model = model
            else:
                print('Default init.')
                self.model = Drill()
        self.model.eval()

    def score(self, node, parent_node=None):
        """ Compute and return predicted Q-value"""
        if parent_node is None and node.is_root:
            return torch.FloatTensor([.0001]).squeeze()

        print('asd')
        with torch.no_grad():
            print(node.concept.embeddings)
            print(parent_node.concept.embeddings)

            exit(1)

            self.S_Prime = next_state_batch
            self.S = current_state.expand(self.S_Prime.shape)
            self.Positives = p.expand(next_state_batch.shape)
            self.Negatives = n.expand(next_state_batch.shape)
            assert self.S.shape == self.S_Prime.shape == self.Positives.shape == self.Negatives.shape
            assert self.S.dtype == self.S_Prime.dtype == self.Positives.dtype == self.Negatives.dtype == torch.float32
            # X.shape()=> batch_size,4, embedding dim)
            self.X = torch.cat([self.S, self.S_Prime, self.Positives, self.Negatives], 1)
            num_points, depth, dim = self.X.shape
            self.X = self.X.view(num_points, depth, dim)

            raise ValueError()

    def apply(self, node, parent_node=None):
        """ Assign predicted Q-value to node object."""
        predicted_q_val = self.score(node, parent_node)
        node.heuristic = predicted_q_val


"""
class DrillTrainer(BaseRLTrainer):
    def __init__(self,
                 knowledge_base,
                 refinement_operator,
                 quality_func=F1(),
                 search_tree=SearchTreePriorityQueue(),
                 reward_func=Reward(),
                 path_pretrained_agent=None,
                 learning_problem_generator=None,
                 instance_embeddings=None,
                 num_epochs_per_replay=10,
                 num_episode=759,
                 epsilon=1.0,
                 epsilon_decay=.001,
                 max_len_replay_memory=1024,
                 verbose=0):
        super().__init__(kb=knowledge_base,
                         rho=refinement_operator,
                         quality_func=quality_func,
                         search_tree=search_tree,
                         reward_func=reward_func,
                         num_epochs_per_replay=num_epochs_per_replay,
                         num_episode=num_episode,
                         epsilon=epsilon,
                         epsilon_decay=epsilon_decay,
                         max_len_replay_memory=max_len_replay_memory,
                         verbose=verbose)

        # Hyperparameters to reproduce.
        # Concept Learning related.
        self.increment_length = 1  # LengthBasedRho(x,maxlength=len(x)+increment_length)
        self.max_num_rho_applied = 15
        # Nets related.
        self.learning_rate, self.decay_rate, self.batch_size = .001, 1.0, 256
        self.num_epochs_per_replay = 5
        # DQL related.
        #self.replay_modulo_per_episode = 10
        if path_pretrained_agent:
            self.model = Drill()
            self.model.load_state_dict(torch.load(path_pretrained_agent))
        else:
            self.model = Drill()
            self.model.init()
        self.lp_gen = learning_problem_generator
        self.instance_embeddings = instance_embeddings  # Normalize input if necessary

        self.experiences = deque(maxlen=self.max_len_replay_memory)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = ExponentialLR(self.optimizer, self.decay_rate)

        self.pos, self.neg = None, None
        self.emb_pos, self.emb_neg = None, None

        # Logging and using cuda.
        self.logger = create_logger(name='Drill', p=self.storage_path)
        settings = dict()

        settings.update({'batch_size': self.batch_size,
                         'learning_rate': self.learning_rate,
                         'decay_rate': self.decay_rate,
                         'num_of_epochs': self.num_epochs_per_replay,
                         'max_len_replay_memory': self.max_len_replay_memory,
                         'epsilon': self.epsilon,
                         'epsilon_decay': self.epsilon_decay,
                         'max_replay_mem': self.max_len_replay_memory,
                         'learn_per_modulo_iter': self.replay_modulo_per_episode,
                         'max_num_rho_applied': self.max_num_rho_applied})
        with open(self.storage_path + '/settings.json', 'w') as file_descriptor:
            json.dump(settings, file_descriptor)

        self.verbose = verbose
        self.seen_examples = dict()


    def apply_rho(self, node: Node) -> List:
        assert isinstance(node, Node)
        return [self.rho.getNode(i, parent_node=node) for i in
                self.rho.refine(node, maxlength=len(node) + self.increment_length)]

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

    def next_node_to_expand(self, current_state: Node, next_states: List[Node]) -> Node:
        if np.random.random() < self.epsilon:  # Exploration
            next_state = random.choice(next_states)
            self.assign_embeddings(next_state)
        else:  # Exploitation
            next_state = self.exploitation(current_state, next_states)
        return next_state

    def assign_embeddings(self, node: Node):
        if node.concept.embeddings is None:
            str_idx = [get_full_iri(i).replace('\n', '') for i in node.concept.instances]
            emb = torch.tensor(self.instance_embeddings.loc[str_idx].values, dtype=torch.float32)
            emb = torch.mean(emb, dim=0)
            emb = emb.view(1, 1, emb.shape[0])
            node.concept.embeddings = emb
        if torch.isnan(node.concept.embeddings).any() or torch.isinf(node.concept.embeddings).any():
            node.concept.embeddings = torch.zeros((1, 1, self.instance_embeddings.shape[1]))

    def sequence_of_actions(self, root):
        current_state = root
        path_of_concepts = []
        rewards = []

        for _ in range(self.max_num_rho_applied):
            next_states = self.apply_rho(current_state)
            if len(next_states) == 0:  # DEAD END, what to do ?
                break
            next_state = self.next_node_to_expand(current_state, next_states)
            assert next_state
            assert current_state
            if next_state.concept.str == 'Nothing':  # Dead END
                break
            path_of_concepts.append((current_state, next_state))
            rewards.append(self.reward_func.calculate(current_state, next_state))
            current_state = next_state

        return path_of_concepts, rewards

    def form_experiences(self, state_pairs: List, rewards: List) -> None:

        for th, consecutive_states in enumerate(state_pairs):
            e, e_next = consecutive_states
            self.experiences.append(
                (e, e_next, max(rewards[th:])))  # given e, e_next, Q val is the max Q value reachable.

    # @performance_debugger(func_name='learn_from_replay_memory')
    def learn_from_replay_memory(self) -> None:
        current_state_batch = []
        next_state_batch = []
        q_values = []
        for experience in self.experiences:
            s, s_prime, q = experience
            current_state_batch.append(s.concept.embeddings)
            next_state_batch.append(s_prime.concept.embeddings)
            q_values.append(q)

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

    def start(self):
        for example_node in self.lp_gen:
            # Instances of example concept conversion to URIs in string format.
            # All concept learners must be able to perform on string representations of instances.
            string_all_pos = set(
                self.kb.convert_owlready2_individuals_to_uri_from_iterable(example_node.concept.instances))
            string_all_neg = set(self.kb.convert_owlready2_individuals_to_uri_from_iterable(
                self.kb.get_all_individuals().difference(example_node.concept.instances)))
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
                continue
            # string to owlready2 object conversion
            self.pos = set(self.kb.convert_uri_instance_to_obj_from_iterable(string_balanced_pos))
            self.neg = set(self.kb.convert_uri_instance_to_obj_from_iterable(string_balanced_neg))
            self.reward_func.pos = self.pos
            self.reward_func.neg = self.neg

            self.emb_pos = torch.tensor(self.instance_embeddings.loc[list(string_balanced_pos)].values,
                                        dtype=torch.float32)
            self.emb_neg = torch.tensor(self.instance_embeddings.loc[list(string_balanced_neg)].values,
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

            self.train(TargetConcept=example_node.concept.str)  # Let model to train to solve given problem

            self.seen_examples.setdefault(example_node.concept.str, dict()).update(
                {'Positives': list(string_balanced_pos), 'Negatives': list(string_balanced_neg)})

        # json serialize
        with open(self.storage_path + '/seen_lp.json', 'w') as file_descriptor:
            json.dump(self.seen_examples, file_descriptor)

    # @performance_debugger(func_name='train')
    def train(self, TargetConcept: str) -> None:
        
        root = self.rho.getNode(self.start_class, root=True)
        self.assign_embeddings(root)
        sum_of_rewards_per_actions = []
        self.epsilon = 1
        for th in range(self.num_episode):  # Inner training loop
            path_of_concepts, rewards = self.sequence_of_actions(root)

            if th % 10 == 0:
                self.logger.info(
                    '{0}.th iter. SumOfRewards: {1:.2f}\tEpsilon:{2:.2f}\t Num.Exp:{3}'.format(th, sum(rewards),
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

        plt.scatter(range(len(sum_of_rewards_per_actions)), sum_of_rewards_per_actions)
        plt.ylabel('Sum of rewards per episode')
        plt.xlabel('Episodes')
        plt.title('DQL agent on {0}'.format(TargetConcept))
        plt.savefig(self.storage_path + '/{0}_RewardsPerEpisode.pdf'.format(TargetConcept), fontsize=16)
        if self.verbose > 0:
            plt.show()
        # Save model.
        torch.save(self.model.state_dict(), self.storage_path + '/model.pth')

    def test(self, pos: Set[AnyStr], neg: Set[AnyStr]):
        print('Test start.')
        # string to owlready2 object conversion
        self.pos = {self.kb.__str_to_instance_obj[i] for i in pos}
        self.neg = {self.kb.__str_to_instance_obj[i] for i in neg}

        self.search_tree.quality_func.set_positive_examples(self.pos)
        self.search_tree.quality_func.set_negative_examples(self.neg)

        self.search_tree.quality_func.applied = 0

        # owlready obj to  integer conversion
        self.idx_pos = torch.LongTensor([self.kb.idx_of_instances[_] for _ in self.pos])
        self.idx_neg = torch.LongTensor([self.kb.idx_of_instances[_] for _ in self.neg])
        self.search_tree.heuristic_func = DrillHeuristic(pos=self.idx_pos, neg=self.idx_neg, model=self.model)
        self.search_tree.clean()
        assert len(self.search_tree) == 0  # search tree is empty.
        # init pretrained agent as heuristic func.
        root = self.rho.getNode(self.start_class, root=True)
        self.assign_embeddings(root)

        self.search_tree.quality_func.apply(root)
        self.search_tree.add_root(root)
        goal_found = False
        for _ in range(10):
            node_to_expand = self.search_tree.get_most_promising()
            for ref in self.apply_rho(node_to_expand):

                self.assign_embeddings(ref)
                goal_found = self.search_tree.add_node(node=ref, parent_node=node_to_expand)
                if goal_found:
                    print(
                        'Goal found after {0} number of concepts tested.'.format(self.search_tree.quality_func.applied))
                    break
            if goal_found:
                break
        print('#### Top predictions after {0} number of concepts tested. ###'.format(
            self.search_tree.quality_func.applied))
        for i in self.search_tree.get_top_n(n=10):
            print(i)
        print('#######')
"""


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



class PrepareBatchOfTraining(torch.utils.data.Dataset):

    def __init__(self, current_state_batch: torch.Tensor, next_state_batch: torch.Tensor, p: torch.Tensor,
                 n: torch.Tensor, q: torch.Tensor):
        if torch.isnan(current_state_batch).any() or torch.isinf(current_state_batch).any():
            raise ValueError('invalid value detected in current_state_batch,\n{0}'.format(current_state_batch))
        if torch.isnan(next_state_batch).any() or torch.isinf(next_state_batch).any():
            raise ValueError('invalid value detected in next_state_batch,\n{0}'.format(next_state_batch))
        if torch.isnan(p).any() or torch.isinf(p).any():
            raise ValueError('invalid value detected in p,\n{0}'.format(p))
        if torch.isnan(n).any() or torch.isinf(n).any():
            raise ValueError('invalid value detected in p,\n{0}'.format(n))
        if torch.isnan(q).any() or torch.isinf(q).any():
            raise ValueError('invalid Q value  detected during batching.')

        self.S = current_state_batch
        self.S_Prime = next_state_batch
        self.y = q.view(len(q), 1)
        assert self.S.shape == self.S_Prime.shape
        assert len(self.y) == len(self.S)

        self.Positives = p.expand(next_state_batch.shape)
        self.Negatives = n.expand(next_state_batch.shape)
        assert self.S.shape == self.S_Prime.shape == self.Positives.shape == self.Negatives.shape
        assert self.S.dtype == self.S_Prime.dtype == self.Positives.dtype == self.Negatives.dtype == torch.float32
        # X.shape()=> batch_size,4,embeddingdim)
        self.X = torch.cat([self.S, self.S_Prime, self.Positives, self.Negatives], 1)
        num_points, depth, dim = self.X.shape
        self.X = self.X.view(num_points, depth, dim)

        if torch.isnan(self.X).any() or torch.isinf(self.X).any():
            print('invalid input detected during batching in X')
            raise ValueError
        if torch.isnan(self.y).any() or torch.isinf(self.y).any():
            print('invalid Q value  detected during batching in Y')
            raise ValueError
        # self.X, self.y = self.X.to(device), self.y.to(device)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
