from collections import OrderedDict, defaultdict
from functools import total_ordering
from abc import ABCMeta, abstractmethod, ABC
from .owlapy.model import OWLClassExpression
from .util import get_full_iri, balanced_sets, read_csv

from typing import Set, Dict, List, Tuple, Iterable, Generator, SupportsFloat
import random
import pandas as pd
import torch
from .data_struct import PrepareBatchOfTraining, PrepareBatchOfPrediction, Experience
import json
import numpy as np
import time

random.seed(0)


@total_ordering
class BaseConcept(metaclass=ABCMeta):
    """Base class for Concept."""
    __slots__ = ['owl', 'full_iri', 'str', 'is_atomic', '__instances', '__idx_instances', 'length',
                 'form', 'role', 'filler', 'concept_a', 'concept_b']

    @abstractmethod
    def __init__(self, concept: ThingClass, kwargs, world=None):
        assert isinstance(concept, ThingClass)
        assert kwargs['form'] in ['Class', 'ObjectIntersectionOf', 'ObjectUnionOf', 'ObjectComplementOf',
                                  'ObjectSomeValuesFrom', 'ObjectAllValuesFrom']

        self.owl = concept
        self.world = world
        self.full_iri = get_full_iri(concept)  # .namespace.base_iri + concept.name
        self.str = concept.name
        self.form = kwargs['form']

        self.is_atomic = True if self.form == 'Class' else False  # self.__is_atomic()  # TODO consider the necessity.
        self.length = self.__calculate_length()
        self.__idx_instances = None

        self.__instances = {jjj for jjj in self.owl.instances(world=self.world)}  # be sure of the memory usage.
        if self.__instances is None:
            self.__instances = set()

    @property
    def instances(self) -> Set:
        """ Returns all instances belonging to the concept."""
        return self.__instances

    @instances.setter
    def instances(self, x: Set):
        """ Setter of instances."""
        self.__instances = x

    @property
    def idx_instances(self):
        """ Getter of integer indexes of instances."""
        return self.__idx_instances

    @idx_instances.setter
    def idx_instances(self, x):
        """ Setter of idx_instances."""
        self.__idx_instances = x

    def __str__(self):
        return '{self.__repr__}\t{self.full_iri}'.format(self=self)

    def __len__(self):
        return self.length

    def __calculate_length(self):
        """
        The length of a concept is defined as
        the sum of the numbers of
            concept names, role names, quantifiers,and connective symbols occurring in the concept

        The length |A| of a concept CAis defined inductively:
        |A| = |\top| = |\bot| = 1
        |¬D| = |D| + 1
        |D \sqcap E| = |D \sqcup E| = 1 + |D| + |E|
        |∃r.D| = |∀r.D| = 2 + |D|
        :return:
        """
        num_of_exists = self.str.count("∃")
        num_of_for_all = self.str.count("∀")
        num_of_negation = self.str.count("¬")
        is_dot_here = self.str.count('.')

        num_of_operand_and_operator = len(self.str.split())
        count = num_of_negation + num_of_operand_and_operator + num_of_exists + is_dot_here + num_of_for_all
        return count

    def __is_atomic(self):
        """
        @todo Atomic class definition must be explicitly defined.
        Currently we consider all concepts having length=1 as atomic.
        :return: True if self is atomic otherwise False.
        """
        if '∃' in self.str or '∀' in self.str:
            return False
        elif '⊔' in self.str or '⊓' in self.str or '¬' in self.str:
            return False
        return True

    def __lt__(self, other):
        return self.length < other.length

    def __gt__(self, other):
        return self.length > other.length


@total_ordering
class BaseNode(metaclass=ABCMeta):
    """Base class for Concept."""
    __slots__ = ['concept', '__heuristic_score', '__horizontal_expansion', '__quality_score',
                 '___refinement_count', '__refinement_count', '__depth', '__children', '__embeddings', 'length',
                 'parent_node']

    @abstractmethod
    def __init__(self, concept: OWLClassExpression, parent_node, is_root=False):
        self.__quality_score, self.__heuristic_score = None, None
        self.__is_root = is_root
        self.__horizontal_expansion, self.__refinement_count = 0, 0
        self.concept = concept
        self.parent_node = parent_node
        self.__embeddings = None
        self.__children = set()
        self.length = len(self.concept)

        if self.parent_node is None:
            assert len(concept) == 1 and self.__is_root
            self.__depth = 0
        else:
            self.__depth = self.parent_node.depth + 1

    def __len__(self):
        return len(self.concept)

    @property
    def embeddings(self):
        return self.__embeddings

    @embeddings.setter
    def embeddings(self, value):
        self.__embeddings = value

    @property
    def children(self):
        return self.__children

    @property
    def refinement_count(self):
        return self.__refinement_count

    @refinement_count.setter
    def refinement_count(self, n):
        self.__refinement_count = n

    @property
    def depth(self):
        return self.__depth

    @depth.setter
    def depth(self, n: int):
        self.__depth = n

    @property
    def h_exp(self):
        return self.__horizontal_expansion

    @property
    def heuristic(self) -> float:
        return self.__heuristic_score

    @heuristic.setter
    def heuristic(self, val: float):
        self.__heuristic_score = val

    @property
    def quality(self) -> float:
        return self.__quality_score

    @quality.setter
    def quality(self, val: float):
        self.__quality_score = val

    @property
    def is_root(self):
        return self.__is_root

    def add_children(self, n):
        self.__children.add(n)

    def remove_child(self, n):
        self.__children.remove(n)

    def increment_h_exp(self, val=0):
        self.__horizontal_expansion += val + 1

    def __lt__(self, other):
        return self.concept.length < other.concept.length

    def __gt__(self, other):
        return self.concept.length > other.concept.length


class AbstractScorer(ABC):
    """
    An abstract class for quality and heuristic functions.
    """

    @abstractmethod
    def __init__(self, pos, neg, unlabelled):
        self.pos = pos
        self.neg = neg
        self.unlabelled = unlabelled
        self.applied = 0

    def set_positive_examples(self, instances):
        self.pos = instances

    def set_negative_examples(self, instances):
        self.neg = instances

    def set_unlabelled_examples(self, instances):
        self.unlabelled = instances

    @abstractmethod
    def score(self, *args, **kwargs):
        pass

    @abstractmethod
    def apply(self, *args, **kwargs):
        pass

    def clean(self):
        self.pos = None
        self.neg = None
        self.unlabelled = None
        self.applied = 0


class BaseRefinement(metaclass=ABCMeta):
    """
    Base class for Refinement Operators.

    Let C, D \in N_c where N_c os a finite set of concepts.

    * Proposition 3.3 (Complete and Finite Refinement Operators) [1]
        ** ρ(C) = {C ⊓ T} ∪ {D | D is not empty AND D \sqset C}
        *** The operator is finite,
        *** The operator is complete as given a concept C, we can reach an arbitrary concept D such that D subset of C.

    *) Theoretical Foundations of Refinement Operators [1].




    *) Defining a top-down refimenent operator that is a proper is crutial.
        4.1.3 Achieving Properness [1]
    *) Figure 4.1 [1] defines of the refinement operator

    [1] Learning OWL Class Expressions
    """

    @abstractmethod
    def __init__(self, kb, max_size_of_concept=10_000, min_size_of_concept=0):
        self.kb = kb
        self.max_size_of_concept = max_size_of_concept
        self.min_size_of_concept = min_size_of_concept
        # self.concepts_to_nodes = dict()

    def set_kb(self, kb):
        self.kb = kb

    # def set_concepts_node_mapping(self, m: dict):
    #    self.concepts_to_nodes = m

    @abstractmethod
    def getNode(self, *args, **kwargs):
        pass

    @abstractmethod
    def refine(self, *args, **kwargs):
        pass

    @abstractmethod
    def refine_atomic_concept(self, *args, **kwargs):
        pass

    @abstractmethod
    def refine_complement_of(self, *args, **kwargs):
        pass

    @abstractmethod
    def refine_object_some_values_from(self, *args, **kwargs):
        pass

    @abstractmethod
    def refine_object_all_values_from(self, *args, **kwargs):
        pass

    @abstractmethod
    def refine_object_union_of(self, *args, **kwargs):
        pass

    @abstractmethod
    def refine_object_intersection_of(self, *args, **kwargs):
        pass


class AbstractTree(ABC):
    @abstractmethod
    def __init__(self, quality_func, heuristic_func):
        self.quality_func = quality_func
        self.heuristic_func = heuristic_func
        self._nodes = dict()

    def __len__(self):
        return len(self._nodes)

    def __getitem__(self, item):
        return self._nodes[item]

    def __setitem__(self, k, v):
        self._nodes[k] = v

    def __iter__(self):
        for k, node in self._nodes.items():
            yield node

    def get_top_n_nodes(self, n: int, key='quality'):
        self.sort_search_tree_by_decreasing_order(key=key)
        for ith, dict_ in enumerate(self._nodes.items()):
            if ith >= n:
                break
            k, node = dict_
            yield node

    def set_quality_func(self, f: AbstractScorer):
        self.quality_func = f

    def set_heuristic_func(self, h):
        self.heuristic_func = h

    def redundancy_check(self, n):
        if n in self._nodes:
            return False
        return True

    @property
    def nodes(self):
        return self._nodes

    @abstractmethod
    def add(self, *args, **kwargs):
        pass

    def sort_search_tree_by_decreasing_order(self, *, key: str):
        if key == 'heuristic':
            sorted_x = sorted(self._nodes.items(), key=lambda kv: kv[1].heuristic, reverse=True)
        elif key == 'quality':
            sorted_x = sorted(self._nodes.items(), key=lambda kv: kv[1].quality, reverse=True)
        elif key == 'length':
            sorted_x = sorted(self._nodes.items(), key=lambda kv: len(kv[1]), reverse=True)
        else:
            raise ValueError('Wrong Key. Key must be heuristic, quality or concept_length')

        self._nodes = OrderedDict(sorted_x)

    def best_hypotheses(self, n=10) -> List[BaseNode]:
        assert self.search_tree is not None
        assert len(self.search_tree) > 1
        return [i for i in self.search_tree.get_top_n_nodes(n)]

    def show_search_tree(self, th, top_n=10):
        """
        Show search tree.
        """
        print('######## ', th, 'step Search Tree ###########')
        predictions = list(self.get_top_n_nodes(top_n))
        for ith, node in enumerate(predictions):
            print('{0}-\t{1}\t{2}:{3}\tHeuristic:{4}:'.format(ith + 1, node.concept.str,
                                                              self.quality_func.name, node.quality, node.heuristic))
        print('######## Search Tree ###########\n')
        return predictions

    def show_best_nodes(self, top_n, key=None):
        assert key
        self.sort_search_tree_by_decreasing_order(key=key)
        return self.show_search_tree('Final', top_n=top_n + 1)

    @staticmethod
    def save_current_top_n_nodes(key=None, n=10, path=None):

        """
        Save current top_n nodes
        """
        assert path
        assert key
        assert isinstance(n, int)
        pass

    def clean(self):
        """
        Clearn
        @return:
        """
        self._nodes.clear()


class AbstractKnowledgeBase(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def save(self, path: str, rdf_format="rdfxml"):
        pass

    def describe(self):
        print(f'Number of concepts: {len(self.uri_to_concepts)}\n'
              f'Number of individuals: {len(self.individuals)}\n'
              f'Number of properties: {len(self.property_hierarchy)}')

    @abstractmethod
    def clean(self):
        raise NotImplementedError

    @property
    def concepts(self) -> Dict:
        """
        Returns a dictionary where keys are string representation of concept objects
        and values are concept objects.
        @return:
        """
        return dict(zip([i.str for i in self.uri_to_concepts.values()], self.uri_to_concepts.values()))

    def get_all_concepts(self):
        return set(self.uri_to_concepts.values())


class AbstractDrill(ABC):
    """
    Abstract class for Convolutional DQL concept learning
    """

    def __init__(self, path_of_embeddings, reward_func, learning_rate=None,
                 num_episode=None, num_of_sequential_actions=None, max_len_replay_memory=None,
                 representation_mode=None, batch_size=1024, epsilon_decay=None, epsilon_min=None,
                 num_epochs_per_replay=None, num_workers=32):
        # @TODO refactor the code for the sake of readability
        self.instance_embeddings = read_csv(path_of_embeddings)
        self.embedding_dim = self.instance_embeddings.shape[1]
        self.reward_func = reward_func
        assert reward_func
        self.representation_mode = representation_mode
        assert representation_mode in ['averaging', 'sampling']
        # Will be filled by child class
        self.heuristic_func = None
        self.num_workers = num_workers
        # constants
        self.epsilon = 1
        self.learning_rate = learning_rate
        self.num_episode = num_episode
        self.num_of_sequential_actions = num_of_sequential_actions
        self.num_epochs_per_replay = num_epochs_per_replay
        self.max_len_replay_memory = max_len_replay_memory
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size

        if self.learning_rate is None:
            self.learning_rate = .001
        if self.num_episode is None:
            self.num_episode = 759
        if self.max_len_replay_memory is None:
            self.max_len_replay_memory = 1024
        if self.num_of_sequential_actions is None:
            self.num_of_sequential_actions = 10
        if self.epsilon_decay is None:
            self.epsilon_decay = .001
        if self.epsilon_min is None:
            self.epsilon_min = 0
        if self.num_epochs_per_replay is None:
            self.num_epochs_per_replay = 100

        # will be filled
        self.optimizer = None  # torch.optim.Adam(self.model_net.parameters(), lr=self.learning_rate)

        self.seen_examples = dict()
        self.emb_pos, self.emb_neg = None, None
        self.start_time = None
        self.goal_found = False
        self.experiences = Experience(maxlen=self.max_len_replay_memory)

    def default_state_rl(self):
        self.emb_pos, self.emb_neg = None, None
        self.goal_found = False
        self.start_time = None

    @abstractmethod
    def init_training(self, *args, **kwargs):
        """
        Initialize training for a given E+,E- and K.
        @param args:
        @param kwargs:
        @return:
        """

    @abstractmethod
    def terminate_training(self):
        """
        Save weights and training data after training phase.
        @return:
        """

    def next_node_to_expand(self, t: int = None) -> BaseNode:
        """
        Return a node that maximizes the heuristic function at time t
        @param t:
        @return:
        """
        if self.verbose > 1:
            self.search_tree.show_search_tree(t)
        return self.search_tree.get_most_promising()

    def form_experiences(self, state_pairs: List, rewards: List) -> None:
        """
        Form experiences from a sequence of concepts and corresponding rewards.

        state_pairs - a list of tuples containing two consecutive states
        reward      - a list of reward.

        Gamma is 1.

        Return
        X - a list of embeddings of current concept, next concept, positive examples, negative examples
        y - argmax Q value.
        """

        for th, consecutive_states in enumerate(state_pairs):
            e, e_next = consecutive_states
            self.experiences.append(
                (e, e_next, max(rewards[th:])))  # given e, e_next, Q val is the max Q value reachable.

    def learn_from_replay_memory(self) -> None:
        """
        Learning by replaying memory
        @return:
        """

        current_state_batch, next_state_batch, q_values = self.experiences.retrieve()
        current_state_batch = torch.cat(current_state_batch, dim=0)
        next_state_batch = torch.cat(next_state_batch, dim=0)
        q_values = torch.Tensor(q_values)

        try:
            assert current_state_batch.shape[1] == next_state_batch.shape[1] == self.emb_pos.shape[1] == \
                   self.emb_neg.shape[1]

        except AssertionError as e:
            print(current_state_batch.shape)
            print(next_state_batch.shape)
            print(self.emb_pos.shape)
            print(self.emb_neg.shape)
            print('Wrong format.')
            print(e)
            exit(1)

        assert current_state_batch.shape[2] == next_state_batch.shape[2] == self.emb_pos.shape[2] == self.emb_neg.shape[
            2]
        dataset = PrepareBatchOfTraining(current_state_batch=current_state_batch,
                                         next_state_batch=next_state_batch,
                                         p=self.emb_pos, n=self.emb_neg, q=q_values)
        num_experience = len(dataset)
        data_loader = torch.utils.data.DataLoader(dataset,
                                                  batch_size=self.batch_size, shuffle=True,
                                                  num_workers=self.num_workers)
        print(f'Number of experiences:{num_experience}')
        print('DQL agent is learning via experience replay')
        self.heuristic_func.net.train()
        for m in range(self.num_epochs_per_replay):
            total_loss = 0
            for X, y in data_loader:
                self.optimizer.zero_grad()  # zero the gradient buffers
                # forward
                predicted_q = self.heuristic_func.net.forward(X)
                # loss
                loss = self.heuristic_func.net.loss(predicted_q, y)
                total_loss += loss.item()
                # compute the derivative of the loss w.r.t. the parameters using backpropagation
                loss.backward()
                # clip gradients if gradients are killed. =>torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
        self.heuristic_func.net.train().eval()

    def sequence_of_actions(self, root: BaseNode) -> Tuple[List[Tuple[BaseNode, BaseNode]], List[SupportsFloat]]:
        """
        Perform self.num_of_sequential_actions number of actions

        (1) Make a sequence of **self.num_of_sequential_actions** actions
            (1.1) Get next states in a generator and convert them to list
            (1.2) Exit, if If there is no next state. @TODO Is it possible to 0 next states ?! Nothing should in the set of refinements, shouldn't it ?, i.e. [Nothing]
            (1.3) Find next state.
            (1.4) Exit, if next state is **Nothing**
            (1.5) Compute reward.
            (1.6) Update current state.

        (2) Return path_of_concepts, rewards

        """
        assert isinstance(root, BaseNode)

        current_state = root
        path_of_concepts = []
        rewards = []
        # (1)
        for _ in range(self.num_of_sequential_actions):
            # (1.1)
            next_states = list(self.apply_rho(current_state))
            # (1.2)
            if len(next_states) == 0:  # DEAD END
                assert (len(current_state) + 3) <= self.max_child_length
                break
            # (1.3)
            next_state = self.exploration_exploitation_tradeoff(current_state, next_states)

            # (1.3)
            if next_state.concept.str == 'Nothing':  # Dead END
                break
            # (1.4)
            path_of_concepts.append((current_state, next_state))
            # (1.5)
            rewards.append(self.reward_func.calculate(current_state, next_state))
            # (1.6)
            current_state = next_state
        # (2)
        return path_of_concepts, rewards

    def update_search(self, concepts, predicted_Q_values):
        """
        @param concepts:
        @param predicted_Q_values:
        @return:
        """
        # simple loop.
        for child_node, pred_Q in zip(concepts, predicted_Q_values):
            child_node.heuristic = pred_Q
            self.search_tree.quality_func.apply(child_node)
            if child_node.quality > 0:  # > too weak, ignore.
                self.search_tree.add(child_node)
            if child_node.quality == 1:
                return child_node

    def apply_rho(self, node: BaseNode) -> Generator:
        """
        Refine an OWL Class expression |= Observing next possible states

        Computation O(N).

        1. Generate concepts by refining a node
            1.1 Compute allowed length of refinements
            1.2. Convert concepts if concepts do not belong to  self.concepts_to_ignore
                Note that          i.str not in self.concepts_to_ignore => O(1) if a set is being used.
        3. Return Generator
        """
        assert isinstance(node, BaseNode)
        # 1.
        # (1.1)
        length = len(node) + 3 if len(node) + 3 <= self.max_child_length else self.max_child_length
        # (1.2)
        for i in self.rho.refine(node, maxlength=length):  # O(N)
            if i.str not in self.concepts_to_ignore:  # O(1)
                yield self.rho.getNode(i, parent_node=node)  # O(1)

    def assign_embeddings(self, node: BaseNode) -> None:
        assert isinstance(node, BaseNode)
        # (1) Detect mode
        if self.representation_mode == 'averaging':
            # (2) if input node has not seen before, assign embeddings.
            if node.embeddings is None:
                str_idx = [get_full_iri(i).replace('\n', '') for i in node.concept.instances]
                if len(str_idx) == 0:
                    emb = torch.zeros(self.sample_size, self.instance_embeddings.shape[1])
                else:
                    emb = torch.tensor(self.instance_embeddings.loc[str_idx].values, dtype=torch.float32)
                    emb = torch.mean(emb, dim=0)
                emb = emb.view(1, self.sample_size, self.instance_embeddings.shape[1])
                node.embeddings = emb
            else:
                """ Embeddings already assigned."""
                try:
                    assert node.embeddings.shape == (1, self.sample_size, self.instance_embeddings.shape[1])
                except AssertionError as e:
                    print(e)
                    print(node)
                    print(node.embeddings.shape)
                    print((1, self.sample_size, self.instance_embeddings.shape[1]))
                    exit(1)
        elif self.representation_mode == 'sampling':
            if node.embeddings is None:
                str_idx = [get_full_iri(i).replace('\n', '') for i in node.concept.instances]
                if len(str_idx) >= self.sample_size:
                    sampled_str_idx = random.sample(str_idx, self.sample_size)
                    emb = torch.tensor(self.instance_embeddings.loc[sampled_str_idx].values, dtype=torch.float32)
                else:
                    num_rows_to_fill = self.sample_size - len(str_idx)
                    emb = torch.tensor(self.instance_embeddings.loc[str_idx].values, dtype=torch.float32)
                    emb = torch.cat((torch.zeros(num_rows_to_fill, self.instance_embeddings.shape[1]), emb))
                emb = emb.view(1, self.sample_size, self.instance_embeddings.shape[1])
                node.embeddings = emb
            else:
                """ Embeddings already assigned."""
                try:
                    assert node.embeddings.shape == (1, self.sample_size, self.instance_embeddings.shape[1])
                except AssertionError:
                    print(node)
                    print(self.sample_size)
                    print(node.embeddings.shape)
                    print((1, self.sample_size, self.instance_embeddings.shape[1]))
                    raise ValueError
        else:
            raise ValueError

        # @todo remove this testing in experiments.
        if torch.isnan(node.embeddings).any() or torch.isinf(node.embeddings).any():
            # No individual contained in the input concept.
            # Sanity checking.
            raise ValueError

    def save_weights(self):
        """
        Save pytorch weights.
        @return:
        """
        # Save model.
        torch.save(self.heuristic_func.net.state_dict(),
                   self.storage_path + '/{0}.pth'.format(self.heuristic_func.name))

    def rl_learning_loop(self, pos_uri: Set[str], neg_uri: Set[str]) -> List[float]:
        """
        RL agent learning loop over learning problem defined
        @param pos_uri: A set of URIs indicating E^+
        @param neg_uri: A set of URIs indicating E^-

        Computation

        1. Initialize training

        2. Learning loop: Stopping criteria
            ***self.num_episode** OR ***self.epsilon < self.epsilon_min***

        2.1. Perform sequence of actions

        2.2. Decrease exploration rate

        2.3. Form experiences

        2.4. Experience Replay

        2.5. Return sum of actions

        @return: List of sum of rewards per episode.
        """

        # (1)
        self.init_training(pos_uri=pos_uri, neg_uri=neg_uri)
        root = self.rho.getNode(self.start_class, root=True)
        self.assign_embeddings(root)

        sum_of_rewards_per_actions = []
        log_every_n_episodes = int(self.num_episode * .1) + 1

        # (2)
        for th in range(self.num_episode):
            # (2.1)
            sequence_of_states, rewards = self.sequence_of_actions(root)

            if th % log_every_n_episodes == 0:
                self.logger.info(
                    '{0}.th iter. SumOfRewards: {1:.2f}\tEpsilon:{2:.2f}\t|ReplayMem.|:{3}'.format(th, sum(rewards),
                                                                                                   self.epsilon, len(
                            self.experiences)))

            # (2.2)
            self.epsilon -= self.epsilon_decay
            if self.epsilon < self.epsilon_min:
                break

            # (2.3)
            self.form_experiences(sequence_of_states, rewards)

            # (2.4)
            if th % self.num_epochs_per_replay == 0 and len(self.experiences) > 1:
                self.learn_from_replay_memory()
            sum_of_rewards_per_actions.append(sum(rewards))
        return sum_of_rewards_per_actions

    def exploration_exploitation_tradeoff(self, current_state: BaseNode, next_states: List[BaseNode]) -> BaseNode:
        """
        Exploration vs Exploitation tradeoff at finding next state.
        (1) Exploration
        (2) Exploitation
        """
        if np.random.random() < self.epsilon:
            next_state = random.choice(next_states)
            self.assign_embeddings(next_state)
        else:
            next_state = self.exploitation(current_state, next_states)
        return next_state

    def exploitation(self, current_state: BaseNode, next_states: List[BaseNode]) -> BaseNode:
        """
        Find next node that is assigned with highest predicted Q value.

        (1) Predict Q values : predictions.shape => torch.Size([n, 1]) where n = len(next_states)

        (2) Find the index of max value in predictions

        (3) Use the index to obtain next state.

        (4) Return next state.
        """
        predictions: torch.Tensor = self.predict_Q(current_state, next_states)
        argmax_id = int(torch.argmax(predictions))
        next_state = next_states[argmax_id]
        """
        # Sanity checking
        print('#'*10)
        for s, q in zip(next_states, predictions):
            print(s, q)
        print('#'*10)
        print(next_state,f'\t {torch.max(predictions)}')
        """
        return next_state

    def predict_Q(self, current_state: BaseNode, next_states: List[BaseNode]) -> torch.Tensor:
        """
        Predict promise of next states given current state.
        @param current_state:
        @param next_states:
        @return: predicted Q values.
        """
        self.assign_embeddings(current_state)
        assert len(next_states) > 0
        with torch.no_grad():
            self.heuristic_func.net.eval()
            # create batch batch.
            next_state_batch = []
            for _ in next_states:
                self.assign_embeddings(_)
                next_state_batch.append(_.embeddings)
            next_state_batch = torch.cat(next_state_batch, dim=0)
            ds = PrepareBatchOfPrediction(current_state.embeddings,
                                          next_state_batch,
                                          self.emb_pos,
                                          self.emb_neg)
            predictions = self.heuristic_func.net.forward(ds.get_all())
        return predictions

    def train(self, dataset: Iterable[Tuple[str, Set, Set]], relearn_ratio: int = 2):
        """
        Train RL agent on learning problems with relearn_ratio.
        @param dataset: An iterable containing training data. Each item corresponds to a tuple of string representation
        of target concept, a set of positive examples in the form of URIs amd a set of negative examples in the form of
        URIs, respectively.
        @param relearn_ratio: An integer indicating the number of times dataset is iterated.

        # @TODO determine Big-O

        Computation
        1. Dataset and relearn_ratio loops: Learn each problem relearn_ratio times,

        2. Learning loop

        3. Take post process action that implemented by subclass.

        @return: self
        """
        # We need a better way of login,
        self.logger.info('Training starts.')
        print(f'Training starts.\nNumber of learning problem:{len(dataset)},\t Relearn ratio:{relearn_ratio}')
        counter = 1
        # 1.
        for _ in range(relearn_ratio):
            for (alc_concept_str, positives, negatives) in dataset:
                self.logger.info(
                    'Goal Concept:{0}\tE^+:[{1}] \t E^-:[{2}]'.format(alc_concept_str,
                                                                      len(positives), len(negatives)))
                # 2.
                print(f'RL training on {counter}.th learning problem starts')
                sum_of_rewards_per_actions = self.rl_learning_loop(pos_uri=positives, neg_uri=negatives)

                print(f'Sum of Rewards in first 3 trajectory:{sum_of_rewards_per_actions[:3]}')
                print(f'Sum of Rewards in last 3 trajectory:{sum_of_rewards_per_actions[:3]}')
                self.seen_examples.setdefault(counter, dict()).update(
                    {'Concept': alc_concept_str, 'Positives': list(positives), 'Negatives': list(negatives)})

                counter += 1
                if counter % 100 == 0:
                    self.save_weights()
                # 3.
        return self.terminate_training()
