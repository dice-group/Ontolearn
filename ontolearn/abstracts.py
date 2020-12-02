from collections import OrderedDict, defaultdict
from functools import total_ordering
from abc import ABCMeta, abstractmethod, ABC
from owlready2 import ThingClass, Ontology
from .util import get_full_iri, balanced_sets
from typing import Set, Dict, List, Tuple
import random
import pandas as pd
import torch
from .data_struct import PrepareBatchOfTraining, PrepareBatchOfPrediction
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
    def __init__(self, concept, parent_node, is_root=False):
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
        pass

    def clean(self):
        """
        Clearn
        @return:
        """
        self._nodes.clear()


class AbstractKnowledgeBase(ABC):

    def __init__(self):
        self.uri_to_concepts = dict()
        self.thing = None
        self.nothing = None
        self.top_down_concept_hierarchy = defaultdict(set)  # Next time thing about including this into Concepts.
        self.top_down_direct_concept_hierarchy = defaultdict(set)
        self.down_top_concept_hierarchy = defaultdict(set)
        self.down_top_direct_concept_hierarchy = defaultdict(set)
        self.concepts_to_leafs = defaultdict(set)
        self.property_hierarchy = None
        self.individuals = None
        self.uri_individuals = None  # string representation of uris

    def save(self, path, rdf_format="rdfxml"):
        """
        path .owl
        @param path:
        @param rdf_format:
        @return:
        """
        self.onto.save(file=path, format=rdf_format)

    def describe(self):
        print('Number of individuals: {0}'.format(len(self.individuals)))
        print('Number of concepts: {0}'.format(len(self.uri_to_concepts)))

    @abstractmethod
    def clean(self):
        raise NotImplementedError


class AbstractDrill(ABC):

    def __init__(self, drill_heuristic, instance_embeddings, reward_func, learning_rate=None,
                 num_episode=None, num_of_sequential_actions=None, max_len_replay_memory=None,
                 representation_mode=None, batch_size=None, epsilon_decay=None, epsilon_min=None,
                 num_epochs_per_replay=None):

        assert isinstance(instance_embeddings, pd.DataFrame)
        assert (instance_embeddings.all()).all()  # all columns and all rows are not none.
        assert drill_heuristic
        assert drill_heuristic.model

        assert reward_func
        self.representation_mode = representation_mode
        self.drill_heuristic = drill_heuristic
        self.model_name = self.drill_heuristic.name
        self.model_net = self.drill_heuristic.model
        self.instance_embeddings = instance_embeddings
        self.reward_func = reward_func

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
            self.num_epochs_per_replay = 10
        if self.batch_size is None:
            self.batch_size = 1024

        self.optimizer = torch.optim.Adam(self.model_net.parameters(), lr=self.learning_rate)

        self.seen_examples = dict()
        self.emb_pos, self.emb_neg = None, None
        self.start_time = None
        self.goal_found = False

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

        assert current_state_batch.shape[2] == next_state_batch.shape[2] == self.emb_pos.shape[2] == \
               self.emb_neg.shape[2]

        ds = PrepareBatchOfTraining(current_state_batch=current_state_batch,
                                    next_state_batch=next_state_batch,
                                    p=self.emb_pos, n=self.emb_neg, q=q_values)
        self.model_net.train()
        for m in range(self.num_epochs_per_replay):
            total_loss = 0
            for X, y in torch.utils.data.DataLoader(ds, batch_size=self.batch_size, shuffle=True, num_workers=4):
                if len(X) == 1:
                    continue
                self.optimizer.zero_grad()  # zero the gradient buffers
                # forward
                predicted_q = self.model_net.forward(X)
                # loss
                loss = self.model_net.loss(predicted_q, y)
                total_loss += loss.item()
                # compute the derivative of the loss w.r.t. the parameters using backpropagation
                loss.backward()
                # clip gradients if gradients are killed. =>torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
        self.model_net.eval()

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

    def apply_rho(self, node: BaseNode):
        assert isinstance(node, BaseNode)
        refinements = (self.rho.getNode(i, parent_node=node) for i in
                       self.rho.refine(node,
                                       maxlength=len(node) + 3 if len(node) + 3 <= self.max_length else self.max_length)
                       if i.str not in self.concepts_to_ignore)
        return refinements

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
        torch.save(self.model_net.state_dict(), self.storage_path + '/{0}_{1}.pth'.format(time.time(), self.model_name))

    def rl_learning_loop(self, pos_uri: Set[str], neg_uri: Set[str]) -> List[float]:
        """
        RL agent training loop over given positive and negative examples.


        @return: List of sum of rewards per episode.
        """
        # (1) initialize training.
        self.init_training(pos_uri=pos_uri, neg_uri=neg_uri)
        root = self.rho.getNode(self.start_class, root=True)
        # (2) Assign embeddings of root/first state.
        self.assign_embeddings(root)
        sum_of_rewards_per_actions = []
        for th in range(self.num_episode):
            # (3) Take sequence of actions.
            path_of_concepts, rewards = self.sequence_of_actions(root)
            if th % 100 == 0:
                """
                self.logger.info('{0}.th iter. SumOfRewards: {1:.2f}\tEpsilon:{2:.2f}\t|ReplayMem.|:{3}'.format(th, sum(rewards),self.epsilon,len(self.experiences)))
                """
            # (4) Decrease exploration rate.
            self.epsilon -= self.epsilon_decay
            if self.epsilon < self.epsilon_min:
                break
            # (5) Form experience.
            self.form_experiences(path_of_concepts, rewards)
            # (6) Adjust weights through experience replay.
            if th % self.num_epochs_per_replay == 0 and len(self.experiences) > 1:
                self.learn_from_replay_memory()
            sum_of_rewards_per_actions.append(sum(rewards))
        return sum_of_rewards_per_actions

    def exploration_exploitation_tradeoff(self, current_state: BaseNode, next_states: List[BaseNode]) -> BaseNode:
        """
        Exploration vs Exploitation tradeoff at finding next state.
        """
        if np.random.random() < self.epsilon:  # Exploration
            next_state = random.choice(next_states)
            self.assign_embeddings(next_state)
        else:  # Exploitation
            next_state = self.exploitation(current_state, next_states)
        return next_state

    def exploitation(self, current_state: BaseNode, next_states: List[BaseNode]) -> BaseNode:
        """

        @param current_state:
        @param next_states:
        @return:
        """
        self.assign_embeddings(current_state)
        with torch.no_grad():
            self.model_net.eval()
            # create batch batch.
            next_state_batch = []
            for n in next_states:
                self.assign_embeddings(n)
                next_state_batch.append(n.embeddings)
            next_state_batch = torch.cat(next_state_batch, dim=0)

            ds = PrepareBatchOfPrediction(current_state.embeddings,
                                          next_state_batch,
                                          self.emb_pos,
                                          self.emb_neg)
            predictions = self.model_net.forward(ds.get_all())
            argmax_id = int(torch.argmax(predictions))
            next_state = next_states[argmax_id]
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
            self.model_net.eval()
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
            predictions = self.model_net.forward(ds.get_all())
        return predictions

    def train(self, dataset: List[Tuple[str, Set, Set]], relearn_ratio: int = 1):
        """
        Train RL agent on learning problems with relearn_ratio.

        @param dataset: An iterable containing training data. Each item corresponds to a tuple of string representation
        of target concept, a set of positive examples in the form of URIs amd a set of negative examples in the form of
        URIs, respectively.
        @param relearn_ratio: An integer indicating the number of times dataset is iterated.
        @return: itself
        """
        assert len(dataset) > 0

        counter = 0
        for _ in range(relearn_ratio):  # repeat training over learning problems.
            for (alc_concept_str, positives, negatives) in dataset:
                # self.logger.info(
                #    'Concept:{0}\tE^+:[{1}] \t E^-:[{2}]'.format(alc_concept_str, len(positives), len(negatives)))
                self.rl_learning_loop(pos_uri=positives, neg_uri=negatives)
                self.seen_examples.setdefault(alc_concept_str, dict()).update(
                    {'Positives': list(positives), 'Negatives': list(negatives)})
                counter += 1
                if counter % 100 == 0:
                    self.save_weights()

        return self.terminate_training()
