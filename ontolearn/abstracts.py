import random
import time
import weakref
from _weakref import ReferenceType
from abc import ABCMeta, abstractmethod, ABC
from collections import OrderedDict
from typing import Set, List, Tuple, Iterable, TypeVar, Optional, Type, Generic, ClassVar, cast

import numpy as np
import pandas as pd
import torch

from .core.owl.utils import OWLClassExpressionLengthMetric
from .core.owl.hierarchy import ClassHierarchy
from .data_struct import PrepareBatchOfTraining, PrepareBatchOfPrediction
from .owlapy.model import OWLClassExpression, OWLOntology
from .owlapy.utils import iter_count
from .utils import balanced_sets, read_csv
from .owlready2.utils import get_full_iri
from typing import Set, Dict, List, Tuple, Iterable, Generator, SupportsFloat, TypeVar, Optional, Type, Generic
import random
from .data_struct import PrepareBatchOfTraining, PrepareBatchOfPrediction
import json
import time

# random.seed(0)  # Note: a module should not set the seed

_N = TypeVar('_N')


# class AbstractNode(metaclass=ABCMeta):
#     """Base class for Concept."""
#     __slots__ = 'concept', '__heuristic_score', '__quality_score', \
#                 '__refinement_count', '__depth', '__children', '__embeddings', \
#                 '__parent_node_ref', '__is_root', '__weakref__'
#
#     concept: OWLClassExpression
#     __parent_node_ref: Optional[ReferenceType]
#     __quality_score: Optional[float]
#     __heuristic_score: Optional[float]
#     __is_root: bool
#     __refinement_count: int
#     __children: Set[_N]
#
#     @abstractmethod
#     def __init__(self: _N, concept: OWLClassExpression, parent_node: Optional[_N] = None, is_root: bool = False):
#         self.__quality_score = None
#         self.__heuristic_score = None
#         self.__is_root = is_root
#         self.__refinement_count = 0
#         self.concept = concept
#         self.__embeddings = None
#         self.__children = set()
#         self.__parent_node_ref = None
#         self.parent_node = parent_node
#
#     @abstractmethod
#     def __eq__(self, other):
#         raise NotImplementedError
#
#     @abstractmethod
#     def __hash__(self):
#         raise NotImplementedError
#
#     @property
#     def parent_node(self) -> _N:
#         return cast(type(self), self.__parent_node_ref()) if self.__parent_node_ref is not None else None
#
#     @parent_node.setter
#     def parent_node(self, parent_node: _N):
#         self.__parent_node_ref = weakref.ref(parent_node) if parent_node is not None else None
#         if parent_node is None:
#             assert self.__is_root
#             self.__depth = 0
#         else:
#             assert not self.__is_root
#             self.__depth = parent_node.depth + 1
#             parent_node.add_child(self)
#
#     @property
#     def embeddings(self):
#         return self.__embeddings
#
#     @embeddings.setter
#     def embeddings(self, value):
#         self.__embeddings = value
#
#     @property
#     def children(self) -> Set[_N]:
#         return self.__children
#
#     @property
#     def refinement_count(self) -> int:
#         return self.__refinement_count
#
#     @refinement_count.setter
#     def refinement_count(self, n: int):
#         self.__refinement_count = n
#
#     @property
#     def depth(self) -> int:
#         return self.__depth
#
#     @depth.setter
#     def depth(self, n: int):
#         self.__depth = n
#
#     @property
#     def heuristic(self) -> Optional[float]:
#         return self.__heuristic_score
#
#     @heuristic.setter
#     def heuristic(self, val: Optional[float]):
#         if val is not None and self.__heuristic_score is not None:
#             raise ValueError("Node heuristic already calculated", self)
#         self.__heuristic_score = val
#
#     @property
#     def quality(self) -> float:
#         return self.__quality_score
#
#     @quality.setter
#     def quality(self, val: float):
#         self.__quality_score = val
#
#     @property
#     def is_root(self) -> bool:
#         return self.__is_root
#
#     def add_child(self: _N, n: _N) -> None:
#         assert type(self) is type(n)
#         print(n)
#         self.__children.add(n)
#
#     def remove_child(self: _N, n: _N) -> None:
#         assert type(self) is type(n)
#         self.__children.remove(n)


class AbstractScorer(Generic[_N], metaclass=ABCMeta):
    """
    An abstract class for quality and heuristic functions.
    """
    __slots__ = 'pos', 'neg', 'unlabelled', 'applied'

    name: ClassVar

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
    def apply(self, node: _N, individuals):
        pass

    def clean(self):
        self.pos = None
        self.neg = None
        self.unlabelled = None
        self.applied = 0


class AbstractHeuristic(Generic[_N], metaclass=ABCMeta):
    __slots__ = 'applied'

    applied: int

    @abstractmethod
    def apply(self, node: _N):
        pass

    @abstractmethod
    def clean(self):
        self.applied = 0


_KB = TypeVar('_KB', bound='AbstractKnowledgeBase')


class BaseRefinement(Generic[_N], metaclass=ABCMeta):
    """
    Base class for Refinement Operators.

    Let C, D \\in N_c where N_c os a finite set of concepts.

    * Proposition 3.3 (Complete and Finite Refinement Operators) [1]
        ** ρ(C) = {C ⊓ T} ∪ {D | D is not empty AND D \\sqset C}
        *** The operator is finite,
        *** The operator is complete as given a concept C, we can reach an arbitrary concept D such that D subset of C.

    *) Theoretical Foundations of Refinement Operators [1].




    *) Defining a top-down refimenent operator that is a proper is crutial.
        4.1.3 Achieving Properness [1]
    *) Figure 4.1 [1] defines of the refinement operator

    [1] Learning OWL Class Expressions
    """
    __slots__ = 'kb'

    kb: _KB

    @abstractmethod
    def __init__(self, knowledge_base: _KB):
        self.kb = knowledge_base

    @abstractmethod
    def refine(self, *args, **kwargs) -> Iterable[OWLClassExpression]:
        """Refine a given concept
        """
        pass

    def len(self, concept: OWLClassExpression) -> int:
        """The length of a concept

        Args:
            concept: concept

        Returns:
            length of concept according to some metric
        """
        return self.kb.cl(concept)


class AbstractTree(Generic[_N], metaclass=ABCMeta):
    __slots__ = ()

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def clean(self):
        pass

    @abstractmethod
    def add(self, node: _N):
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass


class AbstractNode(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self):
        raise NotImplementedError


class AbstractKnowledgeBase(metaclass=ABCMeta):
    __slots__ = ()

    thing: OWLClassExpression

    @abstractmethod
    def save(self, path: str, rdf_format="rdfxml"):
        pass

    @abstractmethod
    def ontology(self) -> OWLOntology:
        """The base ontology of this knowledge base"""
        pass

    def describe(self) -> None:
        """Print a short description of the Knowledge Base to standard output"""
        properties_count = iter_count(self.ontology().object_properties_in_signature()) + iter_count(
            self.ontology().data_properties_in_signature())
        print(f'Number of named classes: {iter_count(self.ontology().classes_in_signature())}\n'
              f'Number of individuals: {self.individuals_count()}\n'
              f'Number of properties: {properties_count}')

    @abstractmethod
    def clean(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def individuals_count(self) -> int:
        """Total number of individuals in this knowledge base"""
        pass

    @abstractmethod
    def individuals_set(self, *args, **kwargs) -> Set:
        pass


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

    def next_node_to_expand(self, t: int = None) -> AbstractNode:
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

    def sequence_of_actions(self, root: AbstractNode) -> Tuple[List[Tuple[AbstractNode, AbstractNode]], List[SupportsFloat]]:
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
        assert isinstance(root, AbstractNode)

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

    def apply_rho(self, node: AbstractNode) -> Generator:
        """
        Refine an OWL Class expression |= Observing next possible states

        Computation O(N).

        1. Generate concepts by refining a node
            1.1 Compute allowed length of refinements
            1.2. Convert concepts if concepts do not belong to  self.concepts_to_ignore
                Note that          i.str not in self.concepts_to_ignore => O(1) if a set is being used.
        3. Return Generator
        """
        assert isinstance(node, AbstractNode)
        # 1.
        # (1.1)
        length = len(node) + 3 if len(node) + 3 <= self.max_child_length else self.max_child_length
        # (1.2)
        for i in self.operator.refine(node, maxlength=length):  # O(N)
            if i.str not in self.concepts_to_ignore:  # O(1)
                yield self.operator.get_node(i, parent_node=node)  # O(1)

    def assign_embeddings(self, node: AbstractNode) -> None:
        assert isinstance(node, AbstractNode)
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
        root = self.operator.get_node(self.start_class, root=True)
        # (2) Assign embeddings of root/first state.
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

    def exploration_exploitation_tradeoff(self, current_state: AbstractNode, next_states: List[AbstractNode]) -> AbstractNode:
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

    def exploitation(self, current_state: AbstractNode, next_states: List[AbstractNode]) -> AbstractNode:
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

    def predict_Q(self, current_state: AbstractNode, next_states: List[AbstractNode]) -> torch.Tensor:
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
