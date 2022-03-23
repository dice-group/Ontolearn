import logging
from abc import ABCMeta, abstractmethod
from typing import Set, List, Tuple, Iterable, TypeVar, Generic, ClassVar, Optional, Protocol

from owlapy.model import OWLClassExpression, OWLOntology
from owlapy.util import iter_count
from .data_struct import Experience
from .utils import read_csv
from collections import OrderedDict

_N = TypeVar('_N')  #:
_KB = TypeVar('_KB', bound='AbstractKnowledgeBase')  #:

logger = logging.getLogger(__name__)

# @TODO:CD: Each Class definiton in abstract.py should share a prefix, e.g., BaseX or AbstractX.

class EncodedLearningProblem(metaclass=ABCMeta):
    """Encoded Abstract learning problem for use in Scorers"""
    __slots__ = ()

class EncodedPosNegLPStandardKind(EncodedLearningProblem, metaclass=ABCMeta):
    __slots__ = ()

# @TODO: Why we need Generic[_N] and if we need it why we di not use it in all other abstract classes?
class AbstractScorer(Generic[_N], metaclass=ABCMeta):
    """
    An abstract class for quality functions.
    """
    __slots__ = ()

    name: ClassVar[str]

    def __init__(self, *args, **kwargs):
        """Create a new quality function"""
        pass

    def score_elp(self, instances:set, learning_problem: EncodedLearningProblem) -> Tuple[bool, Optional[float]]:
        """Quality score for a set of instances with regard to the learning problem

        Args:
            instances (set): instances to calculate a quality score for
            learning_problem: underlying learning problem to compare the quality to

        Returns:
             Tuple, first position indicating if the function could be applied, second position the quality value
                in the range 0.0--1.0
        """
        if len(instances) == 0:
            return False, 0

        from ontolearn.learning_problem import EncodedPosNegLPStandard
        if isinstance(learning_problem, EncodedPosNegLPStandard):
            tp = len(learning_problem.kb_pos.intersection(instances))
            tn = len(learning_problem.kb_neg.difference(instances))

            fp = len(learning_problem.kb_neg.intersection(instances))
            fn = len(learning_problem.kb_pos.difference(instances))

            return self.score2(tp=tp, tn=tn, fp=fp, fn=fn)
        else:
            raise NotImplementedError(learning_problem)

    @abstractmethod
    def score2(self, tp: int, fn: int, fp: int, tn: int) -> Tuple[bool, Optional[float]]:
        """Quality score for a coverage count

        Args:
            tp: true positive count
            fn: false negative count
            fp: false positive count
            tn: true negative count

        Returns:
             Tuple, first position indicating if the function could be applied, second position the quality value
                in the range 0.0--1.0
        """
        pass

    # @TODO:CD: Why there is '..' in AbstractNode
    def apply(self, node: 'AbstractNode', instances, learning_problem: EncodedLearningProblem) -> bool:
        """Apply the quality function to a search tree node after calculating the quality score on the given instances

        Args:
            node: search tree node to set the quality on
            instances (set): instances to calculate the quality for
            learning_problem: underlying learning problem to compare the quality to

        Returns:
            True if the quality function was applied successfully
        """
        assert isinstance(learning_problem, EncodedLearningProblem), \
            f'Expected EncodedLearningProblem but got {type(learning_problem)}'
        assert isinstance(node, AbstractNode), \
            f'Expected AbstractNode but got {type(node)}'
        from ontolearn.search import _NodeQuality
        assert isinstance(node, _NodeQuality), \
            f'Expected _NodeQuality but got {type(_NodeQuality)}'

        ret, q = self.score_elp(instances, learning_problem)
        if q is not None:
            node.quality = q
        return ret

class AbstractHeuristic(Generic[_N], metaclass=ABCMeta):
    """Abstract base class for heuristic functions.

    Heuristic functions can guide the search process."""
    __slots__ = ()

    @abstractmethod
    def __init__(self):
        """Create a new heuristic function"""
        pass

    @abstractmethod
    def apply(self, node: _N, instances, learning_problem: EncodedLearningProblem):
        """Apply the heuristic on a search tree node and set its heuristic property to the calculated value

        Args:
            node: node to set the heuristic on
            instances (set, optional): set of instances covered by this node
            learning_problem: underlying learning problem to compare the heuristic to
        """
        pass

class AbstractFitness(metaclass=ABCMeta):
    """Abstract base class for fitness functions.

    Fitness functions guide the evolutionary process."""
    __slots__ = ()

    name: ClassVar[str]

    @abstractmethod
    def __init__(self):
        """Create a new fitness function"""
        pass

    @abstractmethod
    def apply(self, individual):
        """Apply the fitness function on an individual and set its fitness attribute to the calculated value

        Args:
            individual: individual to set the fitness on
        """
        pass


class BaseRefinement(Generic[_N], metaclass=ABCMeta):
    """
    Base class for Refinement Operators.

    Let C, D \\in N_c where N_c os a finite set of concepts.

    * Proposition 3.3 (Complete and Finite Refinement Operators) [1]
      * ρ(C) = {C ⊓ T} ∪ {D \\| D is not empty AND D \\sqset C}
        * The operator is finite,
        * The operator is complete as given a concept C, we can reach an arbitrary concept D such that D subset of C.

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
        """Construct a new base refinement operator

        Args:
            knowledge_base: knowledge base to operate on
        """
        self.kb = knowledge_base

    @abstractmethod
    def refine(self, *args, **kwargs) -> Iterable[OWLClassExpression]:
        """Refine a given concept

        Args:
            ce (OWLClassExpression): concept to refine

        Returns:
            new refined concepts
        """
        pass

    def len(self, concept: OWLClassExpression) -> int:
        """The length of a concept

        Args:
            concept: concept

        Returns:
            length of concept according to some metric configured in the knowledge base
        """
        return self.kb.concept_len(concept)


class AbstractNode(metaclass=ABCMeta):
    """Abstract search tree node"""
    __slots__ = ()

    @abstractmethod
    def __init__(self):
        """Create an abstract search tree node"""
        pass

    def __str__(self):
        """string representation of node, by default its internal memory address"""
        addr = hex(id(self))
        addr = addr[0:2] + addr[6:-1]
        return f'{type(self)} at {addr}'


class AbstractOEHeuristicNode(metaclass=ABCMeta):
    """Abstract Node for the CELOEHeuristic heuristic function

    This node must support quality, horizontal expansion (h_exp), is_root, parent_node and refinement_count
    """
    __slots__ = ()

    @property
    @abstractmethod
    def quality(self) -> Optional[float]:
        pass

    @property
    @abstractmethod
    def h_exp(self) -> int:
        pass

    @property
    @abstractmethod
    def is_root(self) -> bool:
        pass

    @property
    @abstractmethod
    def parent_node(self: _N) -> Optional[_N]:
        pass

    @property
    @abstractmethod
    def refinement_count(self) -> int:
        pass

    @property
    @abstractmethod
    def heuristic(self) -> Optional[float]:
        pass

    @heuristic.setter
    @abstractmethod
    def heuristic(self, v: float):
        pass


class AbstractConceptNode(metaclass=ABCMeta):
    """Abstract search tree node which has a concept"""
    __slots__ = ()

    @property
    @abstractmethod
    def concept(self) -> OWLClassExpression:
        pass


class AbstractKnowledgeBase(metaclass=ABCMeta):
    """Abstract knowledge base"""
    __slots__ = ()

    thing: OWLClassExpression

    @abstractmethod
    def ontology(self) -> OWLOntology:
        """The base ontology of this knowledge base"""
        pass

    def describe(self) -> None:
        """Print a short description of the Knowledge Base to the info logger output"""
        properties_count = iter_count(self.ontology().object_properties_in_signature()) + iter_count(
            self.ontology().data_properties_in_signature())
        logger.info(f'Number of named classes: {iter_count(self.ontology().classes_in_signature())}\n'
                    f'Number of individuals: {self.individuals_count()}\n'
                    f'Number of properties: {properties_count}')

    @abstractmethod
    def clean(self) -> None:
        """This method should reset any caches and statistics in the knowledge base"""
        raise NotImplementedError

    @abstractmethod
    def individuals_count(self) -> int:
        """Total number of individuals in this knowledge base"""
        pass

    @abstractmethod
    def individuals_set(self, *args, **kwargs) -> Set:
        """Encode an individual, an iterable of individuals or the individuals that are instances of a given concept
        into a set.

        Args:
            arg (OWLNamedIndividual): individual to encode
            arg (Iterable[OWLNamedIndividual]): individuals to encode
            arg (OWLClassExpression): encode individuals that are instances of this concept

        Returns:
            encoded set representation of individual(s)
        """
        pass

    @abstractmethod
    def concept_len(self, ce: OWLClassExpression) -> int:
        """Calculate the length of a concept

        Args:
            ce: concept

        Returns:
            length of the concept
        """
        pass


class AbstractLearningProblem(metaclass=ABCMeta):
    """Abstract learning problem"""
    __slots__ = ()

    @abstractmethod
    def __init__(self, *args, **kwargs):
        """create a new abstract learning problem"""
        pass

    @abstractmethod
    def encode_kb(self, knowledge_base: AbstractKnowledgeBase) -> 'EncodedLearningProblem':
        """encode the learning problem into the knowledge base"""
        pass


class LBLSearchTree(Generic[_N], metaclass=ABCMeta):
    """Abstract search tree for the Length based learner"""

    @abstractmethod
    def get_most_promising(self) -> _N:
        """Find most "promising" node in the search tree that should be refined next

        Returns:
            most promising search tree node
        """
        pass

    @abstractmethod
    def add_node(self, node: _N, parent_node: _N, kb_learning_problem: EncodedLearningProblem):
        """Add a node to the search tree

        Args:
            node: node to add
            parent_node: parent of that node
            kb_learning_problem: underlying learning problem to compare the quality to
        """
        pass

    @abstractmethod
    def clean(self):
        """Reset the search tree state"""
        pass

    @abstractmethod
    def get_top_n(self, n: int) -> List[_N]:
        """Retrieve the best n search tree nodes

        Args:
            n: maximum number of nodes

        Returns:
            list of top n search tree nodes
        """
        pass

    @abstractmethod
    def show_search_tree(self, root_concept: OWLClassExpression, heading_step: str):
        """Debugging function to print the search tree to standard output

        Args:
            root_concept: the tree is printed starting from this search tree node
            heading_step: message to print at top of the output
        """
        pass

    @abstractmethod
    def add_root(self, node: _N, kb_learning_problem: EncodedLearningProblem):
        """Add the root node to the search tree

        Args:
            node: root node to add
            kb_learning_problem: underlying learning problem to compare the quality to
        """
        pass


class AbstractDrill:
    """
    Abstract class for Convolutional DQL concept learning
    """

    def __init__(self, path_of_embeddings, reward_func, learning_rate=None,
                 num_episode=None, num_episodes_per_replay=None, epsilon=None,
                 num_of_sequential_actions=None, max_len_replay_memory=None,
                 representation_mode=None, batch_size=None, epsilon_decay=None, epsilon_min=None,
                 num_epochs_per_replay=None, num_workers=None, verbose=0):
        self.name = 'DRILL'
        self.instance_embeddings = read_csv(path_of_embeddings)
        self.embedding_dim = self.instance_embeddings.shape[1]
        self.reward_func = reward_func
        self.representation_mode = representation_mode
        assert representation_mode in ['averaging', 'sampling']
        # Will be filled by child class
        self.heuristic_func = None
        self.num_workers = num_workers
        # constants
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.num_episode = num_episode
        self.num_of_sequential_actions = num_of_sequential_actions
        self.num_epochs_per_replay = num_epochs_per_replay
        self.max_len_replay_memory = max_len_replay_memory
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.verbose = verbose
        self.num_episodes_per_replay = num_episodes_per_replay

        # will be filled
        self.optimizer = None  # torch.optim.Adam(self.model_net.parameters(), lr=self.learning_rate)

        self.seen_examples = dict()
        self.emb_pos, self.emb_neg = None, None
        self.start_time = None
        self.goal_found = False
        self.experiences = Experience(maxlen=self.max_len_replay_memory)

    def attributes_sanity_checking_rl(self):
        assert len(self.instance_embeddings) > 0
        assert self.embedding_dim > 0
        if self.num_workers is None:
            self.num_workers = 4
        if self.epsilon is None:
            self.epsilon = 1
        if self.learning_rate is None:
            self.learning_rate = .001
        if self.num_episode is None:
            self.num_episode = 1
        if self.num_of_sequential_actions is None:
            self.num_of_sequential_actions = 3
        if self.num_epochs_per_replay is None:
            self.num_epochs_per_replay = 1
        if self.max_len_replay_memory is None:
            self.max_len_replay_memory = 256
        if self.epsilon_decay is None:
            self.epsilon_decay = 0.01
        if self.epsilon_min is None:
            self.epsilon_min = 0
        if self.batch_size is None:
            self.batch_size = 1024
        if self.verbose is None:
            self.verbose = 0
        if self.num_episodes_per_replay is None:
            self.num_episodes_per_replay = 2

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


class DRILLAbstractTree:
    @abstractmethod
    def __init__(self):
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

    def best_hypotheses(self, n=10) -> List:
        assert self.search_tree is not None
        assert len(self.search_tree) > 1
        return [i for i in self.search_tree.get_top_n_nodes(n)]

    def show_search_tree(self, th, top_n=10):
        """
        Show search tree.
        """
        print(f'######## {th}.step\t Top 10 nodes in Search Tree \t |Search Tree|={self.__len__()} ###########')
        predictions = list(self.get_top_n_nodes(top_n))
        for ith, node in enumerate(predictions):
            print(f'{ith + 1}-\t{node}')
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
        self._nodes.clear()
