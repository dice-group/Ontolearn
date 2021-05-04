import weakref
from _weakref import ReferenceType
from abc import abstractmethod, ABCMeta
from functools import total_ordering
from queue import PriorityQueue
from typing import List, Optional, ClassVar, Final, Iterable, TypeVar, Generic, Set, Tuple, Dict

from owlapy.io import OWLObjectRenderer
from owlapy.model import OWLClassExpression
from owlapy.render import DLSyntaxObjectRenderer
from owlapy.util import as_index, OrderedOWLObject
from superprop import super_prop
from .abstracts import AbstractNode, AbstractHeuristic, AbstractScorer, AbstractOEHeuristicNode, LBLSearchTree, \
    AbstractConceptNode

_N = TypeVar('_N')  #:


# Due to a bug in Python, we cannot use the slots like we should be able to. Hence, the attribute access is also
# invalid but there is nothing we can do. See https://mail.python.org/pipermail/python-list/2002-December/126637.html

# noinspection PyUnresolvedReferences
# noinspection PyDunderSlots
class _NodeConcept(metaclass=ABCMeta):
    __slots__ = ()

    renderer: ClassVar[OWLObjectRenderer] = DLSyntaxObjectRenderer()

    _concept: OWLClassExpression

    @abstractmethod
    def __init__(self, concept: OWLClassExpression):
        self._concept = concept

    @property
    def concept(self) -> OWLClassExpression:
        return self._concept

    @abstractmethod
    def __str__(self):
        return _NodeConcept.renderer.render(self.concept)


# noinspection PyUnresolvedReferences
# noinspection PyDunderSlots
class _NodeLen(metaclass=ABCMeta):
    __slots__ = ()

    _len: int

    @abstractmethod
    def __init__(self, length: int):
        self._len = length

    @property
    def len(self) -> int:
        return self._len


# noinspection PyUnresolvedReferences
# noinspection PyDunderSlots
class _NodeIndividualsCount(metaclass=ABCMeta):
    __slots__ = ()

    _individuals_count: Optional[int]

    @abstractmethod
    def __init__(self, individuals_count: Optional[int] = None):
        self._individuals_count = individuals_count

    @property
    def individuals_count(self) -> Optional[int]:
        return self._individuals_count

    @individuals_count.setter
    def individuals_count(self, v: int):
        if self._individuals_count is not None:
            raise ValueError("Individuals already counted", self)
        self._individuals_count = v

    @abstractmethod
    def __str__(self):
        return f'|Indv.|:{self.individuals_count}'


# noinspection PyUnresolvedReferences
# noinspection PyDunderSlots
class _NodeHeuristic(metaclass=ABCMeta):
    __slots__ = ()

    _heuristic: Optional[float]

    @abstractmethod
    def __init__(self, heuristic: Optional[float] = None):
        self._heuristic = heuristic

    @property
    def heuristic(self) -> float:
        return self._heuristic

    @heuristic.setter
    def heuristic(self, v: float):
        if v is not None and self._heuristic is not None:
            raise ValueError("Node heuristic already calculated", self)
        self._heuristic = v

    @abstractmethod
    def __str__(self):
        return f'Heuristic:{self.heuristic}'


class _NodeParentRef(Generic[_N], metaclass=ABCMeta):
    __slots__ = ()

    _parent_ref: Optional[ReferenceType]  # Optional[ReferenceType[OENode]]

    @abstractmethod
    def __init__(self, parent_node: Optional[_N] = None, is_root: bool = False):
        if is_root:
            self._parent_ref = None
        else:
            self._parent_ref = weakref.ref(parent_node)

    @property
    def is_root(self) -> bool:
        return self._parent_ref is None

    @property
    def parent_node(self) -> Optional[_N]:
        if self._parent_ref is None:
            return None
        return self._parent_ref()

    def depth(self) -> int:
        d = 0
        n = self
        while True:
            n = n.parent_node
            if not n:
                break
            d += 1
        return d

    @abstractmethod
    def __str__(self):
        return f'Depth:{self.depth()}'


# noinspection PyUnresolvedReferences
# noinspection PyDunderSlots
class _NodeQuality(metaclass=ABCMeta):
    __slots__ = ()

    _quality: Optional[float]

    @abstractmethod
    def __init__(self, quality: Optional[float] = None):
        self._quality = quality

    @property
    def quality(self) -> float:
        return self._quality

    @quality.setter
    def quality(self, v: float):
        if self._quality is not None:
            raise ValueError("Node already evaluated", self)
        self._quality = v

    @abstractmethod
    def __str__(self):
        return f'Quality:{self.quality}'


class Node(_NodeConcept, _NodeLen, _NodeIndividualsCount, AbstractNode):
    __slots__ = '_concept', '_len', '_individuals_count'

    def __init__(self, concept: OWLClassExpression, length: int):
        _NodeConcept.__init__(self, concept)
        _NodeLen.__init__(self, length)
        _NodeIndividualsCount.__init__(self)
        AbstractNode.__init__(self)

    def __str__(self):
        return "\t".join((
            AbstractNode.__str__(self),
            _NodeConcept.__str__(self),
            _NodeIndividualsCount.__str__(self),
        ))


class OENode(_NodeConcept, _NodeLen, _NodeIndividualsCount, _NodeQuality, _NodeHeuristic,
             _NodeParentRef['OENode'], AbstractNode, AbstractConceptNode, AbstractOEHeuristicNode):
    __slots__ = '_concept', '_len', '_individuals_count', '_quality', '_heuristic', \
                '_parent_ref', '_horizontal_expansion', \
                '_refinement_count', '__weakref__'

    renderer: ClassVar[OWLObjectRenderer] = DLSyntaxObjectRenderer()

    _horizontal_expansion: int
    _refinement_count: int

    def __init__(self, concept: OWLClassExpression, length: int, parent_node: Optional['OENode'] = None,
                 is_root: bool = False):
        _NodeConcept.__init__(self, concept)
        _NodeLen.__init__(self, length)
        _NodeIndividualsCount.__init__(self)
        _NodeQuality.__init__(self)
        _NodeHeuristic.__init__(self)
        _NodeParentRef.__init__(self, parent_node, is_root)
        self._horizontal_expansion = length
        self._refinement_count = 0
        AbstractNode.__init__(self)

    @property
    def h_exp(self) -> int:
        return self._horizontal_expansion

    def increment_h_exp(self):
        self._heuristic = None
        self._horizontal_expansion += 1

    @property
    def refinement_count(self) -> int:
        return self._refinement_count

    @refinement_count.setter
    def refinement_count(self, v: int):
        self._heuristic = None
        self._refinement_count = v

    def __str__(self):
        return "\t".join((
            AbstractNode.__str__(self),
            _NodeConcept.__str__(self),
            _NodeQuality.__str__(self),
            _NodeHeuristic.__str__(self),
            _NodeParentRef.__str__(self),
            f'H_exp:{self.h_exp}',
            f'|RC|:{self.refinement_count}',
            _NodeIndividualsCount.__str__(self),
        ))


# noinspection PyUnresolvedReferences
# noinspection PyDunderSlots
class _NodeIndividuals(metaclass=ABCMeta):
    __slots__ = ()

    _individuals: Optional[set]

    @abstractmethod
    def __init__(self, individuals: Optional[set] = None):
        self._individuals = individuals

    @property
    def individuals(self):
        return self._individuals

    @property
    def individuals_count(self) -> Optional[int]:
        return len(self._individuals)


class LBLNode(_NodeIndividuals, OENode):
    __slots__ = '_children', '_individuals'

    def __init__(self, concept: OWLClassExpression, length: int, individuals, parent_node: Optional['LBLNode'] = None,
                 is_root: bool = False):
        OENode.__init__(self, concept=concept, length=length, parent_node=parent_node, is_root=is_root)
        _NodeIndividuals.__init__(self, individuals)
        self._children = set()

    def add_child(self, n):
        self._children.add(n)

    def remove_child(self, n):
        self._children.remove(n)

    @property
    def children(self):
        return self._children

    @property
    def parent_node(self) -> Optional['LBLNode']:
        return super_prop(super()).parent_node

    @parent_node.setter
    def parent_node(self, parent_node: Optional['LBLNode']):
        self._parent_ref = weakref.ref(parent_node)


@total_ordering
class LengthOrderedNode(Generic[_N]):
    __slots__ = 'node', 'len'

    node: Final[_N]
    len: Final[int]

    def __init__(self, node: _N, length: int):
        self.node = node
        self.len = length

    def __lt__(self, other):
        if type(other) is not type(self):
            return NotImplemented

        if self.len < other.len:
            return True
        elif other.len < self.len:
            return False
        else:
            return OrderedOWLObject(as_index(self.node.concept)) < OrderedOWLObject(as_index(other.node.concept))

    def __eq__(self, other):
        return self.len == other.len and self.node == other.node


@total_ordering
class HeuristicOrderedNode(Generic[_N]):
    """A comparator that orders the Nodes based on Heuristic, then OrderedOWLObject of the concept"""
    __slots__ = 'node'

    node: Final[_N]

    def __init__(self, node: _N):
        self.node = node

    def __lt__(self: _N, other: _N):
        if self.node.heuristic is None:
            raise ValueError("node heuristic not calculated", self.node)
        if other.node.heuristic is None:
            raise ValueError("other node heuristic not calculcated", other.node)

        if self.node.heuristic < other.node.heuristic:
            return True
        elif self.node.heuristic > other.node.heuristic:
            return False
        else:
            return OrderedOWLObject(as_index(self.node.concept)) < OrderedOWLObject(as_index(other.node.concept))

    def __eq__(self: _N, other: _N):
        return self.node == other.node


@total_ordering
class QualityOrderedNode:
    __slots__ = 'node'

    node: Final[OENode]

    def __init__(self, node: OENode):
        self.node = node

    def __lt__(self, other):
        if self.node.quality is None:
            raise ValueError("node not evaluated", self.node)
        if other.node.quality is None:
            raise ValueError("other node not evaluated", other.node)

        if self.node.quality < other.node.quality:
            return True
        elif self.node.quality > other.node.quality:
            return False
        else:
            if self.node.len > other.node.len:  # shorter is better, ie. greater
                return True
            elif self.node.len < other.node.len:
                return False
            else:
                return OrderedOWLObject(as_index(self.node.concept)) < OrderedOWLObject(as_index(other.node.concept))

    def __eq__(self, other):
        return self.node == other.node


def _node_and_all_children(n: _N) -> Iterable[_N]:
    """Get a node and all of its children (recursively) in an iterable"""
    yield n
    for c in n.children:
        yield from _node_and_all_children(c)


class SearchTreePriorityQueue(LBLSearchTree[LBLNode]):
    """

    Search tree based on priority queue.

    Args:
        quality_func: An instance of a subclass of AbstractScorer that measures the quality of a node.
        heuristic_func: An instance of a subclass of AbstractScorer that measures the promise of a node.

    Attributes:
        quality_func: An instance of a subclass of AbstractScorer that measures the quality of a node.
        heuristic_func: An instance of a subclass of AbstractScorer that measures the promise of a node.
        items_in_queue: An instance of PriorityQueue Class.
        .nodes: A dictionary where keys are string representation of nodes and values are corresponding node objects.
        nodes: A property method for ._nodes.
        expressionTests: not being used .
        str_to_obj_instance_mapping: not being used.
    """

    quality_func: AbstractScorer
    heuristic_func: AbstractHeuristic
    nodes: Dict[OWLClassExpression, LBLNode]
    items_in_queue: 'PriorityQueue[Tuple[float, HeuristicOrderedNode[LBLNode]]]'

    def __init__(self, quality_func, heuristic_func):
        self.quality_func = quality_func
        self.heuristic_func = heuristic_func
        self.nodes = dict()
        self.items_in_queue = PriorityQueue()

    def add(self, n: LBLNode):
        """
        Append a node into the search tree.

        Args:
            n: A Node object

        Returns:
            None
        """
        self.items_in_queue.put((-n.heuristic, HeuristicOrderedNode(n)))  # gets the smallest one.
        self.nodes[n.concept] = n

    def add_root(self, *, node: LBLNode):
        assert node.is_root
        assert not self.nodes
        self.quality_func.apply(node, node.individuals)
        self.heuristic_func.apply(node)
        self.items_in_queue.put((-node.heuristic, HeuristicOrderedNode(node)))  # gets the smallest one.
        self.nodes[node.concept] = node

    def add_node(self, *, node: LBLNode, parent_node: LBLNode) -> Optional[bool]:
        """
        Add a node into the search tree after calculating heuristic value given its parent.

        Args:
            node: A Node object
            parent_node: A Node object

        Returns:
            True if node is a "goal node", i.e. quality_metric(node)=1.0
            False if node is a "weak node", i.e. quality_metric(node)=0.0
            None otherwise

        Notes:
            node is a refinement of refined_node
        """
        if node.concept in self.nodes and node.parent_node != parent_node:
            old_heuristic = node.heuristic
            self.heuristic_func.apply(node)
            new_heuristic = node.heuristic
            if new_heuristic > old_heuristic:
                node.parent_node.remove_child(node)
                node.parent_node = parent_node
                parent_node.add_child(node)
                self.items_in_queue.put((-node.heuristic, HeuristicOrderedNode(node)))  # gets the smallest one.
                self.nodes[node.concept] = node
        else:
            # @todos reconsider it.
            self.quality_func.apply(node, node.individuals)
            if node.quality == 0:
                return False
            self.heuristic_func.apply(node)
            self.items_in_queue.put((-node.heuristic, HeuristicOrderedNode(node)))  # gets the smallest one.
            self.nodes[node.concept] = node
            parent_node.add_child(node)
            if node.quality == 1:
                return True

    def get_most_promising(self) -> LBLNode:
        """
        Gets the current most promising node from Queue.

        Returns:
            node: A node object
        """
        _, most_promising_str = self.items_in_queue.get()  # get
        try:
            node = self.nodes[most_promising_str.node.concept]
            self.items_in_queue.put((-node.heuristic, HeuristicOrderedNode(node)))  # put again into queue.
            return node
        except KeyError:
            print(most_promising_str, 'is not found')
            print('####')
            for k, v in self.nodes.items():
                print(k)
            raise

    def get_top_n(self, n: int, key='quality') -> List[LBLNode]:
        """
        Gets the top n nodes determined by key from the search tree.

        Returns:
            top_n_predictions: A list of node objects
        """

        if key == 'quality':
            top_n_predictions = sorted(self.nodes.values(), key=lambda node: node.quality, reverse=True)[:n]
        elif key == 'heuristic':
            top_n_predictions = sorted(self.nodes.values(), key=lambda node: node.heuristic, reverse=True)[:n]
        elif key == 'length':
            top_n_predictions = sorted(self.nodes.values(), key=lambda node: node.len, reverse=True)[:n]
        else:
            print('Wrong Key:{0}\tProgram exist.'.format(key))
            raise KeyError
        return top_n_predictions

    def clean(self):
        self.items_in_queue = PriorityQueue()
        self.nodes.clear()

    def show_search_tree(self, root_concept: OWLClassExpression, heading_step: str):
        rdr = DLSyntaxObjectRenderer()

        print('######## ', heading_step, 'step Search Tree ###########')

        def node_as_length_ordered_concept(node: LBLNode):
            return LengthOrderedNode(node, node.len)

        def print_partial_tree_recursive(node: LBLNode, depth: int = 0):
            render_str = rdr.render(node.concept)

            depths = "`" * depth

            print("%s %s \t Q:%f Heur:%s" % (depths, render_str, node.quality, node.heuristic))

            for c in sorted(node.children, key=node_as_length_ordered_concept):
                print_partial_tree_recursive(c, depth + 1)

        print_partial_tree_recursive(self.nodes[root_concept])


_TN = TypeVar('_TN', bound='TreeNode')  #:


class TreeNode(Generic[_N]):
    __slots__ = 'children', 'node'

    node: Final[_N]
    children: Set['TreeNode[_N]']

    def __init__(self: _TN, node: _N, parent_tree_node: Optional[_TN] = None, is_root: bool = False):
        self.node = node
        self.children = set()
        if not is_root:
            assert isinstance(parent_tree_node, TreeNode)
            parent_tree_node.children.add(self)
