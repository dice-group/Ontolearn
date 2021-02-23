import weakref
from _weakref import ReferenceType
from functools import total_ordering
from queue import PriorityQueue
from typing import List, Optional, ClassVar, Final, Iterable, TypeVar, Generic, Set

from owlapy.io import OWLObjectRenderer
from owlapy.model import OWLClassExpression
from owlapy.render import DLSyntaxRenderer
from owlapy.utils import as_index
from .abstracts import AbstractTree, AbstractNode
from .core.owl.utils import OrderedOWLObject

_N = TypeVar('_N')


class OENode(AbstractNode):
    __slots__ = '_concept', '_parent_ref', '_quality', '_heuristic', '_horizontal_expansion', '_len', \
                '_refinement_count', '_individuals_count', '__weakref__'

    renderer: ClassVar[OWLObjectRenderer] = DLSyntaxRenderer()

    _parent_ref: Optional[ReferenceType]  # Optional[ReferenceType[OENode]]
    _quality: Optional[float]
    _heuristic: Optional[float]
    _horizontal_expansion: int
    _len: int
    _refinement_count: int
    _individuals_count: Optional[int]
    _concept: OWLClassExpression

    def __init__(self, concept: OWLClassExpression, length: int, parent_node: Optional['OENode'] = None,
                 is_root: bool = False):
        self._concept = concept
        if is_root:
            self._parent_ref = None
        else:
            self._parent_ref = weakref.ref(parent_node)
        self._quality = None
        self._heuristic = None
        self._horizontal_expansion = length
        self._len = length
        self._refinement_count = 0
        self._individuals_count = None
        super().__init__()

    @property
    def is_root(self) -> bool:
        return self._parent_ref is None

    @property
    def quality(self) -> float:
        return self._quality

    @quality.setter
    def quality(self, v: float):
        if self._quality is not None:
            raise ValueError("Node already evaluated", self)
        self._quality = v

    @property
    def heuristic(self) -> float:
        return self._heuristic

    @heuristic.setter
    def heuristic(self, v: float):
        if v is not None and self._heuristic is not None:
            raise ValueError("Node heuristic already calculated", self)
        self._heuristic = v

    @property
    def h_exp(self) -> int:
        return self._horizontal_expansion

    def increment_h_exp(self):
        self._heuristic = None
        self._horizontal_expansion += 1

    @property
    def len(self) -> int:
        return self._len

    @property
    def parent_node(self) -> Optional['OENode']:
        if self._parent_ref is None:
            return None
        return self._parent_ref()

    @property
    def refinement_count(self) -> int:
        return self._refinement_count

    @refinement_count.setter
    def refinement_count(self, v: int):
        self._heuristic = None
        self._refinement_count = v

    @property
    def concept(self) -> OWLClassExpression:
        return self._concept

    @property
    def individuals_count(self) -> Optional[int]:
        return self._individuals_count

    @individuals_count.setter
    def individuals_count(self, v: int):
        if self._individuals_count is not None:
            raise ValueError("Individuals already counted", self)
        self._individuals_count = v

    def depth(self) -> int:
        d = 0
        n = self
        while True:
            n = n.parent_node
            if not n:
                break
            d += 1
        return d

    def __str__(self):
        addr = hex(id(self))
        addr = addr[0:2] + addr[6:-1]
        return f'OENode at {addr}\t{OENode.renderer.render(self.concept)}\tQuality:{self.quality}\t' \
               f'Heuristic:{self.heuristic}\tDepth:{self.depth()}\tH_exp:{self.h_exp}\t' \
               f'|RC|:{self.refinement_count}\t|Indv.|:{self.individuals_count}'


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


def _node_and_all_children(n: _N) -> Iterable[_N]:
    """Get a node and all of its children (recursively) in an iterable"""
    yield n
    for c in n.children:
        yield from _node_and_all_children(c)


class Node(AbstractNode):
    __slots__ = '_concept', '_len', '_individuals_count'

    renderer: ClassVar[OWLObjectRenderer] = DLSyntaxRenderer()

    _individuals_count: Optional[int]
    _concept: OWLClassExpression
    _len: int

    def __init__(self, concept: OWLClassExpression, length: int):
        self._concept = concept
        self._len = length
        self._individuals_count = None
        super().__init__()

    @property
    def concept(self) -> OWLClassExpression:
        return self._concept

    @property
    def len(self) -> int:
        return self._len

    @property
    def individuals_count(self) -> Optional[int]:
        return self._individuals_count

    @individuals_count.setter
    def individuals_count(self, v: int):
        if self._individuals_count is not None:
            raise ValueError("Individuals already counted", self)
        self._individuals_count = v

    def __str__(self):
        addr = hex(id(self))
        addr = addr[0:2] + addr[6:-1]
        return f'Node at {addr}\t{Node.renderer.render(self.concept)}\t' \
               f'|Indv.|:{self.individuals_count}'


class SearchTreePriorityQueue(AbstractTree):
    """

    Search tree based on priority queue.

    Parameters
    ----------
    quality_func : An instance of a subclass of AbstractScorer that measures the quality of a node.
    heuristic_func : An instance of a subclass of AbstractScorer that measures the promise of a node.

    Attributes
    ----------
    quality_func : An instance of a subclass of AbstractScorer that measures the quality of a node.
    heuristic_func : An instance of a subclass of AbstractScorer that measures the promise of a node.
    items_in_queue: An instance of PriorityQueue Class.
    .nodes: A dictionary where keys are string representation of nodes and values are corresponding node objects.
    nodes: A property method for ._nodes.
    expressionTests: not being used .
    str_to_obj_instance_mapping: not being used.
    """

    def __init__(self, quality_func=None, heuristic_func=None):
        super().__init__(quality_func, heuristic_func)
        self.items_in_queue = PriorityQueue()

    def add(self, n: Node):
        """
        Append a node into the search tree.
        Parameters
        ----------
        n : A Node object
        Returns
        -------
        None
        """
        self.items_in_queue.put((-n.heuristic, n.concept.str))  # gets the smallest one.
        self.nodes[n.concept.str] = n

    def add_node(self, *, node: Node, parent_node: Node):
        """
        Add a node into the search tree after calculating heuristic value given its parent.

        Parameters
        ----------
        node : A Node object
        parent_node : A Node object

        Returns
        -------
        True if node is a "goal node", i.e. quality_metric(node)=1.0
        False if node is a "weak node", i.e. quality_metric(node)=0.0
        None otherwise

        Notes
        -----
        node is a refinement of refined_node
        """
        if node.concept.str in self.nodes and node.parent_node != parent_node:
            old_heuristic = node.heuristic
            self.heuristic_func.apply(node, parent_node=parent_node)
            new_heuristic = node.heuristic
            if new_heuristic > old_heuristic:
                node.parent_node.remove_child(node)
                node.parent_node = parent_node
                parent_node.add_child(node)
                self.items_in_queue.put((-node.heuristic, node.concept.str))  # gets the smallest one.
                self.nodes[node.concept.str] = node
        else:
            # @todos reconsider it.
            self.quality_func.apply(node)
            if node.quality == 0:
                return False
            self.heuristic_func.apply(node, parent_node=parent_node)
            self.items_in_queue.put((-node.heuristic, node.concept.str))  # gets the smallest one.
            self.nodes[node.concept.str] = node
            parent_node.add_child(node)
            if node.quality == 1:
                return True

    def get_most_promising(self) -> Node:
        """
        Gets the current most promising node from Queue.

        Returns
        -------
        node: A node object
        """
        _, most_promising_str = self.items_in_queue.get()  # get
        try:
            node = self.nodes[most_promising_str]
            self.items_in_queue.put((-node.heuristic, node.concept.str))  # put again into queue.
            return node
        except KeyError:
            print(most_promising_str, 'is not found')
            print('####')
            for k, v in self.nodes.items():
                print(k)
            exit(1)

    def get_top_n(self, n: int, key='quality') -> List[Node]:
        """
        Gets the top n nodes determined by key from the search tree.

        Returns
        -------
        top_n_predictions: A list of node objects
        """

        if key == 'quality':
            top_n_predictions = sorted(self.nodes.values(), key=lambda node: node.quality, reverse=True)[:n]
        elif key == 'heuristic':
            top_n_predictions = sorted(self.nodes.values(), key=lambda node: node.heuristic, reverse=True)[:n]
        elif key == 'length':
            top_n_predictions = sorted(self.nodes.values(), key=lambda node: len(node), reverse=True)[:n]
        else:
            print('Wrong Key:{0}\tProgram exist.'.format(key))
            raise KeyError
        return top_n_predictions

    def clean(self):
        self.items_in_queue = PriorityQueue()
        self._nodes.clear()


_TN = TypeVar('_TN', bound='TreeNode')


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
