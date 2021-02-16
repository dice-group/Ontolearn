from functools import total_ordering
from queue import PriorityQueue
from typing import List, Optional, ClassVar, Callable, cast, Final, Iterable

from sortedcontainers import SortedSet

from .abstracts import BaseNode, AbstractTree, AbstractScorer
from .core.owl.utils import OrderedOWLObject
from .owlapy.base import HasIndex
from .owlapy.io import OWLObjectRenderer
from .owlapy.model import OWLClassExpression
from .owlapy.render import DLSyntaxRenderer
from .owlapy.utils import as_index


class Node(BaseNode):
    __slots__ = 'individuals_count'

    renderer: ClassVar[OWLObjectRenderer] = DLSyntaxRenderer()
    individuals_count: Optional[int]

    def __init__(self, concept: OWLClassExpression, parent_node: Optional['Node'] = None, root: bool = False):
        super().__init__(concept, parent_node, root)
        self.individuals_count = None

    def __str__(self):
        return f'Node at {hex(id(self))}\t{Node.renderer.render(self.concept)}\tQuality:{self.quality}\t' \
               f'Heuristic:{self.heuristic}\tDepth:{self.depth}\tH_exp:{self.h_exp}\t' \
               f'|Children|:{self.refinement_count}\t|Indv.|:{self.individuals_count}'


class OENode(Node):
    __slots__ = ()


@total_ordering
class OrderedNode:
    __slots__ = 'node', 'len'

    node: Final[Node]
    len: Final[int]

    def __init__(self, node: Node, length: int):
        self.node = node
        self.len = length

    def __lt__(self, other):
        if type(other) is not type(self):
            return NotImplemented

        if self.len < other.len:
            return True
        elif other.len < self.len:
            return False

        return OrderedOWLObject(cast(HasIndex, self.node.concept)) < OrderedOWLObject(other.node.concept)


def _node_and_all_children(n: Node) -> Iterable[Node]:
    """Get a node and all of its children (recursively) in an iterable"""
    yield n
    for c in n.children:
        yield from _node_and_all_children(c)


class CELOESearchTree(AbstractTree):
    __slots__ = ()

    nodes: 'SortedSet[OENode]'

    def __init__(self, quality_func: AbstractScorer = None, heuristic_func: AbstractScorer = None):
        super().__init__(quality_func, heuristic_func)

        def key(n: OENode):
            if n.heuristic is None:
                raise ValueError("node not evaluated")
            return n.heuristic, OrderedOWLObject(as_index(n.concept))

        self.nodes = SortedSet(key=key)
        self.expression_tests = 0

    # def update_prepare(self, n: Node) -> None:
    #     """Remove n and its children from search tree.
    #
    #     Args:
    #         n: is a node object containing a concept.
    #     """
    #     subtree = list(_node_and_all_children(n))
    #     self.nodes.difference_update(subtree)

    # def update_done(self, n: Node) -> None:
    #     """Add back n and its children into search tree.
    #
    #     Args:
    #         n: is a node object containing a concept.
    #     """
    #     subtree = list(_node_and_all_children(n))
    #     self.nodes.update(subtree)

    def update_prepare(self, n: OENode) -> None:
        """Remove n search tree.

        Args:
            n: is a node object containing a concept.
        """
        self.nodes.remove(n)

    def update_done(self, n: OENode) -> None:
        """Add back n into search tree.

        Args:
            n: is a node object containing a concept.
        """
        self.nodes.add(n)

    def add(self, node: OENode):
        if node not in self.nodes:
            self.quality_func.apply(node)  # AccuracyOrTooWeak(n)
            if node.quality == 0:  # > too weak
                return False
            self.heuristic_func.apply(node, parent_node=parent_node)
            self.nodes[node] = node

            if parent_node:
                parent_node.add_children(node)
            if node.quality == 1:  # goal found
                return True
            return False
        else:
            raise ValueError
            # if not (node.parent_node is parent_node):
            #     try:
            #         assert parent_node.heuristic is not None
            #         assert node.parent_node.heuristic is not None
            #     except AssertionError:
            #         print('REFINED NODE:', parent_node)
            #         print('NODE TO BE ADDED:', node)
            #         print('previous parent of node to be added', node.parent_node)
            #         for k, v in self.nodes.items():
            #             print(k)
            #         raise ValueError()
            #
            #     if parent_node.heuristic > node.parent_node.heuristic:
            #         """Ignore previous parent"""
            #     else:
            #         if node in node.parent_node.children:
            #             node.parent_node.remove_child(node)
            #         node.parent_node = parent_node
            #         self.heuristic_func.apply(node, parent_node=parent_node)
            #         self.nodes[node] = node


class SearchTree(AbstractTree):
    def __init__(self, quality_func: AbstractScorer = None, heuristic_func=None):
        super().__init__(quality_func, heuristic_func)

    def add(self, node: Node, parent_node: Node = None) -> bool:
        """
        Add a node into the search tree.
        Parameters
        ----------
        @param parent_node:
        @param node:
        Returns
        -------
        None
        """

        if parent_node is None:
            self.nodes[node.concept.str] = node
            return False

        if self.redundancy_check(node):
            self.quality_func.apply(node)  # AccuracyOrTooWeak(n)
            if node.quality == 0:  # > too weak
                return False
            self.heuristic_func.apply(node)
            self.nodes[node] = node
            if parent_node:
                parent_node.add_children(node)
            if node.quality == 1:  # goal found
                return True
        else:
            if not (node.parent_node is parent_node):
                if parent_node.heuristic > node.parent_node.heuristic:
                    # update parent info
                    self.heuristic_func.apply(node, parent_node=parent_node)
                    self.nodes[node] = node
                    parent_node.add_children(node)
        return False


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
                parent_node.add_children(node)
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
            parent_node.add_children(node)
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
