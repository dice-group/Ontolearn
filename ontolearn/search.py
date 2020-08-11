from collections import OrderedDict
from queue import PriorityQueue
from .abstracts import BaseNode, AbstractTree, AbstractScorer
from typing import List


class Node(BaseNode):
    def __init__(self, concept, parent_node=None, root=None):
        super().__init__(concept, parent_node, root)

    def __str__(self):
        return 'Node at {0}\t{self.concept.str}\tQuality:{self.quality}\tHeuristic:{self.heuristic}\tDepth:{' \
               'self.depth}\tH_exp:{self.h_exp}\t|Children|:{self.refinement_count}'.format(hex(id(self)), self=self)


class CELOESearchTree(AbstractTree):
    def __init__(self, quality_func: AbstractScorer = None, heuristic_func=None):
        super().__init__(quality_func, heuristic_func)
        self.expressionTests = 0

    def update_prepare(self, n):
        self.nodes.pop(n)
        for each in n.children:
            if each in self.nodes:
                self.update_prepare(each)

    def update_done(self, n):
        if not n.children:
            self.nodes[n] = n
        else:
            self.nodes[n] = n
            for each in n.children:
                self.update_done(each)

    def add_node(self, *, parent_node=None, child_node=None):
        if self.redundancy_check(child_node):
            self.quality_func.apply(child_node)  # AccuracyOrTooWeak(n)
            self.expressionTests += 1
            if child_node.quality == 0:  # > too weak
                return False

            self.heuristic_func.apply(child_node, parent_node=parent_node)
            self.nodes[child_node] = child_node

            if parent_node:
                parent_node.add_children(child_node)
            if child_node.quality == 1:  # goal found
                return True
            return False
        return False


class SearchTree(AbstractTree):
    def __init__(self, quality_func: AbstractScorer = None, heuristic_func=None):
        super().__init__(quality_func, heuristic_func)

    def add_node(self, *, parent_node=None, child_node=None):

        if self.redundancy_check(child_node):
            self.quality_func.apply(child_node)  # AccuracyOrTooWeak(n)
            self.expressionTests += 1
            if child_node.quality == 0:  # > too weak
                return False
            self.heuristic_func.apply(child_node)
            self.nodes[child_node] = child_node
            if parent_node:
                parent_node.add_children(child_node)
            if child_node.quality == 1:  # goal found
                return True
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
        from queue import PriorityQueue
        self.items_in_queue = PriorityQueue()
        # self.nodes serves as a gate.

    def add_root(self, node: Node):
        """
        Adds a root node into the search tree.

        Parameters
        ----------
        node : A Node object
        Returns
        -------
        None
        """
        self.quality_func.apply(node)
        self.heuristic_func.apply(node)
        self.items_in_queue.put((-node.heuristic, node.concept.str))  # gets the smallest one.
        self.nodes[node.concept.str] = node

    def add_node(self, *, node: Node, refined_node=Node):
        """
        Adds a node into the search tree.

        Parameters
        ----------
        node : A Node object
        refined_node : A Node object

        Returns
        -------
        True if node is a "goal node", i.e. quality_metric(node)=1.0
        False if node is a "weak node", i.e. quality_metric(node)=0.0
        None otherwise

        Notes
        -----
        node is a refinement of refined_node
        """
        if node.concept.str in self.nodes and node.parent_node != refined_node:
            old_heuristic = node.heuristic
            self.heuristic_func.apply(node, parent_node=refined_node)
            new_heuristic = node.heuristic
            if new_heuristic > old_heuristic:
                node.parent_node.children.remove(node)
                node.parent_node = refined_node
                refined_node.add_children(node)
                self.items_in_queue.put((-node.heuristic, node.concept.str))  # gets the smallest one.
                self.nodes[node.concept.str] = node
        else:
            self.quality_func.apply(node)
            if node.quality == 0:
                return False
            self.heuristic_func.apply(node, parent_node=refined_node)
            self.items_in_queue.put((-node.heuristic, node.concept.str))  # gets the smallest one.
            self.nodes[node.concept.str] = node
            refined_node.add_children(node)
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
