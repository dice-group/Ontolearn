from collections import OrderedDict
from queue import PriorityQueue
from .abstracts import BaseNode, AbstractTree, AbstractScorer


class Node(BaseNode):
    def __init__(self, concept, parent_node=None, root=None):
        super().__init__(concept, parent_node, root)

    def __str__(self):
        return 'Node at {0}\t{self.concept.str}\tQuality:{self.quality}\tHeuristic:{self.heuristic}\tDepth:{' \
               'self.depth}\tH_exp:{self.h_exp}\t|Children|:{self.refinement_count}'.format(hex(id(self)), self=self)


class CELOESearchTree(AbstractTree):
    def __init__(self, quality_func: AbstractScorer = None, heuristic_func=None):
        super().__init__(quality_func, heuristic_func)

    def update_prepare(self, n):
        self._nodes.pop(n)
        for each in n.children:
            if each in self._nodes:
                self.update_prepare(each)

    def update_done(self, n):
        if not n.children:
            self._nodes[n] = n
        else:
            self._nodes[n] = n
            for each in n.children:
                self.update_done(each)

    def add_node(self, *, parent_node=None, child_node=None):
        if self.redundancy_check(child_node):
            self.quality_func.apply(child_node)  # AccuracyOrTooWeak(n)
            self.expressionTests += 1
            if child_node.quality == 0:  # > too weak
                return False

            self.heuristic_func.apply(child_node, parent_node=parent_node)
            self._nodes[child_node] = child_node

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
            self._nodes[child_node] = child_node
            if parent_node:
                parent_node.add_children(child_node)
            if child_node.quality == 1:  # goal found
                return True
            return False


class SearchTreePriorityQueue(AbstractTree):
    """
    TODO: NOT yet TESTED.
    SearchTree structure should leverage Concept Learning algorithm and heuristic as good as possible.
    """

    def __init__(self, quality_func=None, heuristic_func=None):
        super().__init__(quality_func, heuristic_func)
        self._nodes = PriorityQueue()
        self.items_in_queue = OrderedDict()  # as memory

    def __len__(self):
        return len(self.items_in_queue)

    def add_root(self, node: Node):
        assert isinstance(node, Node)
        self.quality_func.apply(node)
        self.heuristic_func.apply(node)
        self._nodes.put((-node.heuristic, node.concept.str))  # gets the smallest one.
        self.items_in_queue[node.concept.str] = node

    def get_most_promising(self):
        _, most_promising_str = self._nodes.get()
        node = self.items_in_queue[most_promising_str]
        self._nodes.put((-node.heuristic, node.concept.str))  # put it again
        return node

    def get_current_highest_quality(self):
        return max(self.items_in_queue.values(), key=lambda n: n.quality)

    def add_node(self, *, parent_node: Node, child_node=Node):
        assert isinstance(child_node, Node)
        assert isinstance(parent_node, Node)

        if child_node.concept.str in self.items_in_queue:  # must correspond to redundancy check
            # Compare the parents get look at its parent
            # TODO not yet completed
            return False

        else:
            self.quality_func.apply(child_node)
            if child_node.quality == 0:  # implies too weak
                return False
            parent_node.add_children(child_node)
            self.heuristic_func.apply(child_node, parent_node)
            # Depending on the definition of heuristic function, one might recalculate the heuristic value of parent.
            self._nodes.put((-child_node.heuristic, child_node.concept.str))  # gets the smallest one.
            self.items_in_queue[child_node.concept.str] = child_node
            if child_node.quality == 1:  # implies goal found
                return True
            else:
                return False

