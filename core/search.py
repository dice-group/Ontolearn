from collections import OrderedDict
from .abstracts import BaseNode, AbstractTree


class Node(BaseNode):
    def __init__(self, concept, parent_node=None, root=None):
        super().__init__(concept, parent_node, root)

    def __str__(self):
        return 'Node at {0}\t{self.concept.str}\tQuality:{self.quality}\tHeuristic:{self.heuristic}\tDepth:{' \
               'self.depth}\tH_exp:{self.h_exp}\t|Children|:{self.refinement_count}'.format(hex(id(self)), self=self)


class SearchTree(AbstractTree):
    def __init__(self, quality_func, heuristic_func):
        super().__init__(quality_func, heuristic_func)
        self._nodes = dict()

    def __len__(self):
        return len(self._nodes)

    def __getitem__(self, item):
        return self._nodes[item]

    def __iter__(self):
        for k, node in self._nodes.items():
            yield node

    def redundancy_check(self, n: Node):
        if n in self._nodes:
            return False
        return True

    def add_node(self, n: Node):
        """
        Add node into search tree if the following constraints are satisfied
        1) Node is not in the search tree
        2) Node's quality more than 0.
        @param n:
        @return:
        """
        if self.redundancy_check(n):
            self.quality_func.apply(n)  # AccuracyOrTooWeak(n)
            self.expressionTests += 1
            if n.quality == 0:  # > too weak
                return False, False
            self.heuristic_func.apply(n)
            self._nodes[n] = n

            if n.quality == 1:  # goal found
                return True, True

            return True, False
        return False, False

    def update_prepare(self, n: Node):
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

    def get_most_prominent_node(self):
        """
        @todo This is naive implementation from DL-learner. Very inefficient
        We will use priorty queue
        @return:
        """
        for n in self:
            return n

    def sort_search_tree_by_descending_heuristic_score(self):

        sorted_x = sorted(self._nodes.items(), key=lambda kv: kv[1].heuristic, reverse=True)
        self._nodes = OrderedDict(sorted_x)

    def sort_search_tree_by_descending_score(self):
        sorted_x = sorted(self._nodes.items(), key=lambda kv: kv[1].score, reverse=True)
        self._nodes = OrderedDict(sorted_x)
