from collections import OrderedDict
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

        if parent_node:
            try:
                assert not (parent_node.quality is None)
            except:
                print(parent_node)
                exit(1)
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
