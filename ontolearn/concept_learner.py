from .base_concept_learner import BaseConceptLearner
from .search import Node


class SampleConceptLearner(BaseConceptLearner):
    def __init__(self, *, knowledge_base, quality_func, heuristic_func, iter_bound, verbose, terminate_on_goal=False,
                 ignored_concepts={}):
        super().__init__(knowledge_base=knowledge_base,
                         quality_func=quality_func, heuristic_func=heuristic_func,
                         ignored_concepts=ignored_concepts,
                         terminate_on_goal=terminate_on_goal,
                         iter_bound=iter_bound, verbose=verbose)
        self.max_he, self.min_he = 0, 0

    def next_node_to_expand(self, step):
        self.search_tree.sort_search_tree_by_descending_heuristic_score()
        if self.verbose:
            self.search_tree.show_search_tree(step)
        for n in self.search_tree:
            if n.quality < 1 or (n.h_exp < len(n.concept)):
                return n

    def apply_rho(self, node: Node):
        assert isinstance(node, Node)
        self.search_tree.update_prepare(node)
        # TODO: Very inefficient computation flow as we do not make use of generator.
        # TODO: This chuck of code is obtained from DL-lerner as it is.
        # TODO: Number of refinements must be updated for heuristic value of node
        refinements = [self.rho.getNode(i, parent_node=node)
                       for i in self.rho.refine(node, maxlength=node.h_exp + 1, current_domain=self.start_class)
                       if i.str not in self.concepts_to_ignore]

        node.increment_h_exp()
        node.refinement_count = len(refinements)  # This should be postpone so that we make make use of generator
        self.heuristic.apply(node)

        self.search_tree.update_done(node)
        return refinements

    def predict(self, pos, neg):
        """
        @param pos
        @param neg:
        @return:
        """

        self.search_tree.set_positive_negative_examples(p=pos, n=neg,
                                                        unlabelled=self.kb.thing.instances - (pos.union(neg)))

        self.initialize_root()

        for j in range(1, self.iter_bound):

            node_to_expand = self.next_node_to_expand(j)
            h_exp = node_to_expand.h_exp

            for ref in self.apply_rho(node_to_expand):
                if len(ref) > h_exp:
                    is_added, goal_found = self.search_tree.add_node(ref)
                    if is_added:
                        node_to_expand.add_children(ref)
                    if goal_found:
                        if self.verbose:  # TODO write a function for logging and output that integrates verbose.
                            print('Goal found after {0} number of concepts tested.'.format(
                                self.search_tree.expressionTests))
                        if self.terminate_on_goal:
                            return True
            self.updateMinMaxHorizExp(node_to_expand)

    def updateMinMaxHorizExp(self, node: Node):
        """
        @todo Very inefficient. This chunk of code is obtained from DL-learner as it is.
        @param node:
        @return:
        """
        he = node.h_exp
        # update maximum value
        self.max_he = self.max_he if self.max_he > he else he

        if self.min_he == he - 1:
            threshold_score = node.heuristic + 1 - node.quality
            sorted_x = sorted(self.search_tree._nodes.items(), key=lambda kv: kv[1].heuristic, reverse=True)
            self.search_tree._nodes = dict(sorted_x)

            for item in self.search_tree:
                if node.concept.str != item.concept.str:
                    if item.h_exp == self.min_he:
                        """ we can stop instantly when another node with min. """
                        return
                    if self.search_tree[item].heuristic < threshold_score:
                        """ we can stop traversing nodes when their score is too low. """
                        break
            # inc. minimum since we found no other node which also has min. horiz. exp.
            self.min_he += 1
            print("minimum horizontal expansion is now ", self.min_he)
