from .base_concept_learner import BaseConceptLearner
from .search import Node


class CELOE(BaseConceptLearner):
    def __init__(self, *, knowledge_base, refinement_operator, search_tree, quality_func, heuristic_func, iter_bound,
                 verbose, terminate_on_goal=False, min_horizontal_expansion=0,
                 ignored_concepts=None):
        super().__init__(knowledge_base=knowledge_base, refinement_operator=refinement_operator,
                         search_tree=search_tree,
                         quality_func=quality_func,
                         heuristic_func=heuristic_func,
                         ignored_concepts=ignored_concepts,
                         terminate_on_goal=terminate_on_goal,
                         iter_bound=iter_bound, verbose=verbose)
        self.h_exp_constant = min_horizontal_expansion
        self.max_he, self.min_he = self.h_exp_constant, self.h_exp_constant

    def initialize_root(self):
        root = self.rho.getNode(self.start_class, root=True)
        self.search_tree.quality_func.apply(root)  # AccuracyOrTooWeak(n)
        self.search_tree.heuristic_func.apply(root)  # AccuracyOrTooWeak(n)
        self.search_tree[root]=root

    def next_node_to_expand(self, step):
        self.search_tree.sort_search_tree_by_decreasing_order(key='heuristic')
        if self.verbose:
            self.search_tree.show_search_tree(step)
        for n in self.search_tree:
            # if n.quality < 1:# or (n.h_exp < len(n.concept)):
            return n

    def apply_rho(self, node: Node):
        assert isinstance(node, Node)
        self.search_tree.update_prepare(node)

        refinements = [self.rho.getNode(i, parent_node=node) for i in
                       self.rho.refine(node,
                                       maxlength=node.h_exp + 1 + self.h_exp_constant,
                                       current_domain=self.start_class)
                       if i is not None and i.str not in self.concepts_to_ignore]

        node.increment_h_exp(self.h_exp_constant)
        node.refinement_count = len(refinements)  # This should be postpone so that we make make use of generator
        if node.parent_node is not None and node.parent_node.quality is None:
            self.quality_func.apply(node.parent_node)
        self.heuristic.apply(node, parent_node=node.parent_node)
        self.search_tree.update_done(node)
        return refinements

    def predict(self, pos, neg=None):
        self.search_tree.set_positive_negative_examples(p=pos, n=neg,all_instances=self.kb.thing.instances)

        self.initialize_root()

        for j in range(1, self.iter_bound):

            node_to_expand = self.next_node_to_expand(j)
            for ref in self.apply_rho(node_to_expand):
                goal_found = self.search_tree.add_node(parent_node=node_to_expand, child_node=ref)
                if goal_found:
                    if self.verbose:
                        print('Goal found after {0} number of concepts tested.'.format(
                            self.search_tree.expressionTests))
                    if self.terminate_on_goal:
                        return True

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


class CustomConceptLearner(BaseConceptLearner):
    def __init__(self, *, knowledge_base, refinement_operator, search_tree, quality_func, heuristic_func, iter_bound,
                 verbose, terminate_on_goal=False,
                 ignored_concepts=None):
        super().__init__(knowledge_base=knowledge_base, refinement_operator=refinement_operator,
                         search_tree=search_tree,
                         quality_func=quality_func,
                         heuristic_func=heuristic_func,
                         ignored_concepts=ignored_concepts,
                         terminate_on_goal=terminate_on_goal,
                         iter_bound=iter_bound, verbose=verbose)

    def initialize_root(self):
        root = self.rho.getNode(self.start_class, root=True)
        self.search_tree.add_node(child_node=root)

    def next_node_to_expand(self, step):
        self.search_tree.sort_search_tree_by_decreasing_order(key='heuristic')
        for n in self.search_tree:
            return n

    def apply_rho(self, node: Node):
        assert isinstance(node, Node)

        refinements = (self.rho.getNode(i, parent_node=node)
                       for i in self.rho.refine(node.concept) if i is not None and i.str not in self.concepts_to_ignore)


        # Depending on the heuristic function one can modefiy node.
        # self.heuristic.apply(node)
        return refinements

    def predict(self, pos, neg):
        """
        @param pos
        @param neg:
        @return:
        """

        self.search_tree.set_positive_negative_examples(p=pos, n=neg,all_instances=self.kb.thing.instances)

        self.initialize_root()

        for j in range(1, self.iter_bound):

            node_to_expand = self.next_node_to_expand(j)

            for ref in self.apply_rho(node_to_expand):
                goal_found = self.search_tree.add_node(parent_node=node_to_expand, child_node=ref)
                if goal_found:
                    if self.verbose:  # TODO write a function for logging and output that integrates verbose.
                        print('Goal found after {0} number of concepts tested.'.format(
                            self.search_tree.expressionTests))
                    if self.terminate_on_goal:
                        return True
