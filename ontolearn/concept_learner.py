from .base_concept_learner import BaseConceptLearner
from .search import Node, CELOESearchTree, SearchTree, SearchTreePriorityQueue
from typing import Set, AnyStr, Iterable, List
from owlready2 import get_ontology
from owlready2.entity import ThingClass
import types
import numpy as np
import pandas as pd
from .heuristics import CELOEHeuristic, DLFOILHeuristic, OCELHeuristic
from .refinement_operators import ModifiedCELOERefinement, CustomRefinementOperator, LengthBasedRefinement
from .metrics import F1, Accuracy, Recall

pd.set_option('display.max_columns', 100)


class CELOE(BaseConceptLearner):
    def __init__(self, knowledge_base, heuristic_func=None, quality_func=None, iter_bound=None,
                 max_num_of_concepts_tested=None,
                 ignored_concepts=None, verbose=None, terminate_on_goal=None):
        if heuristic_func is None:
            heuristic_func = CELOEHeuristic()
        super().__init__(knowledge_base=knowledge_base, refinement_operator=ModifiedCELOERefinement(kb=knowledge_base),
                         search_tree=CELOESearchTree(),
                         quality_func=quality_func,
                         heuristic_func=heuristic_func,
                         ignored_concepts=ignored_concepts,
                         terminate_on_goal=terminate_on_goal,
                         iter_bound=iter_bound, max_num_of_concepts_tested=max_num_of_concepts_tested, verbose=verbose)
        self.h_exp_constant = 0  # add a constant for length of refinements.
        self.max_he, self.min_he = self.max_num_of_concepts_tested, 1

    def next_node_to_expand(self, step: int) -> Node:
        """
        Return most promising node/concept based
        """
        # Original implementation of CELOE: Sort search tree at each step. Quite inefficient.
        self.search_tree.sort_search_tree_by_decreasing_order(key='heuristic')
        if self.verbose > 1:
            self.search_tree.show_search_tree(step)
        for n in self.search_tree:
            return n
        raise ValueError('Search Tree can not be empty.')

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
        self.heuristic_func.apply(node, parent_node=node.parent_node)
        self.search_tree.update_done(node)
        return refinements

    def fit(self, pos: Set[AnyStr], neg: Set[AnyStr]):
        """
        Find hypotheses that explain pos and neg.
        """
        self.initialize_learning_problem(pos=pos, neg=neg, all_instances=self.kb.thing.instances)
        for j in range(1, self.iter_bound):
            most_promising = self.next_node_to_expand(j)
            for ref in self.apply_rho(most_promising):
                goal_found = self.search_tree.add_node(parent_node=most_promising, child_node=ref)
                if goal_found:
                    if self.terminate_on_goal:
                        return self.terminate()
            if self.number_of_tested_concepts >= self.max_num_of_concepts_tested:
                return self.terminate()
        return self.terminate()

    def updateMinMaxHorizExp(self, node: Node):
        """
        @todo Very inefficient. This chunk of code is obtained from DL-learner as it is.
        @param node:
        @return:
        """
        # @todo Currently it is not being used. Investigate the impact of

        he = node.h_exp
        # update maximum value
        self.max_he = self.max_he if self.max_he > he else he

        if self.min_he == he - 1:
            threshold_score = node.heuristic + 1 - node.quality
            sorted_x = sorted(self.search_tree.nodes.items(), key=lambda kv: kv[1].heuristic, reverse=True)
            self.search_tree.nodes = dict(sorted_x)

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


class OCEL(CELOE):
    def __init__(self, knowledge_base, quality_func=None, iter_bound=None, max_num_of_concepts_tested=None,
                 ignored_concepts=None, verbose=None, terminate_on_goal=None):
        super().__init__(knowledge_base=knowledge_base,
                         quality_func=quality_func,
                         heuristic_func=OCELHeuristic(),
                         ignored_concepts=ignored_concepts,
                         terminate_on_goal=terminate_on_goal,
                         iter_bound=iter_bound, max_num_of_concepts_tested=max_num_of_concepts_tested, verbose=verbose)


class LengthBaseLearner(BaseConceptLearner):
    def __init__(self, *, knowledge_base, refinement_operator=None, search_tree=None, quality_func=None,
                 heuristic_func=None, iter_bound=10_000,
                 verbose=False, terminate_on_goal=False, max_num_of_concepts_tested=10_000, min_length=1,
                 ignored_concepts=None):

        if ignored_concepts is None:
            ignored_concepts = {}
        if refinement_operator is None:
            refinement_operator = LengthBasedRefinement(kb=knowledge_base)
        if quality_func is None:
            quality_func = Recall()
        if heuristic_func is None:
            heuristic_func = CELOEHeuristic()
        if search_tree is None:
            search_tree = SearchTreePriorityQueue()

        super().__init__(knowledge_base=knowledge_base, refinement_operator=refinement_operator,
                         search_tree=search_tree,
                         quality_func=quality_func,
                         heuristic_func=heuristic_func,
                         ignored_concepts=ignored_concepts,
                         terminate_on_goal=terminate_on_goal,
                         iter_bound=iter_bound, max_num_of_concepts_tested=max_num_of_concepts_tested, verbose=verbose)
        self.min_length = min_length

    def initialize_root(self):
        root = self.rho.getNode(self.start_class, root=True)
        self.search_tree.quality_func.apply(root)  # AccuracyOrTooWeak(n)
        self.search_tree.heuristic_func.apply(root)  # AccuracyOrTooWeak(n)
        self.search_tree.add_root(root)

    def next_node_to_expand(self, step):
        return self.search_tree.get_most_promising()

    def apply_rho(self, node: Node):
        assert isinstance(node, Node)
        refinements = (self.rho.getNode(i, parent_node=node) for i in
                       self.rho.refine(node, maxlength=len(node) + 1 + self.min_length)
                       if i.str not in self.concepts_to_ignore)
        return refinements

    def fit(self, pos: Set[AnyStr], neg: Set[AnyStr]) -> bool:
        self.search_tree.set_positive_negative_examples(p=pos, n=neg, all_instances=self.kb.thing.instances)
        self.initialize_root()
        for j in range(1, self.iter_bound):
            node_to_expand = self.next_node_to_expand(j)
            for ref in self.apply_rho(node_to_expand):
                goal_found = self.search_tree.add_node(node=ref, refined_node=node_to_expand)
                if goal_found:
                    if self.verbose:
                        print('Goal found after {0} number of concepts tested.'.format(self.number_of_tested_concepts))
                    if self.terminate_on_goal:
                        return self
            if self.number_of_tested_concepts >= self.max_num_of_concepts_tested:
                return self
        return self

    def save_predictions(self, predictions, key: str, serialize_name: str):
        raise NotImplementedError
        assert serialize_name
        assert key
        assert len(predictions)
        metric, attribute = self.get_metric_key(key)
        onto = get_ontology(serialize_name)
        with onto:
            for i in predictions:
                bases = tuple(j for j in i.concept.owl.mro()[:-1])
                new_concept = types.new_class(name=i.concept.str, bases=bases)
                new_concept.comment.append("{0}:{1}".format(metric, getattr(i, attribute)))
        onto.save(serialize_name)

    def show_best_predictions(self, key, top_n=10, serialize_name=None):
        """
        top_n:int
        serialize_name= XXX.owl
        """
        predictions = self.search_tree.show_best_nodes(top_n, key=key)
        if serialize_name is not None:
            onto = get_ontology(serialize_name)
            if key == 'quality':
                metric = self.quality_func.name
                attribute = key
            elif key == 'heuristic':
                metric = self.heuristic.name
                attribute = key
            elif key == 'length':
                metric = key
                attribute = key
            else:
                raise ValueError

            with onto:
                for i in predictions:
                    owlready_obj = i.concept.owl

                    bases = tuple(j for j in owlready_obj.mro()[:-1])

                    new_concept = types.new_class(name=i.concept.str, bases=bases)
                    new_concept.comment.append("{0}:{1}".format(metric, getattr(i, attribute)))

            onto.save(serialize_name)


class CustomConceptLearner(BaseConceptLearner):

    def __init__(self, knowledge_base, quality_func=None, iter_bound=None, max_num_of_concepts_tested=None,
                 ignored_concepts=None, verbose=None, terminate_on_goal=None):
        super().__init__(knowledge_base=knowledge_base, refinement_operator=CustomRefinementOperator(kb=knowledge_base),
                         search_tree=SearchTree(),
                         quality_func=quality_func,
                         heuristic_func=DLFOILHeuristic(),
                         ignored_concepts=ignored_concepts,
                         terminate_on_goal=terminate_on_goal,
                         iter_bound=iter_bound, max_num_of_concepts_tested=max_num_of_concepts_tested, verbose=verbose)

    def next_node_to_expand(self, step):
        self.search_tree.sort_search_tree_by_decreasing_order(key='heuristic')
        for n in self.search_tree:
            return n

    def apply_rho(self, node: Node):
        assert isinstance(node, Node)

        refinements = (self.rho.getNode(i, parent_node=node)
                       for i in self.rho.refine(node.concept) if i is not None and i.str not in self.concepts_to_ignore)
        return refinements

    def fit(self, pos, neg):
        self.initialize_learning_problem(pos=pos, neg=neg, all_instances=self.kb.thing.instances)
        for j in range(1, self.iter_bound):
            node_to_expand = self.next_node_to_expand(j)
            for ref in self.apply_rho(node_to_expand):
                goal_found = self.search_tree.add_node(parent_node=node_to_expand, child_node=ref)
                if goal_found:
                    if self.verbose:
                        print('Goal found after {0} number of concepts tested.'.format(
                            self.search_tree.expressionTests))
                    if self.terminate_on_goal:
                        return self
            if self.number_of_tested_concepts >= self.max_num_of_concepts_tested:
                return self
        return self
