from .base_concept_learner import BaseConceptLearner
from .search import Node
from typing import Set, AnyStr
from owlready2 import get_ontology
import types


class CELOE(BaseConceptLearner):
    def __init__(self, *, knowledge_base, refinement_operator, search_tree, quality_func, heuristic_func, iter_bound,
                 verbose, terminate_on_goal=False, max_num_of_concepts_tested=10_000, min_horizontal_expansion=0,
                 ignored_concepts={}):
        super().__init__(knowledge_base=knowledge_base, refinement_operator=refinement_operator,
                         search_tree=search_tree,
                         quality_func=quality_func,
                         heuristic_func=heuristic_func,
                         ignored_concepts=ignored_concepts,
                         terminate_on_goal=terminate_on_goal,
                         iter_bound=iter_bound, max_num_of_concepts_tested=max_num_of_concepts_tested, verbose=verbose)
        self.h_exp_constant = min_horizontal_expansion
        self.max_he, self.min_he = self.h_exp_constant, self.h_exp_constant

    def initialize_root(self):
        root = self.rho.getNode(self.start_class, root=True)
        self.search_tree.quality_func.apply(root)  # AccuracyOrTooWeak(n)
        self.search_tree.heuristic_func.apply(root)  # AccuracyOrTooWeak(n)
        self.search_tree[root] = root

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

    def predict(self, pos: Set[AnyStr], neg: Set[AnyStr]):
        self.search_tree.set_positive_negative_examples(p=pos, n=neg, all_instances=self.kb.thing.instances)

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
                        self.search_tree.sort_search_tree_by_decreasing_order(key='quality')
                        return list(self.search_tree.get_top_n_nodes(1))[0]
            if self.number_of_tested_concepts >= self.max_num_of_concepts_tested:
                break
        return list(self.search_tree.get_top_n_nodes(1))[0]

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
    def __init__(self, *, knowledge_base, refinement_operator, search_tree, quality_func, heuristic_func, iter_bound,
                 verbose, terminate_on_goal=False, min_horizontal_expansion=0, max_num_of_concepts_tested=10_000,
                 ignored_concepts={}):
        super().__init__(knowledge_base=knowledge_base, refinement_operator=refinement_operator,
                         search_tree=search_tree,
                         quality_func=quality_func,
                         heuristic_func=heuristic_func,
                         ignored_concepts=ignored_concepts,
                         terminate_on_goal=terminate_on_goal,
                         iter_bound=iter_bound, max_num_of_concepts_tested=max_num_of_concepts_tested, verbose=verbose)
        self.h_exp_constant = min_horizontal_expansion
        self.max_he, self.min_he = self.h_exp_constant, self.h_exp_constant


class LengthBaseLearner(BaseConceptLearner):
    def __init__(self, *, knowledge_base, refinement_operator, search_tree, quality_func, heuristic_func, iter_bound,
                 verbose, terminate_on_goal=False, max_num_of_concepts_tested=10_000, min_length=1,
                 ignored_concepts={}):

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

    def predict(self, pos: Set[AnyStr], neg: Set[AnyStr], n=10):
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
                        break
            if self.number_of_tested_concepts >= self.max_num_of_concepts_tested:
                break
        return self.search_tree.get_top_n(n=n)

    def __get_metric_key(self, key: str):
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
        return metric, attribute

    def save_predictions(self, predictions, key: str, serialize_name: str):
        assert serialize_name
        assert key
        assert len(predictions)
        metric, attribute = self.__get_metric_key(key)
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
    def __init__(self, *, knowledge_base, refinement_operator, search_tree, quality_func, heuristic_func, iter_bound,
                 verbose, terminate_on_goal=False, max_num_of_concepts_tested=10_000,
                 ignored_concepts={}):
        super().__init__(knowledge_base=knowledge_base, refinement_operator=refinement_operator,
                         search_tree=search_tree,
                         quality_func=quality_func,
                         heuristic_func=heuristic_func,
                         ignored_concepts=ignored_concepts,
                         terminate_on_goal=terminate_on_goal,
                         iter_bound=iter_bound, max_num_of_concepts_tested=max_num_of_concepts_tested, verbose=verbose)

    def initialize_root(self):
        root = self.rho.getNode(self.start_class, root=True)
        self.search_tree.quality_func.apply(root)  # AccuracyOrTooWeak(n)
        self.search_tree.heuristic_func.apply(root)  # AccuracyOrTooWeak(n)
        self.search_tree[root] = root

    def next_node_to_expand(self, step):
        self.search_tree.sort_search_tree_by_decreasing_order(key='heuristic')
        for n in self.search_tree:
            return n

    def apply_rho(self, node: Node):
        assert isinstance(node, Node)

        refinements = (self.rho.getNode(i, parent_node=node)
                       for i in self.rho.refine(node.concept) if i is not None and i.str not in self.concepts_to_ignore)
        return refinements

    def predict(self, pos, neg):
        self.search_tree.set_positive_negative_examples(p=pos, n=neg, all_instances=self.kb.thing.instances)

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
                        self.search_tree.sort_search_tree_by_decreasing_order(key='quality')
                        return list(self.search_tree.get_top_n_nodes(1))[0]
            if self.number_of_tested_concepts >= self.max_num_of_concepts_tested:
                break
        return list(self.search_tree.get_top_n_nodes(1))[0]  # in eficicent.
