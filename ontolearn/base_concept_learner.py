from abc import ABCMeta, abstractmethod
from .refinement_operators import ModifiedCELOERefinement
from .search import Node
from .search import SearchTree
from .metrics import F1, CELOEHeuristic


class BaseConceptLearner(metaclass=ABCMeta):
    """
    Base class for Concept Learning approaches

    Learning problem definition, Let
        * K = (TBOX, ABOX) be a knowledge base.
        * \ALCConcepts be a set of all ALC concepts.
        * \hypotheses be a set of ALC concepts : \hypotheses \subseteq \ALCConcepts.

        * K_N be a set of all instances.
        * K_C be a set of concepts defined in TBOX: K_C \subseteq \ALCConcepts
        * K_R be a set of properties/relations.

        * E^+, E^- be a set of positive and negative instances and the followings hold
            ** E^+ \cup E^- \subseteq K_N
            ** E^+ \cap E^- = \emptyset

    ##################################################################################################
        The goal is to to learn a set of concepts $\hypotheses \subseteq \ALCConcepts$ such that
              ∀  H \in \hypotheses: { (K \wedge H \models E^+) \wedge  \neg( K \wedge H \models E^-) }.
    ##################################################################################################

    """

    @abstractmethod
    def __init__(self, knowledge_base=None, refinement_operator=None,
                 quality_func=None,
                 heuristic_func=None,
                 search_tree=None,
                 terminate_on_goal=True,
                 iter_bound=10,
                 max_child_length=10,
                 verbose=True, ignored_concepts={}, root_concept=None):
        assert knowledge_base
        self.kb = knowledge_base
        self.heuristic = heuristic_func
        self.quality_func = quality_func
        self.rho = refinement_operator
        self.search_tree = search_tree
        self.concepts_to_ignore = ignored_concepts
        self.start_class = root_concept

        # Memoization
        self.concepts_to_nodes = dict()
        self.iter_bound = iter_bound
        self.terminate_on_goal = terminate_on_goal
        self.verbose = verbose

        if self.rho is None:
            self.rho = ModifiedCELOERefinement(self.kb, max_child_length=max_child_length)
        self.rho.set_concepts_node_mapping(self.concepts_to_nodes)

        if self.heuristic is None:
            self.heuristic = CELOEHeuristic()

        if self.quality_func is None:
            self.quality_func = F1()

        if self.search_tree is None:
            self.search_tree = SearchTree(quality_func=self.quality_func, heuristic_func=self.heuristic)

        if self.start_class is None:
            self.start_class = self.kb.thing

    def initialize_root(self):
        root = self.rho.getNode(self.start_class, root=True)
        self.search_tree.add_node(root)

    def show_best_predictions(self, top_n=10):
        self.search_tree.show_best_nodes(top_n)

    @abstractmethod
    def next_node_to_expand(self, step):
        pass
    @abstractmethod
    def apply_rho(self,args):
        pass

    @abstractmethod
    def predict(self, pos, neg):
        """
        @param pos:
        @param neg:
        @return:
        """
        self.search_tree = SearchTree(quality_func=F1(pos=pos, neg=neg), heuristic_func=self.heuristic)

        self.initialize_root()

        for j in range(1, self.iter_bound):

            node_to_expand = self.next_node_to_expand(j)
            h_exp = node_to_expand.h_exp
            for ref in self.apply_rho(node_to_expand):
                if (len(ref) > h_exp) and ref.depth < self.maxdepth:
                    is_added, goal_found = self.search_tree.add_node(ref)
                    if is_added:
                        node_to_expand.add_children(ref)
                    if goal_found:
                        print(
                            'Goal found after {0} number of concepts tested.'.format(self.search_tree.expressionTests))
                        if self.terminate_on_goal:
                            return True
            self.updateMinMaxHorizExp(node_to_expand)


class SampleConceptLearner:
    """
    SampleConceptLearner that is inspired by The CELOE (Class Expression Learner for Ontology Engineering) algorithm.
    Modifications:
        (1) Implementation of Refinement operator.
    """

    def __init__(self, knowledge_base, max_child_length=5, terminate_on_goal=True, verbose=True, iter_bound=10):
        self.kb = knowledge_base

        self.concepts_to_nodes = dict()
        self.rho = ModifiedCELOERefinement(self.kb, max_child_length=max_child_length)
        self.rho.set_concepts_node_mapping(self.concepts_to_nodes)

        self.verbose = verbose
        # Default values
        self.iter_bound = iter_bound
        self._start_class = self.kb.thing
        self.search_tree = None
        self.maxdepth = 10
        self.max_he, self.min_he = 0, 0
        self.terminate_on_goal = terminate_on_goal

        self.heuristic = CELOEHeuristic()

    def apply_rho(self, node: Node):
        assert isinstance(node, Node)
        self.search_tree.update_prepare(node)
        # TODO: Very inefficient computation flow as we do not make use of generator.
        # TODO: This chuck of code is obtained from DL-lerner as it is.
        # TODO: Number of refinements must be updated for heuristic value of node

        refinements = [self.rho.getNode(i, parent_node=node)
                       for i in self.rho.refine(node, maxlength=node.h_exp + 1, current_domain=self._start_class)]

        node.increment_h_exp()
        node.refinement_count = len(refinements)  # This should be postpone so that we make make use of generator
        self.heuristic.apply(node)

        self.search_tree.update_done(node)
        return refinements

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

    def predict(self, pos, neg):
        """

        @param pos:
        @param neg:
        @return:
        """
        self.search_tree = SearchTree(quality_func=F1(pos=pos, neg=neg), heuristic_func=self.heuristic)

        self.initialize_root()

        for j in range(1, self.iter_bound):

            node_to_expand = self.next_node_to_expand(j)
            h_exp = node_to_expand.h_exp
            for ref in self.apply_rho(node_to_expand):
                if (len(ref) > h_exp) and ref.depth < self.maxdepth:
                    is_added, goal_found = self.search_tree.add_node(ref)
                    if is_added:
                        node_to_expand.add_children(ref)
                    if goal_found:
                        print(
                            'Goal found after {0} number of concepts tested.'.format(self.search_tree.expressionTests))
                        if self.terminate_on_goal:
                            return True
            self.updateMinMaxHorizExp(node_to_expand)
