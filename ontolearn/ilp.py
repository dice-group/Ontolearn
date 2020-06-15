from collections import OrderedDict
from .concept import Concept
from .search import Node
from .search import SearchTree
from .metrics import F1, CELOEHeuristic
from .abstracts import BaseRefinement
import copy
from typing import Set, Generator
from itertools import chain, tee


class ModifiedCELOERefinement(BaseRefinement):
    """
     A top down/downward refinement operator refinement operator in ALC.
    """

    def __init__(self, kb, max_child_length):
        super().__init__(kb)

        self.topRefinementsCumulative = dict()
        self.topRefinementsLength = 0
        self.max_child_length = max_child_length

        self.combos = dict()
        self.topRefinements = dict()
        self.topARefinements = dict()
        self.concepts_to_nodes = dict()

    def set_concepts_node_mapping(self, m: dict):
        """

        @param m:
        @return:
        """
        self.concepts_to_nodes = m

    def refine_atomic_concept(self, node: Node, max_length: int = None, current_domain: Concept = None) -> Set:
        """
        Refinement operator implementation in CELOE-DL-learner,
        distinguishes the refinement of atomic concepts and start concept(they called Top concept)
            1) An atomic concept is refined by returning all concepts that are subclass of A.
            2) Top concept is refined by
                        (2.1) Generate all direct_sub_concepts.
                        (2.2) Create negation of all leaf_concepts
                        (2.3) Create ∃.r.T where r is the most general relation.
                        (2.4) Union direct_sub_concepts and negated_all_leaf_concepts
                        (2.5) Create unions of each of direct_sub_concept with all other direct_sub_concepts
                        (2.6) Create unions of all of direct_sub_concepts, and negated_all_leaf_concepts
                        (2.7) Create \forall.r.T and \exists.r.T where r is the most general relation.
                        (Currently we are not able to identify general relations.).


        @param node:
        @param max_length:
        @param current_domain:
        @return:
        """

        iter_container = []
        # (2.1) Generate all direct_sub_concepts
        sub_concepts, temp_sub_concepts = tee(self.kb.get_direct_sub_concepts(node.concept))
        iter_container.append(sub_concepts)

        if max_length >= 2 and (len(node.concept) + 1 <= self.max_child_length):
            # (2.2) Create negation of all leaf_concepts
            iter_container.append(self.kb.negation_from_iterables(self.kb.get_leaf_concepts(node.concept)))

        if max_length >= 3 and (len(node.concept) + 2 <= self.max_child_length):
            # (2.3) Create ∀.r.T and ∃.r.T where r is the most general relation.
            iter_container.append(self.kb.most_general_existential_restrictions(node.concept))
            iter_container.append(self.kb.most_general_universal_restriction(node.concept))

        a, b = tee(chain.from_iterable(iter_container))

        # Compute all possible combinations of the disjunction
        mem = set()
        for i in a:
            yield i
            for j in copy.copy(b):
                if (i == j) or ((i.str, j.str) in mem) or ((j.str, i.str) in mem):
                    continue
                mem.add((j.str, i.str))
                mem.add((i.str, j.str))
                mem.add((j.str, i.str))

                length = len(i) + len(j)

                if (max_length >= length) and (self.max_child_length >= length + 1):
                    yield self.kb.union(i, j)
                    yield self.kb.intersection(i, j)
                else:
                    continue

    def refine_complement_of(self, node: Node, maxlength: int, current_domain: Concept = None) -> Generator:
        """

        @param current_domain:
        @param node:
        @param maxlength:
        @return:
        """
        parents = self.kb.get_direct_parents(self.kb.negation(node.concept))
        yield from self.kb.negation_from_iterables(parents)

    def refine_object_some_values_from(self, node: Node, maxlength: int, current_domain: Concept = None) -> Generator:
        """

        @param current_domain:
        @param node:
        @param maxlength:
        @return:
        """
        assert isinstance(node.concept.filler, Concept)
        # rule 1: EXISTS r.D = > EXISTS r.E
        for i in self.refine(self.getNode(node.concept.filler, parent_node=node), maxlength=maxlength - 2,
                             current_domain=current_domain):
            yield self.kb.existential_restriction(i, node.concept.role)

    def refine_object_all_values_from(self, node: Node, maxlength: int, current_domain: Concept = None):
        """

        @param current_domain:
        @param node:
        @param maxlength:
        @return:
        """

        # rule 1: Forall r.D = > Forall r.E
        for i in self.refine(self.getNode(node.concept.filler, parent_node=node), maxlength=maxlength - 2,
                             current_domain=current_domain):
            yield self.kb.universal_restriction(i, node.concept.role)
        # if (node.concept.filler.str != 'Nothing') and node.concept.filler.isatomic and (len(refs) == 0):
        #    refs.update(self.kb.universal_restriction(i, node.concept.role))# TODO find a way to include nothig concept

    def getNode(self, c: Concept, parent_node=None, root=False):

        if c in self.concepts_to_nodes:
            return self.concepts_to_nodes[c]

        if parent_node is None and root is False:
            print(c)
            raise ValueError

        n = Node(concept=c, parent_node=parent_node, root=root)
        self.concepts_to_nodes[c] = n
        return n

    def refine_object_union_of(self, node: Node, maxlength: int, current_domain: Concept):
        """
        Given a node corresponding a concepts that comprises union operation.
        1) Obtain two concepts A, B
        2) Refine A and union refiements with B.
        3) Repeat (2) for B.
        @param current_domain:
        @param node:
        @param maxlength:
        @return:
        """
        concept_A = node.concept.concept_a
        concept_B = node.concept.concept_b

        for ref_concept_A in self.refine(self.getNode(concept_A, parent_node=node),
                                         maxlength=maxlength - len(concept_A) + len(concept_B),
                                         current_domain=current_domain):
            if maxlength >= len(concept_B) + len(ref_concept_A):
                yield self.kb.union(concept_B, ref_concept_A)

        for ref_concept_B in self.refine(self.getNode(concept_B, parent_node=node),
                                         maxlength=maxlength - len(concept_A) + len(concept_B),
                                         current_domain=current_domain):
            if maxlength >= len(concept_A) + len(ref_concept_B):
                yield self.kb.union(concept_A, ref_concept_B)

    def refine_object_intersection_of(self, node: Node, maxlength: int, current_domain: Concept):
        """
        @param node:
        @param maxlength:
        @param current_domain:
        @return:
        """

        concept_A = node.concept.concept_a
        concept_B = node.concept.concept_b
        for ref_concept_A in self.refine(self.getNode(concept_A, parent_node=node),
                                         maxlength=maxlength - len(concept_A) + len(concept_B),
                                         current_domain=current_domain):
            if maxlength >= len(concept_B) + len(ref_concept_A):
                yield self.kb.intersection(concept_B, ref_concept_A)

        for ref_concept_B in self.refine(self.getNode(concept_B, parent_node=node),
                                         maxlength=maxlength - len(concept_A) + len(concept_B),
                                         current_domain=current_domain):
            if maxlength >= len(concept_A) + len(ref_concept_B):
                # if concept_A.instances.isdisjoint(ref_concept_B.instances):
                #    continue
                yield self.kb.intersection(concept_A, ref_concept_B)

    def refine(self, node, maxlength, current_domain):
        """

        @param node:
        @param maxlength:
        @param current_domain:
        @return:
        """
        assert isinstance(node, Node)
        if node.concept.is_atomic:
            return self.refine_atomic_concept(node, maxlength, current_domain)
        elif node.concept.form == 'ObjectComplementOf':
            return self.refine_complement_of(node, maxlength, current_domain)
        elif node.concept.form == 'ObjectSomeValuesFrom':
            return self.refine_object_some_values_from(node, maxlength, current_domain)
        elif node.concept.form == 'ObjectAllValuesFrom':
            return self.refine_object_all_values_from(node, maxlength, current_domain)
        elif node.concept.form == 'ObjectUnionOf':
            return self.refine_object_union_of(node, maxlength, current_domain)
        elif node.concept.form == 'ObjectIntersectionOf':
            return self.refine_object_intersection_of(node, maxlength, current_domain)
        else:
            raise ValueError


class SampleConceptLearner:
    """
    SampleConceptLearner that is inspired by The CELOE (Class Expression Learner for Ontology Engineering) algorithm.
    Modifications:
        (1) Implementation of Refinement operator.
    """

    def __init__(self, knowledge_base, max_child_length=5, terminate_on_goal=True,verbose=True, iter_bound=10):
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
        self.terminate_on_goal=terminate_on_goal

        self.heuristic = CELOEHeuristic()

    def initialize_root(self):

        root = self.rho.getNode(self._start_class, root=True)
        self.search_tree.add_node(root)

    def show_best_predictions(self, top_n):
        sorted_x = sorted(self.search_tree._nodes.items(), key=lambda kv: kv[1].quality, reverse=True)
        self.search_tree._nodes = OrderedDict(sorted_x)
        self.show_search_tree('Final', top_n=top_n+1)

    def show_search_tree(self, ith, top_n=1000):

        print('\n\t\t\t\t\t######## ', ith, 'step Search Tree ###########')
        counter = 1
        for k, v in enumerate(self.search_tree):
            print('\t\t\t\t\t', counter, '-', v)  # , ' - acc:', v.accuracy)

            counter += 1
            if counter == top_n:
                break
        print('\t\t\t\t\t######## Search Tree ###########\n')

    def next_node_to_expand(self, step):
        self.search_tree.sort_search_tree_by_descending_heuristic_score()

        if self.verbose:
            self.show_search_tree(step)

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
                        print('Goal found after {0} number of concepts tested.'.format(self.search_tree.expressionTests))
                        if self.terminate_on_goal:
                            return True
            self.updateMinMaxHorizExp(node_to_expand)
