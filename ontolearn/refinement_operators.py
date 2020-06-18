from .base import KnowledgeBase
from .util import parametrized_performance_debugger
from .concept import Concept
from .search import Node
from .abstracts import BaseRefinement
import copy
from typing import Set, Generator
from itertools import chain, tee


class Refinement(BaseRefinement):
    """
     A top down/downward refinement operator refinement operator in ALC.
    """

    def __init__(self, kb: KnowledgeBase):
        super().__init__(kb)

    @parametrized_performance_debugger()
    def refine_atomic_concept(self, concept: Concept) -> Set:
        """
        # (1) Create all direct sub concepts of C that are defined in TBOX.
        # (2) Create negations of all leaf concepts in  the concept hierarchy.
        # (3) Create ∀.r.T and ∃.r.T where r is the most general relation.

        :param concept: Concept
        :return: A set of refinements.
        """
        # (1) Generate all direct_sub_concepts
        sub_concepts = self.kb.get_direct_sub_concepts(concept)
        # (2) Create negation of all leaf_concepts
        negs = self.kb.negation_from_iterables(self.kb.get_leaf_concepts(concept))
        # (3) Create ∃.r.T where r is the most general relation.
        existential_rest = self.kb.most_general_existential_restrictions(concept)
        universal_rest = self.kb.most_general_universal_restriction(concept)
        a, b = tee(chain(sub_concepts, negs, existential_rest, universal_rest))

        mem = set()
        for i in a:
            yield i
            for j in copy.copy(b):
                if (i == j) or ((i.str, j.str) in mem) or ((j.str, i.str) in mem):
                    continue
                mem.add((j.str, i.str))
                mem.add((i.str, j.str))
                mem.add((j.str, i.str))
                yield self.kb.union(i, j)
                yield self.kb.intersection(i, j)

    def refine_complement_of(self, concept: Concept):
        """
        :type concept: Concept
        :param concept:
        :return:
        """
        parents = self.kb.get_direct_parents(self.kb.negation(concept))
        return self.kb.negation_from_iterables(parents)

    def refine_object_some_values_from(self, concept: Concept):
        refs = set()
        # rule 1: EXISTS r.D = > EXISTS r.E
        for i in self.refine(concept.filler):
            refs.update(self.kb.existential_restriction(i, concept.role))
        return refs

    def refine_object_all_values_from(self, C: Concept):
        """

        :param C:
        :return:
        """
        refs = set()
        # rule 1: Forall r.D = > Forall r.E
        for i in self.refine(C.filler):
            refs.add(self.kb.universal_restriction(i, C.role))
        return refs

    def refine_object_union_of(self, C: Concept):
        """

        :param C:
        :return:
        """

        result = set()
        concept_A = C.concept_a
        concept_B = C.concept_b
        for ref_concept_A in self.refine(concept_A):
            result.add(self.kb.union(ref_concept_A, concept_B))

        for concept_B in self.refine(concept_A):
            result.add(self.kb.union(concept_B, concept_A))

        return result

    def refine_object_intersection_of(self, C: Concept):
        """

        :param C:
        :return:
        """

        result = set()
        concept_A = C.concept_a
        concept_B = C.concept_b
        for ref_concept_A in self.refine(concept_A):
            result.add(self.kb.intersection(ref_concept_A, concept_B))

        for concept_B in self.refine(concept_A):
            result.add(self.kb.intersection(concept_B, concept_A))

        return result

    def refine(self, concept: Concept):
        assert isinstance(concept, Concept)

        if concept.is_atomic:
            yield from self.refine_atomic_concept(concept)
        elif concept.form == 'ObjectComplementOf':
            yield from self.refine_complement_of(concept)
        elif concept.form == 'ObjectSomeValuesFrom':
            yield from self.refine_object_some_values_from(concept)
        elif concept.form == 'ObjectAllValuesFrom':
            yield from self.refine_object_all_values_from(concept)
        elif concept.form == 'ObjectUnionOf':
            yield from self.refine_object_union_of(concept)
        elif concept.form == 'ObjectIntersectionOf':
            yield from self.refine_object_intersection_of(concept)
        else:
            raise ValueError


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
                length = len(i) + len(j)

                if (max_length >= length) and (self.max_child_length >= length + 1):

                    # if i or j is not == T
                    if i.str != 'T' and j.str != 'T':
                        if (i.instances.union(j.instances)) != self.kb.thing.instances:
                            yield self.kb.union(i, j)

                    if not i.instances.isdisjoint(j.instances):
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
