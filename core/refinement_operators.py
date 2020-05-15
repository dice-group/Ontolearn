import copy

from core.base import KnowledgeBase, Concept
from typing import Set
from itertools import chain, tee


class Refinement:
    """
     A top down/downward refinement operator refinement operator in ALC.
    """

    def __init__(self, kb: KnowledgeBase):
        self.kb = kb

    def refine_atomic_concept(self, C: Concept) -> Set:
        """
        # (1) Create all direct sub concepts of C that are defined in TBOX.
        # (2) Create negations of all leaf concepts in  the concept hierarchy.
        # (3) Create ∀.r.T and ∃.r.T where r is the most general relation.

        :param C: Concept
        :return: A set of refinements.
        """
        # (1) Generate all direct_sub_concepts
        sub_concepts = self.kb.get_direct_sub_concepts(C)
        # (2) Create negation of all leaf_concepts
        negs = self.kb.negation_from_iterables(self.kb.get_leaf_concepts(C))
        # (3) Create ∃.r.T where r is the most general relation.
        existential_rest = self.kb.most_general_existential_restrictions(C)
        universal_rest = self.kb.most_general_universal_restriction(C)
        a, b = tee(chain(sub_concepts, negs, existential_rest, universal_rest))
        for i in a:
            yield i
            for j in copy.copy(b):
                if i == j:
                    continue
                yield self.kb.union(i, j)
                yield self.kb.intersection(i, j)

    def refine_complement_of(self, C: Concept):
        """
        :type C: object
        :param C:
        :return:
        """
        parents = self.kb.get_direct_parents(self.kb.negation(C))
        return self.kb.negation_from_iterables(parents)

    def refine_object_some_values_from(self, C: Concept):
        refs = set()
        # rule 1: EXISTS r.D = > EXISTS r.E
        for i in self.refine(C.filler):
            refs.update(self.kb.existential_restriction(i, C.role))
        return refs

    def refine_object_all_values_from(self, C: Concept):
        """

        :param C:
        :return:
        """
        refs = set()
        # rule 1: Forall r.D = > Forall r.E
        for i in self.refine(C.filler):
            refs.update(self.kb.universal_restriction(i, C.role))
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

    def refine(self, C: Concept):
        assert isinstance(C, Concept)

        result=set()
        if C.is_atomic:
            result.update(self.refine_atomic_concept(C))
        elif C.form == 'ObjectComplementOf':
            result.update(self.refine_complement_of(C))
        elif C.form == 'ObjectSomeValuesFrom':
            result.update(self.refine_object_some_values_from(C))
        elif C.form == 'ObjectAllValuesFrom':
            result.update(self.refine_object_all_values_from(C))
        elif C.form == 'ObjectUnionOf':
            result.update(self.refine_object_union_of(C))
        elif C.form == 'ObjectIntersectionOf':
            result.update(self.refine_object_intersection_of(C))
        else:
            raise ValueError
        return result
