from core.base import KnowledgeBase, Concept

class Refinement:
    """
     A top down/downward refinement operator refinement operator in ALC.
    """

    def __init__(self, kb: KnowledgeBase):
        self.kb = kb

    def refine_atomic_concept(self, C: Concept):
        """
        # (1) Create all direct sub concepts of C that are defined in TBOX.
        # (2) Create negations of all leaf concepts in  the concept hierarchy.
        # (3) Create ∀.r.T and ∃.r.T where r is the most general relation.

        :param C:
        :param A: is an instance of Node Class containing Atomic OWL Class expression
        :return:
        """
        result = set()
        # (1) Generate all direct_sub_concepts
        sub_concepts = self.kb.get_direct_sub_concepts(C)
        # (2) Create negation of all leaf_concepts
        negs = self.kb.negation_from_iterables(self.kb.get_leaf_concepts(C))
        # (3) Create ∃.r.T where r is the most general relation.
        existential_rest = self.kb.most_general_existential_restrictions(C)
        universal_rest = self.kb.most_general_universal_restriction(C)

        result.update(negs)
        result.update(sub_concepts)
        result.update(existential_rest)
        result.update(universal_rest)
        temp = set()
        for i in result:
            if i.str == 'Nothing':
                continue
            for j in result:
                if i == j or (j.str == 'Nothing'):
                    continue
                temp.add(self.kb.union(i, j))
                temp.add(self.kb.intersection(i, j))
                break
            break
        result.update(temp)
        return result

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
        if C.is_atomic:
            return self.refine_atomic_concept(C)
        elif C.form == 'ObjectComplementOf':
            return self.refine_complement_of(C)
        elif C.form == 'ObjectSomeValuesFrom':
            return self.refine_object_some_values_from(C)
        elif C.form == 'ObjectAllValuesFrom':
            return self.refine_object_all_values_from(C)
        elif C.form == 'ObjectUnionOf':
            return self.refine_object_union_of(C)
        elif C.form == 'ObjectIntersectionOf':
            return self.refine_object_intersection_of(C)
        else:
            raise ValueError
