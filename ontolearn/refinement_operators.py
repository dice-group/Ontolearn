from .base import KnowledgeBase
from .util import parametrized_performance_debugger
from .concept import Concept
from .search import Node
from .abstracts import BaseRefinement
import copy
from typing import Set, Generator
from itertools import chain, tee


class LengthBasedRefinement(BaseRefinement):
    """ A top down refinement operator refinement operator in ALC."""

    def __init__(self, kb, max_child_length=10):
        super().__init__(kb)
        self.max_child_length = max_child_length

    def getNode(self, c: Concept, parent_node=None, root=False):

        if c in self.concepts_to_nodes:
            return self.concepts_to_nodes[c]

        if parent_node is None and root is False:
            print(c)
            raise ValueError

        n = Node(concept=c, parent_node=parent_node, root=root)
        self.concepts_to_nodes[c] = n
        return n

    def refine_atomic_concept(self, node: Node, max_length: int = None) -> Set:
        if node.concept.str == 'Nothing':
            yield from {node.concept}
        # Get all subsumption hierarchy
        generator_container = [self.kb.get_all_sub_concepts(node.concept)]
        # GET all negations.
        if max_length >= 2 and (len(node.concept) + 1 < self.max_child_length):
            # (2) Create negation of all leaf_concepts
            generator_container.append(self.kb.negation_from_iterables(self.kb.get_all_sub_concepts(node.concept)))
        # (3) Create ∀.r.T and ∃.r.T where r is the most general relation.
        if max_length >= 3 and (len(node.concept) + 2 < self.max_child_length):
            # (3) Create ∀.r.T and ∃.r.T where r is the most general relation.
            generator_container.append(self.kb.most_general_existential_restrictions(node.concept))
            generator_container.append(self.kb.most_general_universal_restriction(node.concept))

        a, b = tee(chain.from_iterable(generator_container))

        refinement_gate = set()
        cumulative_refinements = dict()
        for i in set(a):
            refinement_gate.add(i)
            cumulative_refinements.setdefault(len(i), set()).add(i)

        if len(cumulative_refinements) > 0:
            old_len_cumulative_refinements = len(cumulative_refinements)
            while True:
                temp = dict()
                for k, v in cumulative_refinements.items():
                    for kk, vv in cumulative_refinements.items():
                        length = k + kk
                        if (max_length > length) and (self.max_child_length > length + 1):
                            for i in v:
                                for j in vv:

                                    if (i, j) in refinement_gate:
                                        continue

                                    refinement_gate.add((i, j))
                                    refinement_gate.add((j, i))
                                    union = self.kb.union(i, j)
                                    temp.setdefault(len(union), set()).add(union)
                                    intersect = self.kb.intersection(i, j)
                                    temp.setdefault(len(intersect), set()).add(intersect)

                cumulative_refinements.update(temp)
                new_len_cumulative_refinements = len(cumulative_refinements)
                if old_len_cumulative_refinements == new_len_cumulative_refinements:
                    break
                old_len_cumulative_refinements = new_len_cumulative_refinements

            for i in chain.from_iterable(cumulative_refinements.values()):
                yield i

    def refine_complement_of(self, node: Node, maxlength: int) -> Generator:
        parents = self.kb.get_direct_parents(self.kb.negation(node.concept))
        yield from self.kb.negation_from_iterables(parents)

    def refine_object_some_values_from(self, node: Node, maxlength: int) -> Generator:
        assert isinstance(node.concept.filler, Concept)
        # rule 1: EXISTS r.D = > EXISTS r.E
        for i in self.refine(self.getNode(node.concept.filler, parent_node=node), maxlength=maxlength):
            yield self.kb.existential_restriction(i, node.concept.role)

        yield self.kb.universal_restriction(node.concept.filler, node.concept.role)

    def refine_object_all_values_from(self, node: Node, maxlength: int):
        # rule 1: for all r.D = > for all r.E
        for i in self.refine(self.getNode(node.concept.filler, parent_node=node), maxlength=maxlength):
            yield self.kb.universal_restriction(i, node.concept.role)

    def refine_object_union_of(self, node: Node, maxlength: int):
        concept_A = node.concept.concept_a
        concept_B = node.concept.concept_b

        for ref_concept_A in self.refine(self.getNode(concept_A, parent_node=node), maxlength=maxlength):
            if maxlength >= len(concept_B) + len(ref_concept_A):
                yield self.kb.union(concept_B, ref_concept_A)

        for ref_concept_B in self.refine(self.getNode(concept_B, parent_node=node), maxlength=maxlength):
            if maxlength >= len(concept_A) + len(ref_concept_B):
                yield self.kb.union(concept_A, ref_concept_B)

    def refine_object_intersection_of(self, node: Node, maxlength: int):
        concept_A = node.concept.concept_a
        concept_B = node.concept.concept_b
        for ref_concept_A in self.refine(self.getNode(concept_A, parent_node=node), maxlength=maxlength):
            if maxlength >= len(concept_B) + len(ref_concept_A):
                yield self.kb.intersection(concept_B, ref_concept_A)

        for ref_concept_B in self.refine(self.getNode(concept_B, parent_node=node), maxlength=maxlength):
            if maxlength >= len(concept_A) + len(ref_concept_B):
                yield self.kb.intersection(concept_A, ref_concept_B)

    def refine(self, node, maxlength) -> Generator:
        assert isinstance(node, Node)
        if node.concept.is_atomic:
            yield from self.refine_atomic_concept(node, maxlength)
        elif node.concept.form == 'ObjectComplementOf':
            yield from self.refine_complement_of(node, maxlength)
        elif node.concept.form == 'ObjectSomeValuesFrom':
            yield from self.refine_object_some_values_from(node, maxlength)
        elif node.concept.form == 'ObjectAllValuesFrom':
            yield from self.refine_object_all_values_from(node, maxlength)
        elif node.concept.form == 'ObjectUnionOf':
            yield from self.refine_object_union_of(node, maxlength)
        elif node.concept.form == 'ObjectIntersectionOf':
            yield from self.refine_object_intersection_of(node, maxlength)
        else:
            raise ValueError


class ModifiedCELOERefinement(BaseRefinement):
    """
     A top down/downward refinement operator refinement operator in ALC.
    """

    def __init__(self, kb, max_child_length=10):
        super().__init__(kb)
        # self.topRefinementsCumulative = dict()
        # self.topRefinementsLength = 0
        # self.combos = dict()
        # self.topRefinements = dict()
        # self.topARefinements = dict()
        self.max_child_length = max_child_length

    def refine_atomic_concept(self, node: Node, max_length: int = None, current_domain: Concept = None) -> Set:
        """
        Refinement operator implementation in CELOE-DL-learner,
        distinguishes the refinement of atomic concepts and start concept(they called Top concept).
        [1] Concept learning, Lehmann et. al

            (1) Generate all subconcepts given C, Denoted by (SH_down(C))
            (2) Generate {A AND C | A \in SH_down(C)}
            (2) Generate {A OR C | A \in SH_down(C)}
            (3) Generate {\not A | A \in SH_down(C) AND_logical \not \exist B in T : B \sqsubset A}
            (4) Generate restrictions.
            (5) Intersect and union (1),(2),(3),(4)
            (6) Create negation of all leaf_concepts

                        (***) The most general relation is not available.


        @param node:
        @param max_length:
        @param current_domain:
        @return:
        """
        iter_container = []
        # (1) Generate all_sub_concepts. Note that originally CELOE obtains only direct subconcepts
        for i in self.kb.get_direct_sub_concepts(node.concept):
            yield i

        # (2.1) Generate all direct_sub_concepts
        for i in self.kb.get_direct_sub_concepts(node.concept):
            yield self.kb.intersection(node.concept, i)
            yield self.kb.union(node.concept, i)

        if max_length >= 2 and (len(node.concept) + 1 <= self.max_child_length):
            # (2.2) Create negation of all leaf_concepts
            iter_container.append(self.kb.negation_from_iterables(self.kb.get_leaf_concepts(node.concept)))

        if max_length >= 3 and (len(node.concept) + 2 <= self.max_child_length):
            # (2.3) Create ∀.r.T and ∃.r.T where r is the most general relation.
            iter_container.append(self.kb.most_general_existential_restrictions(node.concept))
            iter_container.append(self.kb.most_general_universal_restriction(node.concept))

        a, b = tee(chain.from_iterable(iter_container))

        # Compute all possible combinations of the disjunction and conjunctions.
        mem = set()
        for i in a:
            assert i is not None
            yield i
            for j in copy.copy(b):
                assert j is not None
                if (i == j) or ((i.str, j.str) in mem) or ((j.str, i.str) in mem):
                    continue
                mem.add((j.str, i.str))
                mem.add((i.str, j.str))
                length = len(i) + len(j)

                if (max_length >= length) and (self.max_child_length >= length + 1):
                    if i.str != 'Thing' and j.str != 'Thing':
                        if (i.instances.union(j.instances)) != self.kb.thing.instances:
                            yield self.kb.union(i, j)

                    if not i.instances.isdisjoint(j.instances):
                        yield self.kb.intersection(i, j)

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
            if i is not None:
                yield self.kb.existential_restriction(i, node.concept.role)

        yield self.kb.universal_restriction(node.concept.filler, node.concept.role)

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
            if i is not None:
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
            refinement = self.refine_atomic_concept(node, maxlength, current_domain)
        elif node.concept.form == 'ObjectComplementOf':
            refinement = self.refine_complement_of(node, maxlength, current_domain)
        elif node.concept.form == 'ObjectSomeValuesFrom':
            refinement = self.refine_object_some_values_from(node, maxlength, current_domain)
        elif node.concept.form == 'ObjectAllValuesFrom':
            refinement = self.refine_object_all_values_from(node, maxlength, current_domain)
        elif node.concept.form == 'ObjectUnionOf':
            refinement = self.refine_object_union_of(node, maxlength, current_domain)
        elif node.concept.form == 'ObjectIntersectionOf':
            refinement = self.refine_object_intersection_of(node, maxlength, current_domain)
        else:
            raise ValueError

        # @Todo could it be possible? Investigate.
        for i in refinement:
            if i is not None:
                yield i


class CustomRefinementOperator(BaseRefinement):
    def __init__(self, kb: KnowledgeBase = None, max_size_of_concept=1000, min_size_of_concept=1):
        super().__init__(kb, max_size_of_concept, min_size_of_concept)

    def getNode(self, c: Concept, parent_node=None, root=False):

        if c in self.concepts_to_nodes:
            return self.concepts_to_nodes[c]

        if parent_node is None and root is False:
            print(c)
            raise ValueError

        n = Node(concept=c, parent_node=parent_node, root=root)
        self.concepts_to_nodes[c] = n
        return n

    def refine_atomic_concept(self, concept: Concept) -> Set:
        # (1) Generate all all sub_concepts
        sub_concepts = self.kb.get_all_sub_concepts(concept)
        # (2) Create negation of all leaf_concepts
        negs = self.kb.negation_from_iterables(self.kb.get_leaf_concepts(concept))
        # (3) Create ∃.r.T where r is the most general relation.
        existential_rest = self.kb.most_general_existential_restrictions(concept)
        universal_rest = self.kb.most_general_universal_restriction(concept)
        a, b = tee(chain(sub_concepts, negs, existential_rest, universal_rest))

        mem = set()
        for i in a:
            if i is None:
                continue
            yield i
            for j in copy.copy(b):
                if j is None:
                    continue
                if (i == j) or ((i.str, j.str) in mem) or ((j.str, i.str) in mem):
                    continue
                mem.add((j.str, i.str))
                mem.add((i.str, j.str))
                mem.add((j.str, i.str))

                union = self.kb.union(i, j)
                if union:
                    if not (concept.instances.issubset(union.instances)):
                        yield union

                if i.instances.isdisjoint(j.instances):
                    continue
                inter = self.kb.intersection(i, j)

                if inter:
                    yield inter

    def refine_complement_of(self, concept: Concept):
        """
        :type concept: Concept
        :param concept:
        :return:
        """
        for i in self.kb.negation_from_iterables(self.kb.get_direct_parents(self.kb.negation(concept))):
            yield i

    def refine_object_some_values_from(self, concept: Concept):
        assert isinstance(concept, Concept)
        for i in self.refine(concept.filler):
            if isinstance(i, Concept):
                yield self.kb.existential_restriction(i, concept.role)

    def refine_object_all_values_from(self, C: Concept):
        """

        :param C:
        :return:
        """
        # rule 1: Forall r.D = > Forall r.E
        for i in self.refine(C.filler):
            yield self.kb.universal_restriction(i, C.role)

    def refine_object_union_of(self, C: Concept):
        """

        :param C:
        :return:
        """
        concept_A = C.concept_a
        concept_B = C.concept_b
        for ref_concept_A in self.refine(concept_A):
            if isinstance(ref_concept_A, Concept):
                yield self.kb.union(ref_concept_A, concept_B)

        for ref_concept_B in self.refine(concept_B):
            if isinstance(ref_concept_B, Concept):
                yield self.kb.union(ref_concept_B, concept_A)

    def refine_object_intersection_of(self, C: Concept):
        """

        :param C:
        :return:
        """

        concept_A = C.concept_a
        concept_B = C.concept_b
        for ref_concept_A in self.refine(concept_A):
            if isinstance(ref_concept_A, Concept):
                yield self.kb.intersection(ref_concept_A, concept_B)

        for ref_concept_B in self.refine(concept_A):
            if isinstance(ref_concept_B, Concept):
                yield self.kb.intersection(ref_concept_B, concept_A)

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


class oldExampleRefinement(BaseRefinement):
    def getNode(self, *args, **kwargs):
        pass

    def __init__(self, kb: KnowledgeBase):
        super().__init__(kb)

    @parametrized_performance_debugger()
    def refine_atomic_concept(self, concept: Concept) -> Set:
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
        for i in self.refine(concept.filler):
            yield self.kb.existential_restriction(i, concept.role)

    def refine_object_all_values_from(self, C: Concept):
        """

        :param C:
        :return:
        """
        # rule 1: Forall r.D = > Forall r.E
        for i in self.refine(C.filler):
            yield self.kb.universal_restriction(i, C.role)

    def refine_object_union_of(self, C: Concept):
        """

        :param C:
        :return:
        """
        concept_A = C.concept_a
        concept_B = C.concept_b
        for ref_concept_A in self.refine(concept_A):
            yield self.kb.union(ref_concept_A, concept_B)

        for ref_concept_B in self.refine(concept_B):
            yield self.kb.union(ref_concept_B, concept_A)

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

        for ref_concept_A in self.refine(concept_A):
            result.add(self.kb.intersection(ref_concept_A, concept_B))

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
