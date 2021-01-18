from .base import KnowledgeBase
from .core.owl.utils import OWLClassExpressionLengthMetric
from .owlapy.model import OWLClass, OWLObjectComplementOf, OWLObjectSomeValuesFrom, OWLObjectAllValuesFrom, \
    OWLObjectUnionOf, OWLObjectIntersectionOf, OWLClassExpression, OWLNothing
from .utils import parametrized_performance_debugger
from .search import Node
from .abstracts import BaseRefinement
import copy
from typing import Set, Generator, ClassVar
from itertools import chain, tee


class LengthBasedRefinement(BaseRefinement):
    """ A top down refinement operator refinement operator in ALC."""

    length_metric: ClassVar[OWLClassExpressionLengthMetric] = OWLClassExpressionLengthMetric.get_default()

    def __init__(self, kb, max_child_length=10, apply_combinations=True):
        super().__init__(kb)
        self.max_child_length = max_child_length
        self.apply_combinations = apply_combinations
        self.min_num_instances = 0

    def get_node(self, c: OWLClassExpression, parent_node=None, root=False):
        """

        @param c:
        @param parent_node:
        @param root:
        @return:
        """
        if parent_node is None and root is False:
            print(c)
            raise ValueError
        return Node(concept=c, parent_node=parent_node, root=root)

    def refine_top_concept(self, node: Node, max_length: int = None) -> Generator:
        if node.concept.is_owl_nothing():
            yield from {OWLNothing}

        refinement_gate = set()
        # A mapping where keys are lengths (integer) and values are catehgorized refinements of c
        cumulative_refinements = dict()

        # 1.
        generator_container = [self.kb.get_all_sub_concepts(node.concept)]

        # 2.
        if max_length >= 2 and (len(node.concept) + 1 < self.max_child_length):
            generator_container.append(self.kb.negation_from_iterables(self.kb.get_all_sub_concepts(node.concept)))

        # 3. and 4.
        if max_length >= 3 and (len(node.concept) + 2 < self.max_child_length):
            generator_container.append(self.kb.most_general_existential_restrictions(node.concept))
            generator_container.append(self.kb.most_general_universal_restrictions(node.concept))

        a = chain.from_iterable(generator_container)
        for concept_ref in a:
            if len(concept_ref.instances) >= self.min_num_instances:
                if concept_ref in refinement_gate:
                    raise ValueError
                else:
                    refinement_gate.add(concept_ref)
                    cumulative_refinements.setdefault(len(concept_ref), set()).add(concept_ref)
                    yield concept_ref
            else:
                """ Ignore concept that does not satisfy constraint"""

        # 5.
        # The computation in bellow needs to be optimized and parallelized.
        if self.apply_combinations:
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
                                        yield intersect
                                        yield union

                    cumulative_refinements.update(temp)
                    new_len_cumulative_refinements = len(cumulative_refinements)
                    if old_len_cumulative_refinements == new_len_cumulative_refinements:
                        break
                    old_len_cumulative_refinements = new_len_cumulative_refinements

    def refine_atomic_concept(self, node: Node, max_length: int = None) -> Generator:
        """
        Given an atomic class expression c, obtain its refinements by following 5 steps.
        Note that all refinements generated from 1-4 must fulfill constraints,
        e.g. ***self.max_child_length*** and **self.min_num_instances***
        1. Sub    = { x | ( x subClassOf c}
        2. NegSub = { \neg x | ( x subClassOf c}
        3. MGER   = { \exists.r.x | r \in MostGeneral r}
        4. MGUR   = { \forall.r.x | r \in MostGeneral r}

        5. Combine 1-4 until we have all refinements have at most max__length.


        @param node:
        @param max_length:
        @return:
        """
        if node.concept.is_owl_nothing():
            yield from {OWLNothing}

        refinement_gate = set()
        # A mapping where keys are lengths (integer) and values are categorized refinements of c
        cumulative_refinements = dict()

        # 1.
        generator_container = [self.kb.get_all_sub_concepts(node.concept)]

        # 2.
        if max_length >= 2 and (len(node.concept) + 1 < self.max_child_length):
            generator_container.append(self.kb.negation_from_iterables(self.kb.get_all_sub_concepts(node.concept)))

        # 3. and 4.
        if max_length >= 3 and (len(node.concept) + 2 < self.max_child_length):
            generator_container.append(self.kb.most_general_existential_restrictions(node.concept))
            generator_container.append(self.kb.most_general_universal_restrictions(node.concept))

        a = chain.from_iterable(generator_container)
        for concept_ref in a:
            if len(concept_ref.instances) >= self.min_num_instances:
                if concept_ref in refinement_gate:
                    raise ValueError
                else:
                    refinement_gate.add(concept_ref)
                    cumulative_refinements.setdefault(len(concept_ref), set()).add(concept_ref)
                    yield concept_ref
            else:
                """ Ignore concept that does not satisfy constraint"""

        # 5.
        # The computation in bellow needs to be optimized and parallelized.
        if self.apply_combinations:
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
                                        intersect = self.kb.intersection(i, j)
                                        temp.setdefault(len(intersect), set()).add(intersect)
                                        yield intersect

                    cumulative_refinements.update(temp)
                    new_len_cumulative_refinements = len(cumulative_refinements)
                    if old_len_cumulative_refinements == new_len_cumulative_refinements:
                        break
                    old_len_cumulative_refinements = new_len_cumulative_refinements

    def refine_complement_of(self, node: Node, max_length: int) -> Generator:
        parents = self.kb.get_direct_parents(self.kb.negation(node.concept))
        yield from self.kb.negation_from_iterables(parents)

    def refine_object_some_values_from(self, node: Node, max_length: int) -> Generator:
        assert isinstance(node.concept.filler, Concept)
        # rule 1: EXISTS r.D = > EXISTS r.E
        for i in self.refine(self.get_node(node.concept.filler, parent_node=node), max_length=max_length):
            yield self.kb.existential_restriction(i, node.concept.role)
        yield self.kb.universal_restriction(node.concept.filler, node.concept.role)

    def refine_object_all_values_from(self, node: Node, max_length: int):
        # rule 1: for all r.D = > for all r.E
        for i in self.refine(self.get_node(node.concept.filler, parent_node=node), max_length=max_length):
            yield self.kb.universal_restriction(i, node.concept.role)

    def refine_object_union_of(self, node: Node, max_length: int):
        concept_A = node.concept.concept_a
        concept_B = node.concept.concept_b

        for ref_concept_A in self.refine(self.get_node(concept_A, parent_node=node), max_length=max_length):
            if max_length >= len(concept_B) + len(ref_concept_A):
                yield self.kb.union(concept_B, ref_concept_A)

        for ref_concept_B in self.refine(self.get_node(concept_B, parent_node=node), max_length=max_length):
            if max_length >= len(concept_A) + len(ref_concept_B):
                yield self.kb.union(concept_A, ref_concept_B)

    def refine_object_intersection_of(self, node: Node, max_length: int):
        concept_A = node.concept.concept_a
        concept_B = node.concept.concept_b
        for ref_concept_A in self.refine(self.get_node(concept_A, parent_node=node), max_length=max_length):
            if max_length >= len(concept_B) + len(ref_concept_A):
                yield self.kb.intersection(concept_B, ref_concept_A)

        for ref_concept_B in self.refine(self.get_node(concept_B, parent_node=node), max_length=max_length):
            if max_length >= len(concept_A) + len(ref_concept_B):
                yield self.kb.intersection(concept_A, ref_concept_B)

    def refine(self, node, max_length, apply_combinations=None) -> Generator:
        assert isinstance(node, Node)
        if apply_combinations:
            self.apply_combinations = apply_combinations
        if isinstance(node.concept, OWLClass):
            if node.concept.is_owl_thing():
                yield from self.refine_top_concept(node, max_length)
            else:
                yield from self.refine_atomic_concept(node, max_length)
        elif isinstance(node.concept, OWLObjectComplementOf):
            yield from self.refine_complement_of(node, max_length)
        elif isinstance(node.concept, OWLObjectSomeValuesFrom):
            yield from self.refine_object_some_values_from(node, max_length)
        elif isinstance(node.concept, OWLObjectAllValuesFrom):
            yield from self.refine_object_all_values_from(node, max_length)
        elif isinstance(node.concept, OWLObjectUnionOf):
            yield from self.refine_object_union_of(node, max_length)
        elif isinstance(node.concept, OWLObjectIntersectionOf):
            yield from self.refine_object_intersection_of(node, max_length)
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
        """Refinement operator implementation in CELOE-DL-learner,
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

        Args:
            node:
            max_length:
            current_domain:

        Returns:
            ?
        """
        iter_container = []
        # (1) Generate all_sub_concepts. Note that originally CELOE obtains only direct subconcepts
        for i in self.kb.get_direct_sub_concepts(node.concept):
            yield i

        # (2.1) Generate all direct_sub_concepts
        for i in self.kb.get_direct_sub_concepts(node.concept):
            yield self.kb.intersection(node.concept, i)
            yield self.kb.union(node.concept, i)

        # TODO probably not correct/complete
        if max_length >= 2 and (len(node.concept) + 1 <= self.max_child_length):
            # (2.2) Create negation of all leaf_concepts
            iter_container.append(self.kb.negation_from_iterables(self.kb.get_leaf_concepts(node.concept)))

        if max_length >= 3 and (len(node.concept) + 2 <= self.max_child_length):
            # (2.3) Create ∀.r.T and ∃.r.T where r is the most general relation.
            iter_container.append(self.kb.most_general_existential_restrictions(node.concept))
            iter_container.append(self.kb.most_general_universal_restrictions(node.concept))

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

    def refine_complement_of(self, node: Node, max_length: int, current_domain: Concept = None) -> Generator:
        """
        """
        parents = self.kb.get_direct_parents(self.kb.negation(node.concept))
        yield from self.kb.negation_from_iterables(parents)

    def refine_object_some_values_from(self, node: Node, maxlength: int, current_domain: Concept = None) -> Generator:
        """
        """
        assert isinstance(node.concept.filler, Concept)
        # rule 1: EXISTS r.D = > EXISTS r.E
        for i in self.refine(self.get_node(node.concept.filler, parent_node=node), max_length=maxlength - 2,
                             current_domain=current_domain):
            if i is not None:
                yield self.kb.existential_restriction(i, node.concept.role)

        yield self.kb.universal_restriction(node.concept.filler, node.concept.role)

    def refine_object_all_values_from(self, node: Node, max_length: int, current_domain: Concept = None):
        """
        """

        # rule 1: Forall r.D = > Forall r.E
        for i in self.refine(self.get_node(node.concept.filler, parent_node=node), max_length=max_length - 2,
                             current_domain=current_domain):
            if i is not None:
                yield self.kb.universal_restriction(i, node.concept.role)
        # if (node.concept.filler.str != 'Nothing') and node.concept.filler.isatomic and (len(refs) == 0):
        #    refs.update(self.kb.universal_restriction(i, node.concept.role))# TODO find a way to include nothig concept

    def get_node(self, c: Concept, parent_node=None, root=False):

        # if c in self.concepts_to_nodes:
        #    return self.concepts_to_nodes[c]

        if parent_node is None and root is False:
            print(c)
            raise ValueError

        n = Node(concept=c, parent_node=parent_node, root=root)
        # self.concepts_to_nodes[c] = n
        return n

    def refine_object_union_of(self, node: Node, maxlength: int, current_domain: Concept):
        """Given a node corresponding a concepts that comprises union operation.
        1) Obtain two concepts A, B
        2) Refine A and union refiements with B.
        3) Repeat (2) for B.

        Args:
            current_domain:
            node:
            maxlength:

        Returns:
            ?
        """
        concept_A = node.concept.concept_a
        concept_B = node.concept.concept_b

        for ref_concept_A in self.refine(self.get_node(concept_A, parent_node=node),
                                         max_length=maxlength - len(concept_A) + len(concept_B),
                                         current_domain=current_domain):
            if maxlength >= len(concept_B) + len(ref_concept_A):
                yield self.kb.union(concept_B, ref_concept_A)

        for ref_concept_B in self.refine(self.get_node(concept_B, parent_node=node),
                                         max_length=maxlength - len(concept_A) + len(concept_B),
                                         current_domain=current_domain):
            if maxlength >= len(concept_A) + len(ref_concept_B):
                yield self.kb.union(concept_A, ref_concept_B)

    def refine_object_intersection_of(self, node: Node, max_length: int, current_domain: Concept):
        """
        """

        concept_A = node.concept.concept_a
        concept_B = node.concept.concept_b
        for ref_concept_A in self.refine(self.get_node(concept_A, parent_node=node),
                                         max_length=max_length - len(concept_A) + len(concept_B),
                                         current_domain=current_domain):
            if max_length >= len(concept_B) + len(ref_concept_A):
                yield self.kb.intersection(concept_B, ref_concept_A)

        for ref_concept_B in self.refine(self.get_node(concept_B, parent_node=node),
                                         max_length=max_length - len(concept_A) + len(concept_B),
                                         current_domain=current_domain):
            if max_length >= len(concept_A) + len(ref_concept_B):
                # if concept_A.instances.isdisjoint(ref_concept_B.instances):
                #    continue
                yield self.kb.intersection(concept_A, ref_concept_B)

    def refine(self, node: Node, max_length: int, current_domain: OWLClassExpression):
        """

        Args:
            node:
            max_length:
            current_domain:

        Returns:
            ?
        """
        assert isinstance(node, Node)
        if isinstance(node.concept, OWLClass):
            refinement = self.refine_atomic_concept(node, max_length, current_domain)
        elif isinstance(node.concept, OWLObjectComplementOf):
            refinement = self.refine_complement_of(node, max_length, current_domain)
        elif isinstance(node.concept, OWLObjectSomeValuesFrom):
            refinement = self.refine_object_some_values_from(node, max_length, current_domain)
        elif isinstance(node.concept, OWLObjectAllValuesFrom):
            refinement = self.refine_object_all_values_from(node, max_length, current_domain)
        elif isinstance(node.concept, OWLObjectUnionOf):
            refinement = self.refine_object_union_of(node, max_length, current_domain)
        elif isinstance(node.concept, OWLObjectIntersectionOf):
            refinement = self.refine_object_intersection_of(node, max_length, current_domain)
        else:
            raise ValueError

        # @Todo Investigate whether we can avoid this for loop although originally rho in celoe returns none.
        for i in refinement:
            if i is not None:
                yield i


class CustomRefinementOperator(BaseRefinement):
    def __init__(self, kb: KnowledgeBase = None, max_size_of_concept=1000, min_size_of_concept=1):
        super().__init__(kb, max_size_of_concept, min_size_of_concept)

    def get_node(self, c: Concept, parent_node=None, root=False):

        # if c in self.concepts_to_nodes:
        #    return self.concepts_to_nodes[c]

        if parent_node is None and root is False:
            print(c)
            raise ValueError

        n = Node(concept=c, parent_node=parent_node, root=root)
        # self.concepts_to_nodes[c] = n
        return n

    def refine_atomic_concept(self, concept: OWLClass) -> Set:
        # (1) Generate all all sub_concepts
        sub_concepts = self.kb.get_all_sub_concepts(concept)
        # (2) Create negation of all leaf_concepts
        negs = self.kb.negation_from_iterables(self.kb.get_leaf_concepts(concept))
        # (3) Create ∃.r.T where r is the most general relation.
        existential_rest = self.kb.most_general_existential_restrictions(concept)
        universal_rest = self.kb.most_general_universal_restrictions(concept)
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

    def refine_complement_of(self, concept: OWLObjectComplementOf):
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

    def refine_object_all_values_from(self, C: OWLObjectAllValuesFrom):
        """

        :param C:
        :return:
        """
        # rule 1: Forall r.D = > Forall r.E
        for i in self.refine(C.filler):
            yield self.kb.universal_restriction(i, C.role)

    def refine_object_union_of(self, C: OWLObjectUnionOf):
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

    def refine_object_intersection_of(self, C: OWLObjectIntersectionOf):
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

    def refine(self, concept: OWLClassExpression):
        assert isinstance(concept, OWLClassExpression)

        if isinstance(concept, OWLClass):
            yield from self.refine_atomic_concept(concept)
        elif isinstance(concept, OWLObjectComplementOf):
            yield from self.refine_complement_of(concept)
        elif isinstance(concept, OWLObjectSomeValuesFrom):
            yield from self.refine_object_some_values_from(concept)
        elif isinstance(concept, OWLObjectAllValuesFrom):
            yield from self.refine_object_all_values_from(concept)
        elif isinstance(concept, OWLObjectUnionOf):
            yield from self.refine_object_union_of(concept)
        elif isinstance(concept, OWLObjectIntersectionOf):
            yield from self.refine_object_intersection_of(concept)
        else:
            raise ValueError


class ExampleRefinement(BaseRefinement):
    """
     A top down/downward refinement operator refinement operator in ALC.
    """

    def getNode(self, *args, **kwargs):
        pass

    def __init__(self, kb: KnowledgeBase):
        super().__init__(kb)

    @parametrized_performance_debugger()
    def refine_atomic_concept(self, concept: Concept) -> Set:
        """
        # (1) Create all direct sub concepts of C that are defined in TBOX.
        # (2) Create negations of all leaf concepts in  the concept hierarchy.
        # (3) Create ∀.r.T and ∃.r.T where r is the most general relation.
        # (4) Intersect and union set of concepts that are generated in (1-3).
        # Note that this is modified implementation of refinemenet operator proposed in
        Concept Learning in Description Logics Using Refinement Operators

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
            
            
class ExpressRefinement(BaseRefinement):
    """ A top down refinement operator refinement operator in ALC."""

    def __init__(self, kb, max_child_length=25, downsample=True, expressivity=0.3):
        super().__init__(kb)
        self.max_child_length = max_child_length
        self.downsample = downsample
        self.expressivity = expressivity

    def getNode(self, c: Concept, parent_node=None, root=False):

        if c in self.concepts_to_nodes:
            return self.concepts_to_nodes[c]

        if parent_node is None and root is False:
            print(c)
            raise ValueError

        n = Node(concept=c, parent_node=parent_node, root=root)
        self.concepts_to_nodes[c] = n
        return n

    def refine_atomic_concept(self, node: Node):
        #print ("Concept: ", node.concept.str)
        if node.concept.str == 'Nothing':
            yield from {node.concept}
        # Get all subsumption hierarchy
        iter_container_sub = list(self.kb.get_all_sub_concepts(node.concept))
        if iter_container_sub == []:
          iter_container_sub = [node.concept]
        iter_container_restrict = []
        iter_container_neg = list(self.kb.negation_from_iterables(iter_container_sub))
        # (3) Create ∀.r.C and ∃.r.C where r is the most general relation and C in {node.concept}
        for C in {node.concept}:#.union(set(iter_container_sub))).union(set(iter_container_neg)): this ligne can be uncommented for more expressivity
            if C.length + 2 <= self.max_child_length:
                iter_container_restrict.append(set(self.kb.most_general_universal_restriction(C)))
                iter_container_restrict.append(set(self.kb.most_general_existential_restriction(C)))       
        iter_container_restrict = list(set(chain.from_iterable(iter_container_restrict)))
        container = iter_container_restrict + iter_container_neg + iter_container_sub
        if self.downsample:# downsampling is necessary if no enough RAM
            assert self.expressivity < 1, "When downsampling, the expressivity is less than 1"
            m, n = int(0.5*self.expressivity*len(container)), int(self.expressivity*len(iter_container_sub))
            container = random.sample(container, k=max(m,1))
            iter_container_sub_sample = random.sample(iter_container_sub, k=max(1,n))
        else:
            self.expressivity = 1.
            iter_container_sub_sample = iter_container_sub
        if node.concept.str == "Thing":
            yield from iter_container_neg
        del iter_container_restrict, iter_container_neg
        any_refinement = False
        #Yield all subconcepts
        if iter_container_sub:
            yield from iter_container_sub
        #Compute other refinements
        for i in range(len(iter_container_sub_sample)):
            yield iter_container_sub_sample[i]
            any_refinement = True
            for j in range(len(container)):
                if iter_container_sub_sample[i].str != container[j].str and (iter_container_sub_sample[i].length + container[j].length < self.max_child_length):
                    if ((iter_container_sub[i].instances.union(container[j].instances) != self.kb.thing.instances)\
                        and (container[j] in iter_container_sub)) or node.concept.str == "Thing":
                        union = self.kb.union(iter_container_sub_sample[i], container[j])
                        yield union
                        any_refinement = True
                        #self.kb.top_down_direct_concept_hierarchy[node.concept].add(union)
                        #self.kb.down_top_direct_concept_hierarchy[union].add(node.concept)
                    elif (not container[j] in iter_container_sub) and (len(iter_container_sub_sample[i].instances.union(container[j].instances)) < \
                        len(self.kb.thing.instances)):
                        union = self.kb.union(iter_container_sub_sample[i], container[j])
                        union = self.kb.intersection(node.concept, union)
                        yield union
                        any_refinement = True
                    if not iter_container_sub_sample[i].instances.isdisjoint(container[j].instances):
                        intersect = self.kb.intersection(iter_container_sub_sample[i], container[j])
                        #self.kb.top_down_direct_concept_hierarchy[node.concept].add(intersect)
                        #self.kb.down_top_direct_concept_hierarchy[intersect].add(node.concept)
                        yield intersect
                        any_refinement = True
        if not any_refinement:
            yield node.concept
                        
    def refine_complement_of(self, node: Node) -> Generator:
        #print ("Concept: ", node.concept.str)
        any_refinement = False
        parents = self.kb.get_direct_parents(self.kb.negation(node.concept))
        for subs in self.kb.get_all_sub_concepts(node.concept):
            if subs.length <= self.max_child_length:
                any_refinement = True
                yield subs
        for ref in self.kb.negation_from_iterables(parents):
            if ref.length <= self.max_child_length:
                any_refinement = True
                #self.kb.top_down_direct_concept_hierarchy[node.concept].add(ref)
                #self.kb.down_top_direct_concept_hierarchy[ref].add(node.concept)
                yield ref
        if not any_refinement:
            yield node.concept

    def refine_object_some_values_from(self, node: Node) -> Generator:
        #print ("Concept: ", node.concept.str)
        assert isinstance(node.concept.filler, Concept)
        any_refinement = False
        for subs in self.kb.get_all_sub_concepts(node.concept):
            if subs.length <= self.max_child_length:
                any_refinement = True
                yield subs

        for i in self.refine(self.getNode(node.concept.filler, parent_node=node)):
            if i.length <= self.max_child_length:
                any_refinement = True
                #self.kb.top_down_direct_concept_hierarchy[node.concept].add(i)
                #self.kb.down_top_direct_concept_hierarchy[i].add(node.concept)
                yield self.kb.existential_restriction(i, node.concept.role)
        if node.concept.length <= self.max_child_length:
            any_refinement = True
            #self.kb.top_down_direct_concept_hierarchy[node.concept].add(self.kb.universal_restriction(node.concept.filler, node.concept.role))
            #self.kb.down_top_direct_concept_hierarchy[self.kb.universal_restriction(node.concept.filler, node.concept.role)].add(node.concept)
            yield self.kb.universal_restriction(node.concept.filler, node.concept.role)
        if not any_refinement:
            yield node.concept

    def refine_object_all_values_from(self, node: Node):
        #print ("Concept: ", node.concept.str)
        any_refinement = False
        for subs in self.kb.get_all_sub_concepts(node.concept):
            if subs.length <= self.max_child_length:
                any_refinement = True
                yield subs

        for i in self.refine(self.getNode(node.concept.filler, parent_node=node)):
            if i.length <= self.max_child_length:
                any_refinement = True
                #self.kb.top_down_direct_concept_hierarchy[node.concept].add(i)
                #self.kb.down_top_direct_concept_hierarchy[i].add(node.concept)
                yield self.kb.universal_restriction(i, node.concept.role)
        if not any_refinement:
            yield node.concept

    def refine_object_union_of(self, node: Node):
        #print ("Concept: ", node.concept.str)
        concept_A = node.concept.concept_a
        concept_B = node.concept.concept_b
        any_refinement = False
        for subs in self.kb.get_all_sub_concepts(node.concept):
            if subs.length <= self.max_child_length:
                any_refinement = True
                yield subs

        for ref_concept_A in self.refine(self.getNode(concept_A, parent_node=node)):
            if self.max_child_length >= len(concept_B) + len(ref_concept_A) + 1:
                #self.kb.top_down_direct_concept_hierarchy[node.concept].add(self.kb.union(concept_B, ref_concept_A))
                #self.kb.down_top_direct_concept_hierarchy[self.kb.union(concept_B, ref_concept_A)].add(node.concept)
                if ref_concept_A.str != concept_B.str:
                    yield self.kb.union(concept_B, ref_concept_A)
                    any_refinement = True

        for ref_concept_B in self.refine(self.getNode(concept_B, parent_node=node)):
            if self.max_child_length >= len(concept_A) + len(ref_concept_B) + 1:
                #self.kb.top_down_direct_concept_hierarchy[node.concept].add(self.kb.union(concept_A, ref_concept_B))
                #self.kb.down_top_direct_concept_hierarchy[self.kb.union(concept_A, ref_concept_B)].add(node.concept)
                if ref_concept_B.str != concept_A.str:
                    yield self.kb.union(concept_A, ref_concept_B)
                    any_refinement = True
        if not any_refinement:
            yield node.concept

    def refine_object_intersection_of(self, node: Node):
        #print ("Concept: ", node.concept.str)
        concept_A = node.concept.concept_a
        concept_B = node.concept.concept_b
        any_refinement = False
        for subs in self.kb.get_all_sub_concepts(node.concept):
            if subs.length <= self.max_child_length:
                any_refinement = True
                yield subs

        for ref_concept_A in self.refine(self.getNode(concept_A, parent_node=node)):
            if self.max_child_length >= len(concept_B) + len(ref_concept_A) + 1:
                #self.kb.top_down_direct_concept_hierarchy[node.concept].add(self.kb.intersection(concept_B, ref_concept_A))
                #self.kb.down_top_direct_concept_hierarchy[self.kb.intersection(concept_B, ref_concept_A)].add(node.concept)
                if concept_B.str != ref_concept_A.str:
                    any_refinement = True
                    yield self.kb.intersection(concept_B, ref_concept_A)

        for ref_concept_B in self.refine(self.getNode(concept_B, parent_node=node)):
            if self.max_child_length >= len(concept_A) + len(ref_concept_B) + 1:
                #self.kb.top_down_direct_concept_hierarchy[node.concept].add(self.kb.intersection(concept_A, ref_concept_B))
                #self.kb.down_top_direct_concept_hierarchy[self.kb.intersection(concept_A, ref_concept_B)].add(node.concept)
                if concept_A.str != ref_concept_B.str:
                    any_refinement = True
                    yield self.kb.intersection(concept_A, ref_concept_B)
        if not any_refinement:
            yield node.concept

    def refine(self, node) -> Generator:
        assert isinstance(node, Node)
        if node.concept.is_atomic:
            yield from self.refine_atomic_concept(node)
        elif node.concept.form == 'ObjectComplementOf':
            yield from self.refine_complement_of(node)
        elif node.concept.form == 'ObjectSomeValuesFrom':
            yield from self.refine_object_some_values_from(node)
        elif node.concept.form == 'ObjectAllValuesFrom':
            yield from self.refine_object_all_values_from(node)
        elif node.concept.form == 'ObjectUnionOf':
            yield from self.refine_object_union_of(node)
        elif node.concept.form == 'ObjectIntersectionOf':
            yield from self.refine_object_intersection_of(node)
        else:
            raise ValueError
