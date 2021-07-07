import copy
from itertools import chain, tee
import random
from typing import Set, Optional, Iterable, Dict, List, Type, Final, Generator

from .abstracts import BaseRefinement
from .knowledge_base import KnowledgeBase
from owlapy.model import OWLClass, OWLObjectComplementOf, OWLObjectSomeValuesFrom, OWLObjectAllValuesFrom, \
    OWLObjectUnionOf, OWLObjectIntersectionOf, OWLClassExpression, OWLNothing, OWLThing, OWLNaryBooleanClassExpression
from .search import Node, OENode
from .utils import parametrized_performance_debugger


class LengthBasedRefinement(BaseRefinement[Node]):
    """ A top down refinement operator refinement operator in ALC."""
    __slots__ = 'max_child_length', 'apply_combinations', 'min_num_instances'

    _Node: Final = Node

    kb: KnowledgeBase

    max_child_length: int
    apply_combinations: bool
    min_num_instances: int

    def __init__(self, knowledge_base: KnowledgeBase, max_child_length=10, apply_combinations=True):
        super().__init__(knowledge_base)
        self.max_child_length = max_child_length
        self.apply_combinations = apply_combinations
        self.min_num_instances = 0

    def refine_top_concept(self, concept: OWLClass, max_length: int = None) -> Generator:
        if concept.is_owl_nothing():
            yield from {OWLNothing}

        refinement_gate = set()
        # A mapping where keys are lengths (integer) and values are catehgorized refinements of c
        cumulative_refinements = dict()

        # 1.
        generator_container = [self.kb.get_all_sub_concepts(concept)]

        # 2.
        if max_length >= 2 and (self.len(concept) + 1 < self.max_child_length):
            generator_container.append(self.kb.negation_from_iterables(self.kb.get_all_sub_concepts(concept)))

        # 3. and 4.
        if max_length >= 3 and (self.len(concept) + 2 < self.max_child_length):
            generator_container.append(self.kb.most_general_existential_restrictions(domain=concept))
            generator_container.append(self.kb.most_general_universal_restrictions(domain=concept))

        a = chain.from_iterable(generator_container)
        for concept_ref in a:
            if self.kb.individuals_count(concept_ref) >= self.min_num_instances:
                if concept_ref in refinement_gate:
                    raise ValueError
                else:
                    refinement_gate.add(concept_ref)
                    cumulative_refinements.setdefault(self.len(concept_ref), set()).add(concept_ref)
                    yield concept_ref
            else:
                """ Ignore concept that does not satisfy constraint"""

        # 5.
        # The computation in bellow needs to be optimized and parallelized.
        if self.apply_combinations:
            if len(cumulative_refinements) > 0:
                old_len_cumulative_refinements = len(cumulative_refinements)
                while True:
                    temp: Dict[int, Set[OWLClassExpression]] = dict()
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
                                        union = self.kb.union((i, j))
                                        temp.setdefault(self.len(union), set()).add(union)
                                        intersect = self.kb.intersection((i, j))
                                        temp.setdefault(self.len(intersect), set()).add(intersect)
                                        yield intersect
                                        yield union

                    cumulative_refinements.update(temp)
                    new_len_cumulative_refinements = len(cumulative_refinements)
                    if old_len_cumulative_refinements == new_len_cumulative_refinements:
                        break
                    old_len_cumulative_refinements = new_len_cumulative_refinements

    # noinspection DuplicatedCode
    def refine_atomic_concept(self, concept: OWLClass, max_length: int = None) -> Generator:
        """
        Given an atomic class expression c, obtain its refinements by following 5 steps.
        Note that all refinements generated from 1-4 must fulfill constraints,
        e.g. ***self.max_child_length*** and **self.min_num_instances***
        1. Sub    = { x | ( x subClassOf c}
        2. NegSub = { \\neg x | ( x subClassOf c}
        3. MGER   = { \\exists.r.x | r \\in MostGeneral r}
        4. MGUR   = { \\forall.r.x | r \\in MostGeneral r}

        5. Combine 1-4 until we have all refinements have at most max__length.

        Args:
            concept:
            max_length:

        Returns:
            ???
        """
        if concept.is_owl_nothing():
            yield from {OWLNothing}

        refinement_gate = set()
        # A mapping where keys are lengths (integer) and values are categorized refinements of c
        cumulative_refinements = dict()

        # 1.
        generator_container = [self.kb.get_all_sub_concepts(concept)]

        # 2.
        if max_length >= 2 and (self.len(concept) + 1 < self.max_child_length):
            generator_container.append(self.kb.negation_from_iterables(self.kb.get_all_sub_concepts(concept)))

        # 3. and 4.
        if max_length >= 3 and (self.len(concept) + 2 < self.max_child_length):
            generator_container.append(
                self.kb.most_general_existential_restrictions(domain=self.kb.thing, filler=concept))
            generator_container.append(
                self.kb.most_general_universal_restrictions(domain=self.kb.thing, filler=concept))

        a = chain.from_iterable(generator_container)
        for concept_ref in a:
            if self.kb.individuals_count(concept_ref) >= self.min_num_instances:
                if concept_ref in refinement_gate:
                    raise ValueError
                else:
                    refinement_gate.add(concept_ref)
                    cumulative_refinements.setdefault(self.len(concept_ref), set()).add(concept_ref)
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
                                        intersect = self.kb.intersection((i, j))
                                        temp.setdefault(self.len(intersect), set()).add(intersect)
                                        yield intersect

                    cumulative_refinements.update(temp)
                    new_len_cumulative_refinements = len(cumulative_refinements)
                    if old_len_cumulative_refinements == new_len_cumulative_refinements:
                        break
                    old_len_cumulative_refinements = new_len_cumulative_refinements

    def refine_complement_of(self, concept: OWLObjectComplementOf, max_length: int) -> Iterable[OWLClassExpression]:
        parents = self.kb.get_direct_parents(self.kb.negation(concept))
        yield from self.kb.negation_from_iterables(parents)

    def refine_object_some_values_from(self, concept: OWLObjectSomeValuesFrom, max_length: int) -> Iterable[
        OWLClassExpression]:
        assert isinstance(concept, OWLObjectSomeValuesFrom)
        assert isinstance(concept.get_filler(), OWLClassExpression)

        # rule 1: EXISTS r.D = > EXISTS r.E
        for i in self.refine(concept.get_filler(), max_length=max_length):
            yield self.kb.existential_restriction(i, concept.get_property())
        yield self.kb.universal_restriction(concept.get_filler(), concept.get_property())

    def refine_object_all_values_from(self, concept: OWLObjectAllValuesFrom, max_length: int) -> Iterable[
        OWLClassExpression]:
        assert isinstance(concept, OWLObjectAllValuesFrom)

        # rule 1: for all r.D = > for all r.E
        for i in self.refine(concept.get_filler(), max_length=max_length):
            yield self.kb.universal_restriction(i, concept.get_property())

    def _operands_len(self, type_: Type[OWLNaryBooleanClassExpression],
                      ops: List[OWLClassExpression]):
        length = 0
        if len(ops) == 1:
            length += self.len(ops[0])
        elif ops:
            length += self.len(type_(ops))
        return length

    def refine_object_union_of(self, concept: OWLObjectUnionOf, max_length: int) -> Iterable[OWLClassExpression]:
        assert isinstance(concept, OWLObjectUnionOf)

        child: OWLClassExpression
        operands: List[OWLClassExpression] = list(concept.operands())
        for i in range(len(operands)):
            concept_left, concept, concept_right = operands[:i], operands[i], operands[i + 1:]
            other_length = self._operands_len(OWLObjectUnionOf, concept_left + concept_right)

            for ref_concept in self.refine(concept,
                                           max_length=max_length):
                if max_length >= other_length + self.len(ref_concept):
                    yield self.kb.union(concept_left + [ref_concept] + concept_right)

    def refine_object_intersection_of(self, concept: OWLObjectIntersectionOf, max_length: int):
        assert isinstance(concept, OWLObjectIntersectionOf)

        child: OWLClassExpression
        operands: List[OWLClassExpression] = list(concept.operands())
        for i in range(len(operands)):
            concept_left, concept, concept_right = operands[:i], operands[i], operands[i + 1:]
            other_length = self._operands_len(OWLObjectIntersectionOf, concept_left + concept_right)

            for ref_concept in self.refine(concept,
                                           max_length=max_length):
                if max_length >= other_length + self.len(ref_concept):
                    yield self.kb.intersection(concept_left + [ref_concept] + concept_right)

    def refine(self, concept: OWLClass, max_length=3, apply_combinations=None) -> Iterable[OWLClass]:
        """
        OWLClassExpression is not used in the current impl.
        """
        print(OWLClass)
        print(type(concept))
        try:
            assert isinstance(concept, OWLClass)
        except:
            print(concept)
            print(type(concept))
            exit(1)
        if apply_combinations:
            self.apply_combinations = apply_combinations
        if isinstance(concept, OWLClass):
            if concept.is_owl_thing():
                yield from self.refine_top_concept(concept, max_length)
            else:
                yield from self.refine_atomic_concept(concept, max_length)
        elif isinstance(concept, OWLObjectComplementOf):
            yield from self.refine_complement_of(concept, max_length)
        elif isinstance(concept, OWLObjectSomeValuesFrom):
            yield from self.refine_object_some_values_from(concept, max_length)
        elif isinstance(concept, OWLObjectAllValuesFrom):
            yield from self.refine_object_all_values_from(concept, max_length)
        elif isinstance(concept, OWLObjectUnionOf):
            yield from self.refine_object_union_of(concept, max_length)
        elif isinstance(concept, OWLObjectIntersectionOf):
            yield from self.refine_object_intersection_of(concept, max_length)
        else:
            raise ValueError


class ModifiedCELOERefinement(BaseRefinement[OENode]):
    """
     A top down/downward refinement operator refinement operator in ALC.
    """
    __slots__ = 'max_child_length', 'use_negation', 'use_all_constructor'

    _Node: Final = OENode

    kb: KnowledgeBase
    max_child_length: int
    use_negation: bool
    use_all_constructor: bool

    def __init__(self,
                 knowledge_base: KnowledgeBase,
                 max_child_length=10,
                 use_negation: bool = True,
                 use_all_constructor: bool = True):
        # self.topRefinementsCumulative = dict()
        # self.topRefinementsLength = 0
        # self.combos = dict()
        # self.topRefinements = dict()
        # self.topARefinements = dict()
        self.max_child_length = max_child_length
        self.use_negation = use_negation
        self.use_all_constructor = use_all_constructor
        super().__init__(knowledge_base)

    def _operands_len(self, _Type: Type[OWLNaryBooleanClassExpression],
                      ops: List[OWLClassExpression]) -> int:
        """Calculate the length of a OWL Union or Intersection with operands ops

        Args:
            _Type: type of class expression (OWLObjectUnionOf or OWLObjectIntersectionOf)
            ops: list of operands

        Returns:
            length of expression
        """
        length = 0
        if len(ops) == 1:
            length += self.len(ops[0])
        elif ops:
            length += self.len(_Type(ops))
        return length

    def refine_atomic_concept(self, ce: OWLClass, max_length: Optional[int] = None,
                              current_domain: Optional[OWLClassExpression] = None) -> Iterable[OWLClassExpression]:
        """Refinement operator implementation in CELOE-DL-learner,
        distinguishes the refinement of atomic concepts and start concept(they called Top concept).
        [1] Concept learning, Lehmann et. al

            (1) Generate all subconcepts given C, Denoted by (SH_down(C))
            (2) Generate {A AND C | A \\in SH_down(C)}
            (2) Generate {A OR C | A \\in SH_down(C)}
            (3) Generate {\\not A | A \\in SH_down(C) AND_logical \\not \\exist B in T : B \\sqsubset A}
            (4) Generate restrictions.
            (5) Intersect and union (1),(2),(3),(4)
            (6) Create negation of all leaf_concepts

                        (***) The most general relation is not available.

        Args:
            ce:
            max_length:
            current_domain:

        Returns:
            ?
        """
        assert isinstance(ce, OWLClass)

        # iter_container: List[Iterable[OWLClassExpression]] = []
        # (1) Generate all_sub_concepts. Note that originally CELOE obtains only direct subconcepts
        for i in self.kb.get_direct_sub_concepts(ce):
            yield i

        # (2.1) Generate all direct_sub_concepts
        # for i in self.kb.get_direct_sub_concepts(ce):
        #     yield self.kb.intersection((ce, i))
        #     yield self.kb.union((ce, i))

        if self.use_negation:
            # TODO probably not correct/complete
            if max_length >= 2 and (self.len(ce) + 1 <= self.max_child_length):
                # (2.2) Create negation of all leaf_concepts
                # iter_container.append(self.kb.negation_from_iterables(self.kb.get_leaf_concepts(ce)))
                yield from self.kb.negation_from_iterables(self.kb.get_leaf_concepts(ce))

        if max_length >= 3 and (self.len(ce) + 2 <= self.max_child_length):
            # (2.3) Create ∀.r.T and ∃.r.T where r is the most general relation.
            # iter_container.append(self.kb.most_general_existential_restrictions(ce))
            yield from self.kb.most_general_existential_restrictions(domain=ce)
            if self.use_all_constructor:
                # iter_container.append(self.kb.most_general_universal_restriction(ce))
                yield from self.kb.most_general_universal_restrictions(domain=ce)
            yield self.kb.intersection((ce, ce))
            yield self.kb.union((ce, ce))

        # a, b = tee(chain.from_iterable(iter_container))

        # Compute all possible combinations of the disjunction and conjunctions.
        # mem = set()
        # for i in a:
        #     assert i is not None
        #     yield i
        #     for j in copy.copy(b):
        #         assert j is not None
        #         if (i == j) or ((i, j) in mem) or ((j, i) in mem):
        #             continue
        #         mem.add((j, i))
        #         mem.add((i, j))
        #         length = self.len(i) + self.len(j)
        #
        #         if (max_length >= length) and (self.max_child_length >= length + 1):
        #             if not i.is_owl_thing() and not j.is_owl_thing():
        #                 temp_union = self.kb.union((i, j))
        #                 if self.kb.individuals_count(temp_union) < self.kb.individuals_count():
        #                     yield temp_union
        #
        #             temp_intersection = self.kb.intersection((i, j))
        #             if self.kb.individuals_count(temp_intersection) > 0:
        #                 yield temp_intersection

    def refine_complement_of(self, ce: OWLObjectComplementOf, max_length: int,
                             current_domain: Optional[OWLClassExpression] = None) -> Iterable[OWLClassExpression]:
        """
        """
        assert isinstance(ce, OWLObjectComplementOf)

        if self.use_negation:
            parents = self.kb.get_direct_parents(ce.get_operand())
            yield from self.kb.negation_from_iterables(parents)
        else:
            yield from {}

    def refine_object_some_values_from(self, ce: OWLObjectSomeValuesFrom, max_length: int,
                                       current_domain: Optional[OWLClassExpression] = None) \
            -> Iterable[OWLClassExpression]:
        """
        """
        assert isinstance(ce, OWLObjectSomeValuesFrom)
        assert isinstance(ce.get_filler(), OWLClassExpression)

        # rule 1: EXISTS r.D = > EXISTS r.E
        for i in self.refine(ce.get_filler(), max_length=max_length - 2,
                             current_domain=current_domain):
            if i is not None:
                yield self.kb.existential_restriction(i, ce.get_property())

        if self.use_all_constructor:
            yield self.kb.universal_restriction(ce.get_filler(), ce.get_property())

    def refine_object_all_values_from(self, ce: OWLObjectAllValuesFrom, max_length: int,
                                      current_domain: Optional[OWLClassExpression] = None):
        """
        """
        assert isinstance(ce, OWLObjectAllValuesFrom)

        if self.use_all_constructor:
            # rule 1: Forall r.D = > Forall r.E
            for i in self.refine(ce.get_filler(), max_length=max_length - 2,
                                 current_domain=current_domain):
                if i is not None:
                    yield self.kb.universal_restriction(i, ce.get_property())
            # if not concept.get_filler().is_owl_nothing() and concept.get_filler().isatomic and (len(refs) == 0):
            #    # TODO find a way to include nothing concept
            #    refs.update(self.kb.universal_restriction(i, concept.get_property()))
        else:
            yield from {}

    def refine_object_union_of(self, ce: OWLObjectUnionOf, max_length: int,
                               current_domain: Optional[OWLClassExpression]):
        """Given a node corresponding a concepts that comprises union operation.
        1) Obtain two concepts A, B
        2) Refine A and union refiements with B.
        3) Repeat (2) for B.

        Args:
            current_domain:
            node:
            max_length:

        Returns:
            ?
        """
        assert isinstance(ce, OWLObjectUnionOf)

        child: OWLClassExpression
        operands: List[OWLClassExpression] = list(ce.operands())
        for i in range(len(operands)):
            concept_left, concept, concept_right = operands[:i], operands[i], operands[i + 1:]
            concept_length = self.len(concept)
            other_length = self._operands_len(OWLObjectUnionOf, concept_left + concept_right)

            for ref_concept in self.refine(concept,
                                           max_length=max_length - concept_length + other_length,
                                           current_domain=current_domain):
                if max_length >= other_length + self.len(ref_concept):
                    yield self.kb.union(concept_left + [ref_concept] + concept_right)

    def refine_object_intersection_of(self, ce: OWLObjectIntersectionOf, max_length: int,
                                      current_domain: Optional[OWLClassExpression]):
        """
        """
        assert isinstance(ce, OWLObjectIntersectionOf)

        child: OWLClassExpression
        operands: List[OWLClassExpression] = list(ce.operands())
        for i in range(len(operands)):
            concept_left, concept, concept_right = operands[:i], operands[i], operands[i + 1:]
            concept_length = self.len(concept)
            other_length = self._operands_len(OWLObjectIntersectionOf, concept_left + concept_right)

            for ref_concept in self.refine(concept,
                                           max_length=max_length - concept_length + other_length,
                                           current_domain=current_domain):
                if max_length >= other_length + self.len(ref_concept):
                    # if other_concept.instances.isdisjoint(ref_concept.instances):
                    #    continue
                    yield self.kb.intersection(concept_left + [ref_concept] + concept_right)

    def refine(self, ce: OWLClassExpression, max_length: int, current_domain: OWLClassExpression) \
            -> Iterable[OWLClassExpression]:
        """Refine a given concept

        Args:
            ce: concept to refine
            max_length: refine up to this concept length
            current_domain:

        Returns:
            iterable of refined concepts
        """
        assert isinstance(ce, OWLClassExpression)
        if isinstance(ce, OWLClass):
            yield from self.refine_atomic_concept(ce, max_length, current_domain)
        elif isinstance(ce, OWLObjectComplementOf):
            yield from self.refine_complement_of(ce, max_length, current_domain)
        elif isinstance(ce, OWLObjectSomeValuesFrom):
            yield from self.refine_object_some_values_from(ce, max_length, current_domain)
        elif isinstance(ce, OWLObjectAllValuesFrom):
            yield from self.refine_object_all_values_from(ce, max_length, current_domain)
        elif isinstance(ce, OWLObjectUnionOf):
            yield from self.refine_object_union_of(ce, max_length, current_domain)
        elif isinstance(ce, OWLObjectIntersectionOf):
            yield from self.refine_object_intersection_of(ce, max_length, current_domain)
        else:
            raise ValueError


class CustomRefinementOperator(BaseRefinement[Node]):
    def __init__(self, knowledge_base: KnowledgeBase = None, max_size_of_concept=1000, min_size_of_concept=1):
        super().__init__(knowledge_base)

    def get_node(self, c: OWLClassExpression, parent_node=None, root=False):

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

    def refine_object_some_values_from(self, concept: OWLClassExpression):
        assert isinstance(concept, OWLClassExpression)
        for i in self.refine(concept.filler):
            if isinstance(i, OWLClassExpression):
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
            if isinstance(ref_concept_A, OWLClassExpression):
                yield self.kb.union(ref_concept_A, concept_B)

        for ref_concept_B in self.refine(concept_B):
            if isinstance(ref_concept_B, OWLClassExpression):
                yield self.kb.union(ref_concept_B, concept_A)

    def refine_object_intersection_of(self, C: OWLObjectIntersectionOf):
        """

        :param C:
        :return:
        """

        concept_A = C.concept_a
        concept_B = C.concept_b
        for ref_concept_A in self.refine(concept_A):
            if isinstance(ref_concept_A, OWLClassExpression):
                yield self.kb.intersection(ref_concept_A, concept_B)

        for ref_concept_B in self.refine(concept_A):
            if isinstance(ref_concept_B, OWLClassExpression):
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

    def __init__(self, knowledge_base: KnowledgeBase):
        super().__init__(knowledge_base)

    @parametrized_performance_debugger()
    def refine_atomic_concept(self, concept: 'Concept') -> Set:
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

    def refine_complement_of(self, concept: 'Concept'):
        """
        :type concept: Concept
        :param concept:
        :return:
        """
        parents = self.kb.get_direct_parents(self.kb.negation(concept))
        return self.kb.negation_from_iterables(parents)

    def refine_object_some_values_from(self, concept: 'Concept'):
        for i in self.refine(concept.filler):
            yield self.kb.existential_restriction(i, concept.role)

    def refine_object_all_values_from(self, C: 'Concept'):
        """

        :param C:
        :return:
        """
        # rule 1: Forall r.D = > Forall r.E
        for i in self.refine(C.filler):
            yield self.kb.universal_restriction(i, C.role)

    def refine_object_union_of(self, C: 'Concept'):
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

    def refine_object_intersection_of(self, C: 'Concept'):
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

    def refine(self, ce: OWLClassExpression) -> Iterable[OWLClassExpression]:
        assert isinstance(ce, OWLClassExpression)
        if isinstance(ce, OWLClass):
            yield from self.refine_atomic_concept(ce)
        elif isinstance(ce, OWLObjectComplementOf):
            yield from self.refine_complement_of(ce)
        elif isinstance(ce, OWLObjectSomeValuesFrom):
            yield from self.refine_object_some_values_from(ce)
        elif isinstance(ce, OWLObjectAllValuesFrom):
            yield from self.refine_object_all_values_from(ce)
        elif isinstance(ce, OWLObjectUnionOf):
            yield from self.refine_object_union_of(ce)
        elif isinstance(ce, OWLObjectIntersectionOf):
            yield from self.refine_object_intersection_of(ce)
        else:
            raise ValueError


class ExpressRefinement(BaseRefinement[Node]):
    """ A top down refinement operator refinement operator in ALC."""

    def __init__(self, knowledge_base, max_child_length=25, downsample=True, expressivity=0.8):
        super().__init__(knowledge_base)
        self.max_child_length = max_child_length
        self.downsample = downsample
        self.expressivity = expressivity

    def refine_atomic_concept(self, concept: OWLClass):
        if concept.is_owl_nothing():
            yield from {OWLNothing}
        else:
            # Get all subconcepts
            iter_container_sub = list(self.kb.get_all_sub_concepts(concept))
            if iter_container_sub == []:
                iter_container_sub = [concept]
            iter_container_restrict = []
            # Get negations of all subconcepts
            iter_container_neg = list(self.kb.negation_from_iterables(iter_container_sub))
            # (3) Create ∀.r.C and ∃.r.C where r is the most general relation and C in Fillers
            Fillers = {concept, OWLThing, OWLNothing} if len(iter_container_sub) < 5 else {concept, OWLThing,
                                                                                           OWLNothing}.union(
                set(random.sample(iter_container_sub, k=5))).union(set(random.sample(iter_container_neg, k=5)))
            for C in Fillers:
                if self.len(C) + 2 <= self.max_child_length:
                    iter_container_restrict.append(
                        set(self.kb.most_general_universal_restrictions(domain=concept, filler=C)))
                    iter_container_restrict.append(
                        set(self.kb.most_general_existential_restrictions(domain=concept, filler=C)))
            iter_container_restrict = list(set(chain.from_iterable(iter_container_restrict)))
            container = iter_container_restrict + iter_container_neg + iter_container_sub
            if self.downsample:  # downsampling is necessary if no enough computation resources
                assert self.expressivity < 1, "When downsampling, the expressivity is less than 1"
                m = int(self.expressivity * len(container))
                container = random.sample(container, k=max(m, 1))
            else:
                self.expressivity = 1.
            if concept.is_owl_thing():  # If this is satisfied then all possible refinements are subconcepts
                if iter_container_neg + iter_container_restrict:
                    any_refinement = True
                    yield from iter_container_neg + iter_container_restrict
            del iter_container_restrict, iter_container_neg
            any_refinement = False
            # Yield all subconcepts
            if iter_container_sub:
                any_refinement = True
                yield from iter_container_sub
            for sub in iter_container_sub:
                for other_ref in container:
                    if sub != other_ref and self.len(sub) + self.len(other_ref) < self.max_child_length:
                        if concept.is_owl_thing() or (other_ref in iter_container_sub):
                            union = self.kb.union([sub, other_ref])
                            yield union
                            any_refinement = True
                        elif not other_ref in iter_container_sub:
                            union = self.kb.union([sub, other_ref])
                            union = self.kb.intersection([concept, union])
                            if self.len(union) <= self.max_child_length:
                                yield union
                                any_refinement = True
                        intersect = self.kb.intersection([sub, other_ref])
                        if self.len(intersect) <= self.max_child_length:
                            yield intersect
                            any_refinement = True
            if not any_refinement:
                print(f"No refinements found for {repr(concept)}")
                yield concept

    def refine_complement_of(self, concept: OWLObjectComplementOf) -> Generator:
        any_refinement = False
        parents = self.kb.get_direct_parents(self.kb.negation(concept))
        for ref in self.kb.negation_from_iterables(parents):
            if self.len(ref) <= self.max_child_length:
                any_refinement = True
                yield ref
        if not any_refinement:
            yield concept

    def refine_object_some_values_from(self, concept) -> Generator:
        assert isinstance(concept.get_filler(), OWLClassExpression)
        any_refinement = False
        for ref in self.refine(concept.get_filler()):
            if 2 + self.len(ref) <= self.max_child_length:
                any_refinement = True
                reft = self.kb.existential_restriction(ref, concept.get_property())
                yield reft
        if self.len(concept) <= self.max_child_length:
            any_refinement = True
            reft = self.kb.universal_restriction(concept.get_filler(), concept.get_property())
            yield reft
        if not any_refinement:
            yield concept

    def refine_object_all_values_from(self, concept: OWLObjectAllValuesFrom) -> Generator:
        any_refinement = False
        for ref in self.refine(concept.get_filler()):
            if 2 + self.len(ref) <= self.max_child_length:
                any_refinement = True
                reft = self.kb.universal_restriction(ref, concept.get_property())
                yield reft
        if not any_refinement and not concept.get_filler().is_owl_nothing():
            yield concept
        elif not any_refinement and concept.get_filler().is_owl_nothing():
            yield OWLNothing

    def _operands_len(self, type_: Type[OWLNaryBooleanClassExpression],
                      ops: List[OWLClassExpression]):
        length = 0
        if len(ops) == 1:
            length += self.len(ops[0])
        elif ops:
            length += self.len(type_(ops))
        return length

    def refine_object_union_of(self, concept: OWLObjectUnionOf) -> Generator:
        any_refinement = False
        for op in concept.operands():
            if self.len(op) <= self.max_child_length:
                yield op
                any_refinement = True
        operands = list(concept.operands())
        for i in range(len(operands)):
            concept_left, concept_, concept_right = operands[:i], operands[i], operands[i + 1:]
            other_length = self._operands_len(OWLObjectUnionOf, concept_left + concept_right)
            for ref_concept in self.refine(concept_):
                if self.max_child_length >= other_length + self.len(ref_concept):
                    yield self.kb.union(concept_left + [ref_concept] + concept_right)
                    any_refinement = True
        if not any_refinement:
            yield concept

    def refine_object_intersection_of(self, concept: OWLObjectIntersectionOf) -> Generator:
        any_refinement = False
        operands = list(concept.operands())
        for i in range(len(operands)):
            concept_left, concept, concept_right = operands[:i], operands[i], operands[i + 1:]
            other_length = self._operands_len(OWLObjectIntersectionOf, concept_left + concept_right)
            for ref_concept in self.refine(concept):
                if self.max_child_length >= other_length + self.len(ref_concept):
                    yield self.kb.intersection(concept_left + [ref_concept] + concept_right)
                    any_refinement = True
        if not any_refinement:
            yield concept

    def refine(self, concept, **kwargs) -> Generator:
        # we ignore additional arguments like "max_length" or "current_domain" that might be supplied by the
        # algorithm by using **kwargs
        if self.len(concept) == 1:
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
            print(f"{type(concept)} objects are not yet supported")
            raise ValueError
