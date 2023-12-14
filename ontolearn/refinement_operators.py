"""Refinement Operators for refinement-based concept learners."""
from collections import defaultdict
from itertools import chain
import random
from typing import DefaultDict, Dict, Set, Optional, Iterable, List, Type, Final, Generator
from ontolearn.value_splitter import AbstractValueSplitter, BinningValueSplitter
from owlapy.model.providers import OWLDatatypeMaxInclusiveRestriction, OWLDatatypeMinInclusiveRestriction
from owlapy.vocab import OWLFacet

from .abstracts import BaseRefinement
from .concept_generator import ConceptGenerator
from .knowledge_base import KnowledgeBase
from owlapy.model import OWLObjectPropertyExpression, OWLObjectSomeValuesFrom, OWLObjectAllValuesFrom, \
    OWLObjectIntersectionOf, OWLClassExpression, OWLNothing, OWLThing, OWLNaryBooleanClassExpression, \
    OWLObjectUnionOf, OWLClass, OWLObjectComplementOf, OWLObjectMaxCardinality, OWLObjectMinCardinality, \
    OWLDataSomeValuesFrom, OWLDatatypeRestriction, OWLLiteral, OWLObjectInverseOf, OWLDataProperty, \
    OWLDataHasValue, OWLDataPropertyExpression
from .search import OENode
from typing import Callable, Tuple
from enum import Enum
from owlapy.model import OWLObjectPropertyExpression, OWLObjectSomeValuesFrom, OWLObjectUnionOf, \
    OWLClassExpression, OWLDataHasValue, OWLDataPropertyExpression, OWLDataSomeValuesFrom, OWLLiteral, \
    OWLObjectAllValuesFrom, OWLObjectIntersectionOf, NUMERIC_DATATYPES, OWLDataProperty, OWLObjectProperty, \
    OWLObjectExactCardinality, OWLObjectMaxCardinality, OWLObjectMinCardinality

from ontolearn.ea_utils import PrimitiveFactory, OperatorVocabulary, ToolboxVocabulary, Tree, escape, ind_to_string, \
    owlliteral_to_primitive_string


class LengthBasedRefinement(BaseRefinement):
    """ A top-down refinement operator in ALC."""

    def __init__(self, knowledge_base: KnowledgeBase, use_inverse=False,
                 use_data_properties=False, use_card_restrictions=False, card_limit=11):
        super().__init__(knowledge_base)

        self.use_inverse = use_inverse
        self.use_data_properties = use_data_properties
        self.use_card_restrictions = use_card_restrictions
        self.card_limit = card_limit

        # 1. Number of named classes and sanity checking
        num_of_named_classes = len(set(i for i in self.kb.ontology.classes_in_signature()))
        assert num_of_named_classes == len(list(i for i in self.kb.ontology.classes_in_signature()))
        self.max_len_refinement_top = 5

        self.top_refinements = None # {ref for ref in self.refine_top()}

    def from_iterables(self, cls, a_operands, b_operands):
        assert (isinstance(a_operands, Generator) is False) and (isinstance(b_operands, Generator) is False)
        seen = set()
        results = set()
        for i in a_operands:
            for j in b_operands:
                if i == j:
                    results.add(i)
                elif (i, j) in seen:
                    continue
                else:
                    i_and_j = cls((i, j))
                    seen.add((i, j))
                    seen.add((j, i))
                    results.add(i_and_j)
        return results

    def refine_top(self) -> Iterable:
        """ Refine Top Class Expression """

        # (1) A
        concepts = [i for i in self.kb.get_all_sub_concepts(self.kb.generator.thing)]
        yield from concepts

        # (2) A OR A
        yield from self.from_iterables(cls=OWLObjectUnionOf, a_operands=concepts, b_operands=concepts)
        # (3) A AND A
        yield from self.from_iterables(cls=OWLObjectIntersectionOf, a_operands=concepts, b_operands=concepts)

        # (4) Neg (1) the least general concepts
        neg_concepts = [self.kb.generator.negation(i) for i in concepts]
        # (5) neg A
        yield from neg_concepts
        # (6) neg A OR neg A
        yield from self.from_iterables(cls=OWLObjectUnionOf, a_operands=neg_concepts, b_operands=neg_concepts)
        # (7) neg A AND neg A
        yield from self.from_iterables(cls=OWLObjectIntersectionOf, a_operands=neg_concepts, b_operands=neg_concepts)
        # (8) A OR neg A
        yield from self.from_iterables(cls=OWLObjectUnionOf, a_operands=concepts, b_operands=neg_concepts)
        # (9) A AND neg A
        yield from self.from_iterables(cls=OWLObjectIntersectionOf, a_operands=concepts, b_operands=neg_concepts)

        restrictions = []
        # (10) \for, \exist R A
        # (11) \for, \exist R neg A
        # (12) \for, \exist R⁻ A
        # (13) \for, \exist R⁻ neg A
        for c in concepts + neg_concepts + [self.kb.generator.thing, self.kb.generator.nothing]:
            for dl_role in self.kb.get_object_properties():
                inverse_role = dl_role.get_inverse_property()
                restrictions.append(
                    self.kb.generator.existential_restriction(filler=c, property=dl_role))
                restrictions.append(
                    self.kb.generator.universal_restriction(filler=c, property=dl_role))
                restrictions.append(
                    self.kb.generator.existential_restriction(filler=c, property=inverse_role))
                restrictions.append(
                    self.kb.generator.universal_restriction(filler=c, property=inverse_role))
                # (4) All possible \for and \exist with (1) and (2) and \top and \bottom given roles and inverse roles
                for card in range(0, self.card_limit):
                    restrictions.extend(
                        [self.kb.generator.min_cardinality_restriction(c, dl_role, card),
                         self.kb.generator.max_cardinality_restriction(c, dl_role, card),
                         self.kb.generator.exact_cardinality_restriction(c, dl_role, card),
                         self.kb.generator.min_cardinality_restriction(c, inverse_role, card),
                         self.kb.generator.max_cardinality_restriction(c, inverse_role, card),
                         self.kb.generator.exact_cardinality_restriction(c, inverse_role, card)])
        yield from restrictions

        for bool_dp in self.kb.get_boolean_data_properties():
            print("Not yet boolean data properties for DRILL")
            continue
    def apply_union_and_intersection_from_iterable(self, cont: List) -> Iterable:
        """ Create Union and Intersection OWL Class Expressions.
        1. Create OWLObjectIntersectionOf via logical conjunction of cartesian product of input owl class expressions.
        2. Create OWLObjectUnionOf class expression via logical disjunction pf cartesian product of input owl class
         expressions.
        Repeat 1 and 2 until all concepts having max_len_refinement_top reached.
        """
        cumulative_refinements = dict()
        """ 1. Flatten list of generators """
        for class_expression in cont:
            if class_expression is not self.kb.generator.nothing:
                """ 1.2. Store qualifying concepts based on their lengths """
                cumulative_refinements.setdefault(self.len(class_expression), set()).add(class_expression)
            else:
                """ No need to union or intersect Nothing, i.e. ignore concept that does not satisfy constraint"""
                yield class_expression
        """ 2. Lengths of qualifying concepts """
        lengths = [i for i in cumulative_refinements.keys()]

        seen = set()
        larger_cumulative_refinements = dict()
        """ 3. Iterative over lengths """
        for i in lengths:  # type: int
            """ 3.1 Return all class expressions having the length i """
            yield from cumulative_refinements[i]
            """ 3.2 Create intersection and union of class expressions having the length i with class expressions in
             cumulative_refinements """
            for j in lengths:
                """ 3.3 Ignore if we have already createdValid intersection and union """
                if (i, j) in seen or (j, i) in seen:
                    continue

                seen.add((i, j))
                seen.add((j, i))

                len_ = i + j + 1

                if len_ <= self.max_len_refinement_top:
                    """ 3.4 Intersect concepts having length i with concepts having length j"""
                    intersect_of_concepts = self.kb.generator.intersect_from_iterables(cumulative_refinements[i],
                                                                                       cumulative_refinements[j])
                    """ 3.4 Union concepts having length i with concepts having length j"""
                    union_of_concepts = self.kb.generator.union_from_iterables(cumulative_refinements[i],
                                                                               cumulative_refinements[j])
                    res = set(chain(intersect_of_concepts, union_of_concepts))

                    # Store newly generated concepts at 3.2.
                    if len_ in cumulative_refinements:
                        x = cumulative_refinements[len_]
                        cumulative_refinements[len_] = x.union(res)
                    else:
                        if len_ in larger_cumulative_refinements:
                            x = larger_cumulative_refinements[len_]
                            larger_cumulative_refinements[len_] = x.union(res)
                        else:
                            larger_cumulative_refinements[len_] = res

        for k, v in larger_cumulative_refinements.items():
            yield from v

    def refine_atomic_concept(self, class_expression: OWLClassExpression) -> Iterable[OWLClassExpression]:
        assert isinstance(class_expression, OWLClassExpression)
        for i in self.top_refinements:
            if i.is_owl_nothing() is False and (i != class_expression):
                yield self.kb.generator.intersection((class_expression, i))

    def refine_complement_of(self, class_expression: OWLObjectComplementOf) -> Iterable[OWLClassExpression]:
        """
        Refine OWLObjectComplementOf
        1- Get All direct parents,
        2- Negate (1),
        3- Intersection with T.
        """
        assert isinstance(class_expression, OWLObjectComplementOf)
        yield from self.kb.generator.negation_from_iterables(self.kb.get_direct_parents(class_expression._operand))
        # yield self.kb.generator.intersection((class_expression, self.kb.generator.thing))

    def refine_object_some_values_from(self, class_expression: OWLObjectSomeValuesFrom) -> Iterable[OWLClassExpression]:
        assert isinstance(class_expression, OWLObjectSomeValuesFrom)
        # rule 1: \exists r.D = > for all r.E
        for i in self.refine(class_expression.get_filler()):
            yield self.kb.generator.existential_restriction(i, class_expression.get_property())
        # rule 2: \exists r.D = > \exists r.D AND T
        # yield self.kb.generator.intersection((class_expression, self.kb.generator.thing))

    def refine_object_all_values_from(self, class_expression: OWLObjectAllValuesFrom) -> Iterable[OWLClassExpression]:
        assert isinstance(class_expression, OWLObjectAllValuesFrom)
        # rule 1: \forall r.D = > \forall r.E
        for i in self.refine(class_expression.get_filler()):
            yield self.kb.generator.universal_restriction(i, class_expression.get_property())
        # rule 2: \forall r.D = > \forall r.D AND T
        # yield self.kb.generator.intersection((class_expression, self.kb.generator.thing))

    def refine_object_union_of(self, class_expression: OWLObjectUnionOf) -> Iterable[OWLClassExpression]:
        """
        Refine C =A AND B.
        """
        assert isinstance(class_expression, OWLObjectUnionOf)
        operands: List[OWLClassExpression] = list(class_expression.operands())
        for i in operands:
            for ref_concept_A in self.refine(i):
                if ref_concept_A == class_expression:
                    # No need => Person OR MALE => rho(Person) OR MALE => MALE OR MALE
                    yield class_expression
                yield self.kb.generator.union((class_expression, ref_concept_A))

    def refine_object_intersection_of(self, class_expression: OWLClassExpression) -> Iterable[OWLClassExpression]:
        """ Refine C =A AND B. """
        assert isinstance(class_expression, OWLObjectIntersectionOf)
        operands: List[OWLClassExpression] = list(class_expression.operands())
        for i in operands:
            for ref_concept_A in self.refine(i):
                if ref_concept_A == class_expression:
                    yield class_expression
                yield self.kb.generator.intersection((class_expression, ref_concept_A))

    def refine(self, class_expression) -> Iterable[OWLClassExpression]:
        assert isinstance(class_expression, OWLClassExpression)
        if self.top_refinements is None:
            self.top_refinements = {ref for ref in self.refine_top()}

        if class_expression.is_owl_thing():
            yield from self.top_refinements
        elif class_expression.is_owl_nothing():
            yield from {class_expression}
        elif isinstance(class_expression, OWLObjectIntersectionOf):
            yield from self.refine_object_intersection_of(class_expression)
        elif isinstance(class_expression, OWLObjectComplementOf):
            yield from self.refine_complement_of(class_expression)
        elif isinstance(class_expression, OWLObjectAllValuesFrom):
            yield from self.refine_object_all_values_from(class_expression)
        elif isinstance(class_expression, OWLObjectUnionOf):
            yield from self.refine_object_union_of(class_expression)
        elif isinstance(class_expression, OWLObjectSomeValuesFrom):
            yield from self.refine_object_some_values_from(class_expression)
        elif self.len(class_expression) == 1:
            yield from self.refine_atomic_concept(class_expression)

        elif isinstance(class_expression, OWLObjectMaxCardinality):
            yield from (self.kb.generator.intersection((class_expression, i)) for i in self.top_refinements)
        elif isinstance(class_expression, OWLObjectExactCardinality):
            yield from (self.kb.generator.intersection((class_expression, i)) for i in self.top_refinements)
        elif isinstance(class_expression, OWLObjectMinCardinality):
            yield from (self.kb.generator.intersection((class_expression, i)) for i in self.top_refinements)
        else:
            raise ValueError(f"{type(class_expression)} objects are not yet supported")


class ModifiedCELOERefinement(BaseRefinement[OENode]):
    """
     A top down/downward refinement operator in SHIQ(D).
    """
    __slots__ = 'max_child_length', 'use_negation', 'use_all_constructor', 'use_inverse', 'use_card_restrictions', \
        'max_nr_fillers', 'card_limit', 'use_numeric_datatypes', 'use_boolean_datatype', 'dp_splits', \
        'value_splitter', 'use_time_datatypes', 'generator'

    _Node: Final = OENode

    kb: KnowledgeBase
    value_splitter: Optional[AbstractValueSplitter]
    max_child_length: int
    use_negation: bool
    use_all_constructor: bool
    use_card_restrictions: bool
    use_numeric_datatypes: bool
    use_time_datatypes: bool
    use_boolean_datatype: bool
    card_limit: int

    max_nr_fillers: DefaultDict[OWLObjectPropertyExpression, int]
    dp_splits: Dict[OWLDataPropertyExpression, List[OWLLiteral]]
    generator: ConceptGenerator

    def __init__(self,
                 knowledge_base: KnowledgeBase,
                 value_splitter: Optional[AbstractValueSplitter] = None,
                 max_child_length: int = 10,
                 use_negation: bool = True,
                 use_all_constructor: bool = True,
                 use_inverse: bool = True,
                 use_card_restrictions: bool = True,
                 use_numeric_datatypes: bool = True,
                 use_time_datatypes: bool = True,
                 use_boolean_datatype: bool = True,
                 card_limit: int = 10):
        # self.topRefinementsCumulative = dict()
        # self.topRefinementsLength = 0
        # self.combos = dict()
        # self.topRefinements = dict()
        # self.topARefinements = dict()
        self.value_splitter = value_splitter
        self.max_child_length = max_child_length
        self.use_negation = use_negation
        self.use_all_constructor = use_all_constructor
        self.use_inverse = use_inverse
        self.use_card_restrictions = use_card_restrictions
        self.use_numeric_datatypes = use_numeric_datatypes
        self.use_time_datatypes = use_time_datatypes
        self.use_boolean_datatype = use_boolean_datatype
        self.card_limit = card_limit
        self.generator = ConceptGenerator()
        super().__init__(knowledge_base)
        self._setup()

    def _setup(self):
        if self.value_splitter is None:
            self.value_splitter = BinningValueSplitter()

        if self.use_card_restrictions:
            obj_properties = list(self.kb.get_object_properties())
            if self.use_inverse:
                obj_properties.extend(list(map(OWLObjectInverseOf, obj_properties)))

            self.max_nr_fillers = defaultdict(int)
            for prop in obj_properties:
                for ind in self.kb.individuals():
                    num = sum(1 for _ in zip(self.kb.get_object_property_values(ind, prop), range(self.card_limit)))
                    self.max_nr_fillers[prop] = max(self.max_nr_fillers[prop], num)
                    if num == self.card_limit:
                        break

        split_dps = []
        if self.use_numeric_datatypes:
            split_dps.extend(self.kb.get_numeric_data_properties())

        if self.use_time_datatypes:
            split_dps.extend(self.kb.get_time_data_properties())

        if len(split_dps) > 0:
            self.dp_splits = self.value_splitter.compute_splits_properties(self.kb.reasoner, split_dps)

    def _operands_len(self, _Type: Type[OWLNaryBooleanClassExpression],
                      ops: List[OWLClassExpression]) -> int:
        """Calculate the length of a OWL Union or Intersection with operands ops.

        Args:
            _Type: Type of class expression (OWLObjectUnionOf or OWLObjectIntersectionOf)
            ops: list of operands.

        Returns:
            Length of expression.
        """
        length = 0
        if len(ops) == 1:
            length += self.len(ops[0])
        elif ops:
            length += self.len(_Type(ops))
        return length

    def _get_dp_restrictions(self, data_properties: Iterable[OWLDataProperty]) -> List[OWLDataSomeValuesFrom]:
        restrictions = []
        for dp in data_properties:
            splits = self.dp_splits[dp]
            if len(splits) > 0:
                restrictions.append(self.generator.data_existential_restriction(
                    filler=OWLDatatypeMinInclusiveRestriction(splits[0]), property=dp))
                restrictions.append(self.generator.data_existential_restriction(
                    filler=OWLDatatypeMaxInclusiveRestriction(splits[-1]), property=dp))
        return restrictions

    def _get_current_domain(self, property_: OWLObjectPropertyExpression) -> OWLClassExpression:
        func = self.kb.get_object_property_domains \
            if isinstance(property_, OWLObjectInverseOf) else self.kb.get_object_property_ranges
        return func(property_.get_named_property())

    def refine_atomic_concept(self, ce: OWLClass, max_length: int,
                              current_domain: Optional[OWLClassExpression] = None) -> Iterable[OWLClassExpression]:
        """Refinement operator implementation in CELOE-DL-learner,
        distinguishes the refinement of atomic concepts and start concept(they called Top concept).
        [1] Concept learning, Lehmann et. al

            (1) Generate all subconcepts given C, Denoted by (SH_down(C)),
            (2) Generate {A AND C | A \\in SH_down(C)},
            (2) Generate {A OR C | A \\in SH_down(C)},
            (3) Generate {\\not A | A \\in SH_down(C) AND_logical \\not \\exist B in T : B \\sqsubset A},
            (4) Generate restrictions,
            (5) Intersect and union (1),(2),(3),(4),
            (6) Create negation of all leaf_concepts.

                        (***) The most general relation is not available.

        Args:
            ce: Atomic concept to refine.
            max_length: Refine up to this concept length.
            current_domain: Domain.

        Returns:
            Iterable of refined concepts.
        """
        assert isinstance(ce, OWLClass)

        if current_domain is None:
            current_domain = OWLThing

        iter_container: List[Iterable[OWLClassExpression]] = []
        # (1) Generate all_sub_concepts. Note that originally CELOE obtains only direct subconcepts
        iter_container.append(self.kb.get_direct_sub_concepts(ce))
        # for i in self.kb.get_direct_sub_concepts(ce):
        #     yield i

        # (2.1) Generate all direct_sub_concepts
        # for i in self.kb.get_direct_sub_concepts(ce):
        #     yield self.kb.intersection((ce, i))
        #     yield self.kb.union((ce, i))

        if self.use_negation:
            # TODO probably not correct/complete
            if max_length >= 2 and (self.len(ce) + 1 <= self.max_child_length):
                # (2.2) Create negation of all leaf_concepts
                iter_container.append(self.generator.negation_from_iterables(self.kb.get_leaf_concepts(ce)))
                # yield from self.kb.negation_from_iterables(self.kb.get_leaf_concepts(ce))

        if max_length >= 3 and (self.len(ce) + 2 <= self.max_child_length):
            # (2.3) Create ∀.r.T and ∃.r.T where r is the most general relation.
            iter_container.append(self.kb.most_general_existential_restrictions(domain=current_domain))
            # yield from self.kb.most_general_existential_restrictions(domain=ce)
            if self.use_all_constructor:
                iter_container.append(self.kb.most_general_universal_restrictions(domain=current_domain))
                # yield from self.kb.most_general_universal_restrictions(domain=ce)
            if self.use_inverse:
                iter_container.append(self.kb.most_general_existential_restrictions_inverse(domain=current_domain))
                # yield from self.kb.most_general_existential_restrictions_inverse(domain=ce)
                if self.use_all_constructor:
                    iter_container.append(self.kb.most_general_universal_restrictions_inverse(domain=current_domain))
                    # yield from self.kb.most_general_universal_restrictions_inverse(domain=ce)
            if self.use_numeric_datatypes:
                iter_container.append(self._get_dp_restrictions(
                    self.kb.most_general_numeric_data_properties(domain=current_domain)))
            if self.use_time_datatypes:
                iter_container.append(self._get_dp_restrictions(
                    self.kb.most_general_time_data_properties(domain=current_domain)))
            if self.use_boolean_datatype:
                bool_res = []
                for bool_dp in self.kb.most_general_boolean_data_properties(domain=current_domain):
                    bool_res.append(self.generator.data_has_value_restriction(value=OWLLiteral(True), property=bool_dp))
                    bool_res.append(self.generator.data_has_value_restriction(value=OWLLiteral(False),
                                                                              property=bool_dp))
                iter_container.append(bool_res)
            # yield self.kb.intersection((ce, ce))
            # yield self.kb.union((ce, ce))

        if self.use_card_restrictions and max_length >= 4 and (self.max_child_length >= self.len(ce) + 3):
            card_res = []
            for prop in self.kb.most_general_object_properties(domain=current_domain):
                max_ = self.max_nr_fillers[prop]
                if max_ > 1 or (self.use_negation and max_ > 0):
                    card_res.append(self.generator.max_cardinality_restriction(self.generator.thing, prop, max_ - 1))
            iter_container.append(card_res)

        # a, b = tee(chain.from_iterable(iter_container))
        refs = []
        for i in chain.from_iterable(iter_container):
            yield i
            refs.append(i)

        # Compute all possible combinations of the disjunction and conjunctions.
        mem = set()
        for i in refs:
            # assert i is not None
            # yield i
            i_inds = None
            for j in refs:
                # assert j is not None
                if (i, j) in mem or i == j:
                    continue
                mem.add((j, i))
                mem.add((i, j))
                length = self.len(i) + self.len(j) + 1

                if (max_length >= length) and (self.max_child_length >= length + 1):
                    if not i.is_owl_thing() and not j.is_owl_thing():
                        # TODO: remove individuals_set calls
                        if i_inds is None:
                            i_inds = self.kb.individuals_set(i)
                        j_inds = self.kb.individuals_set(j)
                        if not j_inds.difference(i_inds):
                            # already contained
                            continue
                        else:
                            yield self.generator.union((i, j))
                        # if self.kb.individuals_count(temp_union) < self.kb.individuals_count():
                        #     yield temp_union

                        if not j_inds.intersection(i_inds):
                            # empty
                            continue
                        else:
                            yield self.generator.intersection((i, j))
                        # temp_intersection = self.kb.intersection((i, j))
                        # if self.kb.individuals_count(temp_intersection) > 0:

    def refine_complement_of(self, ce: OWLObjectComplementOf) -> Iterable[OWLClassExpression]:
        """ Refine owl:complementOf.

        Args:
            ce (OWLObjectComplementOf): owl:complementOf - class expression.

        Returns:
            Iterable of refined concepts.
        """
        assert isinstance(ce, OWLObjectComplementOf)

        if self.use_negation:
            parents = self.kb.get_direct_parents(ce.get_operand())
            yield from self.generator.negation_from_iterables(parents)
        else:
            yield from {}

    def refine_object_some_values_from(self, ce: OWLObjectSomeValuesFrom, max_length: int) \
            -> Iterable[OWLClassExpression]:
        """ Refine owl:someValuesFrom.

        Args:
            ce (OWLObjectSomeValuesFrom): owl:someValuesFrom class expression.
            max_length (int): Refine up to this concept length.

        Returns:
            Iterable of refined concepts.
        """
        assert isinstance(ce, OWLObjectSomeValuesFrom)
        assert isinstance(ce.get_filler(), OWLClassExpression)

        # rule 1: EXISTS r.D = > EXISTS r.E
        domain = self._get_current_domain(ce.get_property())
        for i in self.refine(ce.get_filler(), max_length=max_length - 2, current_domain=domain):
            if i is not None:
                yield self.generator.existential_restriction(i, ce.get_property())

        for more_special_op in self.kb.object_property_hierarchy. \
                more_special_roles(ce.get_property().get_named_property()):
            yield self.generator.existential_restriction(ce.get_filler(), more_special_op)

        if self.use_all_constructor:
            yield self.generator.universal_restriction(ce.get_filler(), ce.get_property())

        length = self.len(ce)
        if self.use_card_restrictions and length < max_length and \
                length < self.max_child_length and self.max_nr_fillers[ce.get_property()] > 1:
            yield self.generator.min_cardinality_restriction(ce.get_filler(), ce.get_property(), 2)

    def refine_object_all_values_from(self, ce: OWLObjectAllValuesFrom, max_length: int) \
            -> Iterable[OWLObjectAllValuesFrom]:
        """Refine owl:allValuesFrom.

        Args:
            ce (OWLObjectAllValuesFrom): owl:allValuesFrom - class expression.
            max_length (int): Refine up to this concept length.

        Returns:
            Iterable of refined concepts.
        """
        assert isinstance(ce, OWLObjectAllValuesFrom)

        if self.use_all_constructor:
            # rule 1: Forall r.D = > Forall r.E
            domain = self._get_current_domain(ce.get_property())
            for i in self.refine(ce.get_filler(), max_length=max_length - 2, current_domain=domain):
                if i is not None:
                    yield self.generator.universal_restriction(i, ce.get_property())
            # if not concept.get_filler().is_owl_nothing() and concept.get_filler().isatomic and (len(refs) == 0):
            #    # TODO find a way to include nothing concept
            #    refs.update(self.kb.universal_restriction(i, concept.get_property()))
            for more_special_op in self.kb.object_property_hierarchy. \
                    more_special_roles(ce.get_property().get_named_property()):
                yield self.generator.universal_restriction(ce.get_filler(), more_special_op)
        else:
            yield from {}

    def refine_object_min_card_restriction(self, ce: OWLObjectMinCardinality, max_length: int) \
            -> Iterable[OWLObjectMinCardinality]:
        """Refine owl:minCardinality.

        Args:
            ce (OWLObjectMinCardinality): owl:minCardinality - class expression.
            max_length (int): Refine up to this concept length.

        Returns:
            Iterable of refined concepts.
        """
        assert isinstance(ce, OWLObjectMinCardinality)
        assert ce.get_cardinality() >= 0

        domain = self._get_current_domain(ce.get_property())
        for i in self.refine(ce.get_filler(), max_length=max_length - 3, current_domain=domain):
            if i is not None:
                yield self.generator.min_cardinality_restriction(i, ce.get_property(), ce.get_cardinality())

        if ce.get_cardinality() < self.max_nr_fillers[ce.get_property()]:
            yield self.generator.min_cardinality_restriction(ce.get_filler(), ce.get_property(),
                                                             ce.get_cardinality() + 1)

    def refine_object_max_card_restriction(self, ce: OWLObjectMaxCardinality, max_length: int) \
            -> Iterable[OWLObjectMaxCardinality]:
        """Refine owl:maxCardinality.

        Args:
            ce (OWLObjectMaxCardinality): owl:maxCardinality - class expression.
            max_length (int): Refine up to this concept length.

        Returns:
            Iterable of refined concepts.
        """
        assert isinstance(ce, OWLObjectMaxCardinality)
        assert ce.get_cardinality() >= 0

        domain = self._get_current_domain(ce.get_property())
        for i in self.refine(ce.get_filler(), max_length=max_length - 3, current_domain=domain):
            if i is not None:
                yield self.generator.max_cardinality_restriction(i, ce.get_property(), ce.get_cardinality())

        if ce.get_cardinality() > 1 or (self.use_negation and ce.get_cardinality() > 0):
            yield self.generator.max_cardinality_restriction(ce.get_filler(), ce.get_property(),
                                                             ce.get_cardinality() - 1)

    def refine_object_union_of(self, ce: OWLObjectUnionOf, max_length: int,
                               current_domain: Optional[OWLClassExpression]) -> Iterable[OWLObjectUnionOf]:
        """Refine owl:unionOf.

        Given a node corresponding a concepts that comprises union operation:
        1) Obtain two concepts A, B,
        2) Refine A and union refinements with B,
        3) Repeat (2) for B.

        Args:
            ce (OWLObjectUnionOf): owl:unionOf - class expression.
            current_domain (OWLClassExpression): Domain.
            max_length (int): Refine up to this concept length.

        Returns:
            Iterable of refined concepts.
        """
        assert isinstance(ce, OWLObjectUnionOf)

        operands: List[OWLClassExpression] = list(ce.operands())
        for i in range(len(operands)):
            concept_left, concept, concept_right = operands[:i], operands[i], operands[i + 1:]
            concept_length = self.len(concept)

            for ref_concept in self.refine(concept,
                                           max_length=max_length - self.len(ce) + concept_length,
                                           current_domain=current_domain):
                union = self.generator.union(concept_left + [ref_concept] + concept_right)
                if max_length >= self.len(union):
                    yield union

    def refine_object_intersection_of(self, ce: OWLObjectIntersectionOf, max_length: int,
                                      current_domain: Optional[OWLClassExpression]) \
            -> Iterable[OWLObjectIntersectionOf]:
        """Refine owl:intersectionOf.

        Args:
            ce (OWLObjectIntersectionOf): owl:intersectionOf - class expression.
            current_domain (int): Domain.

        Returns:
            Iterable of refined concepts.
        """
        assert isinstance(ce, OWLObjectIntersectionOf)
        # TODO: Add sanity check method for intersections as in DL-Learner?

        operands: List[OWLClassExpression] = list(ce.operands())
        for i in range(len(operands)):
            concept_left, concept, concept_right = operands[:i], operands[i], operands[i + 1:]
            concept_length = self.len(concept)

            for ref_concept in self.refine(concept,
                                           max_length=max_length - self.len(ce) + concept_length,
                                           current_domain=current_domain):
                intersection = self.generator.intersection(concept_left + [ref_concept] + concept_right)
                if max_length >= self.len(ref_concept):
                    # if other_concept.instances.isdisjoint(ref_concept.instances):
                    #    continue
                    yield intersection

    def refine_data_some_values_from(self, ce: OWLDataSomeValuesFrom) -> Iterable[OWLDataSomeValuesFrom]:
        """Refine owl:someValuesFrom for data properties.

        Args:
            ce (OWLDataSomeValuesFrom): owl:someValuesFrom - class expression.

        Returns:
            Iterable of refined concepts.
        """
        assert isinstance(ce, OWLDataSomeValuesFrom)
        datarange = ce.get_filler()
        if isinstance(datarange, OWLDatatypeRestriction) and ce.get_property() in self.dp_splits:
            splits = self.dp_splits[ce.get_property()]
            if len(splits) > 0 and len(datarange.get_facet_restrictions()) > 0:
                facet_res = datarange.get_facet_restrictions()[0]
                val = facet_res.get_facet_value()
                idx = splits.index(val)

                if facet_res.get_facet() == OWLFacet.MIN_INCLUSIVE and (next_idx := idx + 1) < len(splits):
                    yield self.generator.data_existential_restriction(
                        OWLDatatypeMinInclusiveRestriction(splits[next_idx]), ce.get_property())
                elif facet_res.get_facet() == OWLFacet.MAX_INCLUSIVE and (next_idx := idx - 1) >= 0:
                    yield self.generator.data_existential_restriction(
                        OWLDatatypeMaxInclusiveRestriction(splits[next_idx]), ce.get_property())

    def refine_data_has_value(self, ce: OWLDataHasValue) -> Iterable[OWLDataHasValue]:
        """ Refine owl:hasValue.

        Args:
            ce (OWLDataHasValue): owl:hasValue - class expression.

        Returns:
            Iterable of refined concepts.
        """
        assert isinstance(ce, OWLDataHasValue)

        for more_special_dp in self.kb.data_property_hierarchy.more_special_roles(ce.get_property()):
            yield self.generator.data_has_value_restriction(ce.get_filler(), more_special_dp)

    def refine(self, ce: OWLClassExpression, max_length: int, current_domain: Optional[OWLClassExpression] = None) \
            -> Iterable[OWLClassExpression]:
        """Refine a given concept.

        Args:
            ce: Concept to refine.
            max_length: Refine up to this concept length.
            current_domain: Domain.

        Returns:
            Iterable of refined concepts.
        """
        assert isinstance(ce, OWLClassExpression)
        if isinstance(ce, OWLClass):
            yield from self.refine_atomic_concept(ce, max_length, current_domain)
        elif isinstance(ce, OWLObjectComplementOf):
            yield from self.refine_complement_of(ce)
        elif isinstance(ce, OWLObjectSomeValuesFrom):
            yield from self.refine_object_some_values_from(ce, max_length)
        elif isinstance(ce, OWLObjectAllValuesFrom):
            yield from self.refine_object_all_values_from(ce, max_length)
        elif isinstance(ce, OWLObjectMinCardinality):
            yield from self.refine_object_min_card_restriction(ce, max_length)
        elif isinstance(ce, OWLObjectMaxCardinality):
            yield from self.refine_object_max_card_restriction(ce, max_length)
        elif isinstance(ce, OWLObjectUnionOf):
            yield from self.refine_object_union_of(ce, max_length, current_domain)
        elif isinstance(ce, OWLObjectIntersectionOf):
            yield from self.refine_object_intersection_of(ce, max_length, current_domain)
        elif isinstance(ce, OWLDataSomeValuesFrom):
            yield from self.refine_data_some_values_from(ce)
        elif isinstance(ce, OWLDataHasValue):
            yield from self.refine_data_has_value(ce)
        else:
            raise ValueError(f"{type(ce)} objects are not yet supported")


class ExpressRefinement(ModifiedCELOERefinement):
    """ A top-down refinement operator in ALCHIQ(D)."""

    __slots__ = 'expressivity', 'downsample', 'sample_fillers_count', 'generator'

    expressivity: float
    downsample: bool
    sample_fillers_count: int
    generator: ConceptGenerator

    def __init__(self, knowledge_base,
                 downsample: bool = True,
                 expressivity: float = 0.8,
                 sample_fillers_count: int = 5,
                 value_splitter: Optional[AbstractValueSplitter] = None,
                 max_child_length: int = 10,
                 use_inverse: bool = True,
                 use_card_restrictions: bool = True,
                 use_numeric_datatypes: bool = True,
                 use_time_datatypes: bool = True,
                 use_boolean_datatype: bool = True,
                 card_limit: int = 10):
        self.downsample = downsample
        self.sample_fillers_count = sample_fillers_count
        self.expressivity = expressivity
        self.generator = ConceptGenerator()
        super().__init__(knowledge_base,
                         value_splitter=value_splitter,
                         max_child_length=max_child_length,
                         use_inverse=use_inverse,
                         use_card_restrictions=use_card_restrictions,
                         use_numeric_datatypes=use_numeric_datatypes,
                         use_time_datatypes=use_time_datatypes,
                         use_boolean_datatype=use_boolean_datatype,
                         card_limit=card_limit)
        self._setup()

    def refine_atomic_concept(self, ce: OWLClass) -> Iterable[OWLClassExpression]:
        """Refine atomic concept.
        Args:
            ce: Atomic concept to refine.

        Returns:
            Iterable of refined concepts.
        """
        if ce.is_owl_nothing():
            yield OWLNothing
        else:
            any_refinement = False
            # Get all subconcepts
            iter_container_sub = list(self.kb.get_all_sub_concepts(ce))
            if len(iter_container_sub) == 0:
                iter_container_sub = [ce]
            iter_container_restrict = []
            # Get negations of all subconcepts
            iter_container_neg = list(self.generator.negation_from_iterables(iter_container_sub))
            # (3) Create ∀.r.C and ∃.r.C where r is the most general relation and C in Fillers
            fillers: Set[OWLClassExpression] = {OWLThing, OWLNothing}
            if len(iter_container_sub) >= self.sample_fillers_count:
                fillers = fillers | set(random.sample(iter_container_sub, k=self.sample_fillers_count)) | \
                          set(random.sample(iter_container_neg, k=self.sample_fillers_count))
            for c in fillers:
                if self.len(c) + 2 <= self.max_child_length:
                    iter_container_restrict.append(
                        set(self.kb.most_general_universal_restrictions(domain=ce, filler=c)))
                    iter_container_restrict.append(
                        set(self.kb.most_general_existential_restrictions(domain=ce, filler=c)))
                    if self.use_inverse:
                        iter_container_restrict.append(
                            set(self.kb.most_general_existential_restrictions_inverse(domain=ce, filler=c)))
                        iter_container_restrict.append(
                            set(self.kb.most_general_universal_restrictions_inverse(domain=ce, filler=c)))

            if self.use_numeric_datatypes:
                iter_container_restrict.append(set(self._get_dp_restrictions(
                    self.kb.most_general_numeric_data_properties(domain=ce))))
            if self.use_time_datatypes:
                iter_container_restrict.append(set(self._get_dp_restrictions(
                    self.kb.most_general_time_data_properties(domain=ce))))
            if self.use_boolean_datatype:
                bool_res = []
                for bool_dp in self.kb.most_general_boolean_data_properties(domain=ce):
                    bool_res.append(self.generator.data_has_value_restriction(value=OWLLiteral(True), property=bool_dp))
                    bool_res.append(self.generator.data_has_value_restriction(value=OWLLiteral(False),
                                                                              property=bool_dp))
                iter_container_restrict.append(set(bool_res))

            if self.use_card_restrictions and (self.max_child_length >= self.len(ce) + 3):
                card_res = []
                for prop in self.kb.most_general_object_properties(domain=ce):
                    max_ = self.max_nr_fillers[prop]
                    if max_ > 1:
                        card_res.append(self.generator.max_cardinality_restriction(self.generator.thing,
                                                                                   prop, max_ - 1))
                iter_container_restrict.append(set(card_res))

            iter_container_restrict = list(set(chain.from_iterable(iter_container_restrict)))
            container = iter_container_restrict + iter_container_neg + iter_container_sub
            if self.downsample:  # downsampling is necessary if no enough computation resources
                assert self.expressivity < 1, "When downsampling, the expressivity must be less than 1"
                m = int(self.expressivity * len(container))
                container = random.sample(container, k=max(m, 1))
            else:
                self.expressivity = 1.
            if ce.is_owl_thing():  # If this is satisfied then all possible refinements are subconcepts
                if iter_container_neg + iter_container_restrict:
                    any_refinement = True
                    yield from iter_container_neg + iter_container_restrict
            del iter_container_restrict, iter_container_neg
            # Yield all subconcepts
            if iter_container_sub:
                any_refinement = True
                yield from iter_container_sub
            for sub in iter_container_sub:
                for other_ref in container:
                    if sub != other_ref and self.len(sub) + self.len(other_ref) < self.max_child_length:
                        if ce.is_owl_thing() or (other_ref in iter_container_sub):
                            union = self.generator.union([sub, other_ref])
                            yield union
                            any_refinement = True
                        elif other_ref not in iter_container_sub:
                            union = self.generator.union([sub, other_ref])
                            union = self.generator.intersection([ce, union])
                            if self.len(union) <= self.max_child_length:
                                yield union
                                any_refinement = True
                        intersect = self.generator.intersection([sub, other_ref])
                        if self.len(intersect) <= self.max_child_length:
                            yield intersect
                            any_refinement = True
            if not any_refinement:
                yield ce

    def refine_complement_of(self, ce: OWLObjectComplementOf) -> Iterable[OWLClassExpression]:
        """ Refine owl:complementOf.

        Args:
            ce (OWLObjectComplementOf): owl:complementOf - class expression.

        Returns:
            Iterable of refined concepts.
        """
        assert isinstance(ce, OWLObjectComplementOf)
        any_refinement = False
        parents = self.kb.get_direct_parents(self.generator.negation(ce))
        for ref in self.generator.negation_from_iterables(parents):
            if self.len(ref) <= self.max_child_length:
                any_refinement = True
                yield ref
        if not any_refinement:
            yield ce

    def refine_object_some_values_from(self, ce: OWLObjectSomeValuesFrom) -> Iterable[OWLClassExpression]:
        """ Refine owl:someValuesFrom.

        Args:
            ce (OWLObjectSomeValuesFrom): owl:someValuesFrom class expression.

        Returns:
            Iterable of refined concepts.
        """
        assert isinstance(ce, OWLObjectSomeValuesFrom)
        assert isinstance(ce.get_filler(), OWLClassExpression)
        any_refinement = False
        for ref in self.refine(ce.get_filler()):
            if 2 + self.len(ref) <= self.max_child_length:
                any_refinement = True
                reft = self.generator.existential_restriction(ref, ce.get_property())
                yield reft
        if self.len(ce) <= self.max_child_length:
            any_refinement = True
            reft = self.generator.universal_restriction(ce.get_filler(), ce.get_property())
            yield reft

        for more_special_op in self.kb.object_property_hierarchy. \
                more_special_roles(ce.get_property().get_named_property()):
            if self.len(ce) <= self.max_child_length:
                yield self.generator.existential_restriction(ce.get_filler(), more_special_op)
                any_refinement = True

        if self.use_card_restrictions and self.len(ce) <= self.max_child_length and \
                self.max_nr_fillers[ce.get_property()] > 1:
            yield self.generator.min_cardinality_restriction(ce.get_filler(), ce.get_property(), 2)
            any_refinement = True
        if not any_refinement:
            yield ce

    def refine_object_all_values_from(self, ce: OWLObjectAllValuesFrom) -> Iterable[OWLClassExpression]:
        """Refine owl:allValuesFrom.

        Args:
            ce (OWLObjectAllValuesFrom): owl:allValuesFrom - class expression.

        Returns:
            Iterable of refined concepts.
        """
        assert isinstance(ce, OWLObjectAllValuesFrom)
        assert isinstance(ce.get_filler(), OWLClassExpression)
        any_refinement = False
        for ref in self.refine(ce.get_filler()):
            if 2 + self.len(ref) <= self.max_child_length:
                any_refinement = True
                reft = self.generator.universal_restriction(ref, ce.get_property())
                yield reft
        for more_special_op in self.kb.object_property_hierarchy. \
                more_special_roles(ce.get_property().get_named_property()):
            if 2 + self.len(ce.get_filler()) <= self.max_child_length:
                yield self.generator.universal_restriction(ce.get_filler(), more_special_op)
                any_refinement = True
        if not any_refinement and not ce.get_filler().is_owl_nothing():
            yield ce
        elif not any_refinement and ce.get_filler().is_owl_nothing():
            yield OWLNothing

    def refine_object_min_card_restriction(self, ce: OWLObjectMinCardinality) \
            -> Iterable[OWLObjectMinCardinality]:
        """Refine owl:allValuesFrom.

        Args:
            ce (OWLObjectAllValuesFrom): owl:allValuesFrom - class expression.

        Returns:
            Iterable of refined concepts.
        """
        assert isinstance(ce, OWLObjectMinCardinality)
        assert ce.get_cardinality() >= 0

        for ref in self.refine(ce.get_filler()):
            if ref is not None:
                yield self.generator.min_cardinality_restriction(ref, ce.get_property(), ce.get_cardinality())

        if self.use_card_restrictions and ce.get_cardinality() < self.max_nr_fillers[ce.get_property()]:
            yield self.generator.min_cardinality_restriction(ce.get_filler(), ce.get_property(),
                                                             ce.get_cardinality() + 1)

    def refine_object_max_card_restriction(self, ce: OWLObjectMaxCardinality) \
            -> Iterable[OWLObjectMaxCardinality]:
        """Refine owl:maxCardinality.

        Args:
            ce (OWLObjectMaxCardinality): owl:maxCardinality - class expression.

        Returns:
            Iterable of refined concepts.
        """
        assert isinstance(ce, OWLObjectMaxCardinality)
        assert ce.get_cardinality() >= 0

        for ref in self.refine(ce.get_filler()):
            if ref is not None:
                yield self.generator.max_cardinality_restriction(ref, ce.get_property(), ce.get_cardinality())

        if ce.get_cardinality() > 1:
            yield self.generator.max_cardinality_restriction(ce.get_filler(), ce.get_property(),
                                                             ce.get_cardinality() - 1)

    def refine_object_union_of(self, ce: OWLObjectUnionOf) -> Iterable[OWLClassExpression]:
        """Refine owl:unionOf.

        Args:
            ce (OWLObjectUnionOf): owl:unionOf - class expression.

        Returns:
            Iterable of refined concepts.
        """
        assert isinstance(ce, OWLObjectUnionOf)
        any_refinement = False
        for op in ce.operands():
            if self.len(op) <= self.max_child_length:
                yield op
                any_refinement = True
        operands = list(ce.operands())
        for i in range(len(operands)):
            ce_left, ce_, ce_right = operands[:i], operands[i], operands[i + 1:]
            other_length = self._operands_len(OWLObjectUnionOf, ce_left + ce_right)
            for ref_ce in self.refine(ce_):
                if self.max_child_length >= other_length + self.len(ref_ce):
                    yield self.generator.union(ce_left + [ref_ce] + ce_right)
                    any_refinement = True
        if not any_refinement:
            yield ce

    def refine_object_intersection_of(self, ce: OWLObjectIntersectionOf) -> Iterable[OWLClassExpression]:
        """Refine owl:intersectionOf.

        Args:
            ce (OWLObjectIntersectionOf): owl:intersectionOf - class expression.

        Returns:
            Iterable of refined concepts.
        """
        assert isinstance(ce, OWLObjectIntersectionOf)
        any_refinement = False
        operands = list(ce.operands())
        for i in range(len(operands)):
            ce_left, ce, ce_right = operands[:i], operands[i], operands[i + 1:]
            other_length = self._operands_len(OWLObjectIntersectionOf, ce_left + ce_right)
            for ref_ce in self.refine(ce):
                if self.max_child_length >= other_length + self.len(ref_ce):
                    yield self.generator.intersection(ce_left + [ref_ce] + ce_right)
                    any_refinement = True
        if not any_refinement:
            yield ce

    def refine_data_some_values_from(self, ce: OWLDataSomeValuesFrom) -> Iterable[OWLDataSomeValuesFrom]:
        """Refine owl:someValuesFrom for data properties.

        Args:
            ce (OWLDataSomeValuesFrom): owl:someValuesFrom - class expression.

        Returns:
            Iterable of refined concepts.
        """
        assert isinstance(ce, OWLDataSomeValuesFrom)
        any_refinement = False
        datarange = ce.get_filler()
        if isinstance(datarange, OWLDatatypeRestriction) and ce.get_property() in self.dp_splits:
            splits = self.dp_splits[ce.get_property()]
            if len(splits) > 0 and len(datarange.get_facet_restrictions()) > 0:
                facet_res = datarange.get_facet_restrictions()[0]
                val = facet_res.get_facet_value()
                idx = splits.index(val)

                if facet_res.get_facet() == OWLFacet.MIN_INCLUSIVE and (next_idx := idx + 1) < len(splits):
                    yield self.generator.data_existential_restriction(
                        OWLDatatypeMinInclusiveRestriction(splits[next_idx]), ce.get_property())
                    any_refinement = True
                elif facet_res.get_facet() == OWLFacet.MAX_INCLUSIVE and (next_idx := idx - 1) >= 0:
                    yield self.generator.data_existential_restriction(
                        OWLDatatypeMaxInclusiveRestriction(splits[next_idx]), ce.get_property())
                    any_refinement = True
        if not any_refinement:
            yield ce

    def refine_data_has_value(self, ce: OWLDataHasValue) -> Iterable[OWLDataHasValue]:
        """ Refine owl:hasValue.

        Args:
            ce (OWLDataHasValue): owl:hasValue - class expression.

        Returns:
            Iterable of refined concepts.
        """
        assert isinstance(ce, OWLDataHasValue)
        any_refinement = False
        for more_special_dp in self.kb.data_property_hierarchy.more_special_roles(ce.get_property()):
            yield self.generator.data_has_value_restriction(ce.get_filler(), more_special_dp)
            any_refinement = True

        if not any_refinement:
            yield ce

    def refine(self, ce, **kwargs) -> Iterable[OWLClassExpression]:
        """Refine a given concept.

        Args:
            ce: Concept to refine

        Returns:
            Iterable of refined concepts.
        """
        # we ignore additional arguments like "max_length" or "current_domain" that might be supplied by the learning
        # algorithm by using **kwargs
        assert isinstance(ce, OWLClassExpression)
        if self.len(ce) == 1:
            yield from self.refine_atomic_concept(ce)
        elif isinstance(ce, OWLObjectComplementOf):
            yield from self.refine_complement_of(ce)
        elif isinstance(ce, OWLObjectSomeValuesFrom):
            yield from self.refine_object_some_values_from(ce)
        elif isinstance(ce, OWLObjectAllValuesFrom):
            yield from self.refine_object_all_values_from(ce)
        elif isinstance(ce, OWLObjectMinCardinality):
            yield from self.refine_object_min_card_restriction(ce)
        elif isinstance(ce, OWLObjectMaxCardinality):
            yield from self.refine_object_max_card_restriction(ce)
        elif isinstance(ce, OWLObjectUnionOf):
            yield from self.refine_object_union_of(ce)
        elif isinstance(ce, OWLObjectIntersectionOf):
            yield from self.refine_object_intersection_of(ce)
        elif isinstance(ce, OWLDataSomeValuesFrom):
            yield from self.refine_data_some_values_from(ce)
        elif isinstance(ce, OWLDataHasValue):
            yield from self.refine_data_has_value(ce)
        else:
            raise ValueError(f"{type(ce)} objects are not yet supported")
