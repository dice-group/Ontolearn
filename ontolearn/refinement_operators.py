"""Refinement Operators for refinement-based concept learners."""
from collections import defaultdict
from itertools import chain
import random
from typing import DefaultDict, Dict, Set, Optional, Iterable, List, Type, Final, Generator

from owlapy.class_expression import OWLObjectSomeValuesFrom, OWLObjectAllValuesFrom, OWLObjectIntersectionOf, \
    OWLClassExpression, OWLNothing, OWLThing, OWLNaryBooleanClassExpression, OWLObjectUnionOf, OWLClass, \
    OWLObjectComplementOf, OWLObjectMaxCardinality, OWLObjectMinCardinality, OWLDataSomeValuesFrom, \
    OWLDatatypeRestriction, OWLDataHasValue, OWLObjectExactCardinality, OWLObjectHasValue, OWLObjectOneOf
from owlapy.owl_individual import OWLIndividual
from owlapy.owl_literal import OWLLiteral
from owlapy.owl_property import OWLObjectPropertyExpression, OWLObjectInverseOf, OWLDataProperty, \
    OWLDataPropertyExpression, OWLObjectProperty

from ontolearn.value_splitter import AbstractValueSplitter, BinningValueSplitter
from owlapy.providers import owl_datatype_max_inclusive_restriction, owl_datatype_min_inclusive_restriction
from owlapy.vocab import OWLFacet

from .abstracts import BaseRefinement
from .concept_generator import ConceptGenerator
from .knowledge_base import KnowledgeBase
from .search import OENode
from typing import Tuple
import itertools


class LengthBasedRefinement(BaseRefinement):
    """ A top-down length based ("no semantic information leveraged) refinement operator in ALC."""

    def __init__(self, knowledge_base: KnowledgeBase,
                 use_inverse: bool = True,
                 use_data_properties: bool = False,
                 use_card_restrictions: bool = True,
                 use_nominals: bool = True):
        super().__init__(knowledge_base)

        self.use_inverse = use_inverse
        self.use_data_properties = use_data_properties
        self.use_card_restrictions = use_card_restrictions
        self.use_nominals = use_nominals
        self.top_refinements: set = None
        self.pos = None
        self.neg = None

    def set_input_examples(self, pos, neg):
        # TODO: Later, depending on pos and neg, we will not return some refinements
        self.pos = {i for i in pos}
        self.neg = {i for i in neg}

    def refine_top(self) -> Iterable:
        """ Refine Top Class Expression

        rho(T)

        1- Named concepts

        2- Negated leaf Concepts if max_len_refinement_top >2

        3- Union of (1) if max_len_refinement_top>=3

        4- Intersection of not disjoint of (1) if max_len_refinement_top>=3

        5) Restrictions:   \forall \exist R (1)
                           \forall \exist R neg (1)
                           \forall \exist R⁻ (1)
                           \forall \exist R⁻ (1)

        """
        # (1) Return all named concepts:Later get most general  classes
        most_general_concepts = [i for i in self.kb.get_most_general_classes()]
        yield from most_general_concepts

        # (2) OWLDataSomeValuesFrom over double values fillers
        # Two ce for each property returned. Mean value extracted-
        # TODO: Most general_double_data_pro
        if not isinstance(self.kb, KnowledgeBase):
            for i in self.kb.get_double_data_properties():
                doubles = [i.parse_double() for i in self.kb.get_range_of_double_data_properties(i)]
                mean_doubles = sum(doubles) / len(doubles)
                yield OWLDataSomeValuesFrom(property=i,
                                            filler=owl_datatype_min_inclusive_restriction(
                                                min_=OWLLiteral(mean_doubles)))
                yield OWLDataSomeValuesFrom(property=i,
                                            filler=owl_datatype_max_inclusive_restriction(
                                                max_=OWLLiteral(mean_doubles)))
        # (3) Boolean Valued OWLDataHasValue: TODO: Most general_boolean_data_pro
        for i in self.kb.get_boolean_data_properties():
            yield OWLDataHasValue(property=i, value=OWLLiteral(True))
            yield OWLDataHasValue(property=i, value=OWLLiteral(False))

        # (4) Return least general concepts.
        neg_concepts = [OWLObjectComplementOf(i) for i in self.kb.least_general_named_concepts()]
        yield from neg_concepts
        yield from self.from_iterables(cls=OWLObjectUnionOf,
                                       a_operands=most_general_concepts,
                                       b_operands=most_general_concepts)
        yield from self.from_iterables(cls=OWLObjectUnionOf, a_operands=most_general_concepts, b_operands=neg_concepts)

        restrictions = []
        for c in most_general_concepts + [OWLThing, OWLNothing] + neg_concepts:
            dl_role: OWLObjectProperty
            for dl_role in self.kb.get_object_properties():
                # TODO: Check whether the range of OWLObjectProperty contains the respective ce.
                restrictions.append(OWLObjectSomeValuesFrom(filler=c, property=dl_role))
                restrictions.append(OWLObjectAllValuesFrom(filler=c, property=dl_role))
                if self.use_inverse:
                    # TODO: Check whether we can only invert the most specific object properties.
                    inverse_role = dl_role.get_inverse_property()
                    restrictions.append(OWLObjectSomeValuesFrom(filler=c, property=inverse_role))
                    restrictions.append(OWLObjectAllValuesFrom(filler=c, property=inverse_role))

                # Move the card limit into existantial restrictions.
                if self.use_card_restrictions:
                    for card in range(1, 2):
                        temp_res = [OWLObjectMinCardinality(cardinality=card,
                                                            property=dl_role,
                                                            filler=c)]
                        if self.use_inverse:
                            temp_res.extend([OWLObjectMinCardinality(cardinality=card,
                                                                     property=inverse_role,
                                                                     filler=c
                                                                     )])
                        restrictions.extend(temp_res)
                    del temp_res
        yield from restrictions

    def refine_atomic_concept(self, class_expression: OWLClass) -> Generator[
        Tuple[OWLObjectIntersectionOf, OWLObjectOneOf], None, None]:
        assert isinstance(class_expression, OWLClass), class_expression
        for i in self.top_refinements:
            if i.is_owl_nothing() is False:
                # TODO: Include are_owl_concept_disjoint into Knowledgebase class
                if isinstance(i, OWLClass):  #:and self.kb.are_owl_concept_disjoint(class_expression, i) is False:
                    yield OWLObjectIntersectionOf((class_expression, i))
                else:
                    yield OWLObjectIntersectionOf((class_expression, i))

    def refine_complement_of(self, class_expression: OWLObjectComplementOf) -> Generator[
        OWLObjectComplementOf, None, None]:
        assert isinstance(class_expression, OWLObjectComplementOf)
        # not Father => Not Person given Father subclass of Person
        yield from (OWLObjectComplementOf(i) for i in self.kb.get_direct_parents(class_expression.get_operand()))
        yield OWLObjectIntersectionOf((class_expression, OWLThing))

    def refine_object_some_values_from(self, class_expression: OWLObjectSomeValuesFrom) -> Iterable[OWLClassExpression]:
        assert isinstance(class_expression, OWLObjectSomeValuesFrom)
        # Given \exists r. C
        yield OWLObjectIntersectionOf((class_expression, OWLThing))
        yield from (OWLObjectSomeValuesFrom(filler=C,
                                            property=class_expression.get_property()) for C in
                    self.refine(class_expression.get_filler()))

    def refine_object_all_values_from(self, class_expression: OWLObjectAllValuesFrom) -> Iterable[OWLClassExpression]:
        assert isinstance(class_expression, OWLObjectAllValuesFrom)
        yield OWLObjectIntersectionOf((class_expression, OWLThing))
        yield from (OWLObjectAllValuesFrom(filler=C,
                                           property=class_expression.get_property()) for C in
                    self.refine(class_expression.get_filler()))

    def refine_object_union_of(self, class_expression: OWLObjectUnionOf) -> Iterable[OWLClassExpression]:
        """ Refine OWLObjectUnionof by refining each operands:"""
        assert isinstance(class_expression, OWLObjectUnionOf)
        operands: List[OWLClassExpression] = list(class_expression.operands())
        # Refine each operant
        for i, concept in enumerate(operands):
            for refinement_of_concept in self.refine(concept):
                if refinement_of_concept == class_expression:
                    continue
                yield OWLObjectUnionOf(operands[:i] + [refinement_of_concept] + operands[i + 1:])

        yield OWLObjectIntersectionOf((class_expression, OWLThing))

    def refine_object_intersection_of(self, class_expression: OWLObjectIntersectionOf) -> Iterable[OWLClassExpression]:
        """ Refine OWLObjectIntersectionOf by refining each operands:"""
        assert isinstance(class_expression, OWLObjectIntersectionOf)
        operands: List[OWLClassExpression] = list(class_expression.operands())
        # Refine each operant
        for i, concept in enumerate(operands):
            for refinement_of_concept in self.refine(concept):
                if refinement_of_concept == class_expression:
                    continue
                yield OWLObjectIntersectionOf(operands[:i] + [refinement_of_concept] + operands[i + 1:])

        yield OWLObjectIntersectionOf((class_expression, OWLThing))

    def refine(self, class_expression) -> Iterable[OWLClassExpression]:
        assert isinstance(class_expression, OWLClassExpression)
        # (1) Initialize top refinement if it has not been initialized.
        if self.top_refinements is None:
            self.top_refinements = set()
            for i in self.refine_top():
                self.top_refinements.add(i)
                yield i
        if class_expression.is_owl_thing():
            yield from self.top_refinements
        elif isinstance(class_expression, OWLClass):
            yield from self.refine_atomic_concept(class_expression)
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
        elif isinstance(class_expression, OWLObjectMaxCardinality):
            yield from (OWLObjectIntersectionOf((class_expression, i)) for i in self.top_refinements)
        elif isinstance(class_expression, OWLObjectExactCardinality):
            yield from (OWLObjectIntersectionOf((class_expression, i)) for i in self.top_refinements)
        elif isinstance(class_expression, OWLObjectMinCardinality):
            yield from (OWLObjectIntersectionOf((class_expression, i)) for i in self.top_refinements)
        elif isinstance(class_expression, OWLDataSomeValuesFrom):
            """unclear how to refine OWLDataHasValue via refining a the property
            We may need to modify the literal little bit right little bit left fashion
            ∃ lumo.xsd:double[≤ -1.6669212962962956] 
            
            ∃ lumo.xsd:double[≥ -1.6669212962962956] 
            """
            yield from (OWLObjectIntersectionOf((class_expression, i)) for i in self.top_refinements)
        elif isinstance(class_expression, OWLDataHasValue):
            yield from (OWLObjectIntersectionOf((class_expression, i)) for i in self.top_refinements)
        elif isinstance(class_expression, OWLObjectOneOf):
            raise NotImplementedError("Remove an individual from the set of individuals, If empty use bottom.")
        else:
            raise ValueError(f"{type(class_expression)} objects are not yet supported")

    @staticmethod
    def from_iterables(cls, a_operands, b_operands):
        assert (isinstance(a_operands, Generator) is False) and (isinstance(b_operands, Generator) is False)
        seen = set()
        results = set()
        for i in a_operands:
            for j in b_operands:
                #if i == j:
                #    results.add(i)
                if (i, j) in seen:
                    continue
                else:
                    i_and_j = cls((i, j))
                    seen.add((i, j))
                    seen.add((j, i))
                    results.add(i_and_j)
        return results


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
                    filler=owl_datatype_min_inclusive_restriction(splits[0]), property=dp))
                restrictions.append(self.generator.data_existential_restriction(
                    filler=owl_datatype_max_inclusive_restriction(splits[-1]), property=dp))
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

        if self.use_negation:
            # TODO probably not correct/complete
            if max_length >= 2 and (self.len(ce) + 1 <= self.max_child_length):
                # (2.2) Create negation of all leaf_concepts
                iter_container.append(self.generator.negation_from_iterables(self.kb.get_leaf_concepts(ce)))

        if max_length >= 3 and (self.len(ce) + 2 <= self.max_child_length):
            # (2.3) Create ∀.r.T and ∃.r.T where r is the most general relation.
            iter_container.append(self.kb.most_general_existential_restrictions(domain=current_domain))
            if self.use_all_constructor:
                iter_container.append(self.kb.most_general_universal_restrictions(domain=current_domain))
            if self.use_inverse:
                iter_container.append(self.kb.most_general_existential_restrictions_inverse(domain=current_domain))
                if self.use_all_constructor:
                    iter_container.append(self.kb.most_general_universal_restrictions_inverse(domain=current_domain))
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

        if self.use_card_restrictions and max_length >= 4 and (self.max_child_length >= self.len(ce) + 3):
            card_res = []
            for prop in self.kb.most_general_object_properties(domain=current_domain):
                max_ = self.max_nr_fillers[prop]
                if max_ > 1 or (self.use_negation and max_ > 0):
                    card_res.append(self.generator.max_cardinality_restriction(self.generator.thing, prop, max_ - 1))
            iter_container.append(card_res)

        refs = []
        for i in chain.from_iterable(iter_container):
            yield i
            refs.append(i)

        # Compute all possible combinations of the disjunction and conjunctions.
        mem = set()
        for i in refs:
            # assert i is not None
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

                        if not j_inds.intersection(i_inds):
                            # empty
                            continue
                        else:
                            yield self.generator.intersection((i, j))

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
                        owl_datatype_min_inclusive_restriction(splits[next_idx]), ce.get_property())
                elif facet_res.get_facet() == OWLFacet.MAX_INCLUSIVE and (next_idx := idx - 1) >= 0:
                    yield self.generator.data_existential_restriction(
                        owl_datatype_max_inclusive_restriction(splits[next_idx]), ce.get_property())

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
                        owl_datatype_min_inclusive_restriction(splits[next_idx]), ce.get_property())
                    any_refinement = True
                elif facet_res.get_facet() == OWLFacet.MAX_INCLUSIVE and (next_idx := idx - 1) >= 0:
                    yield self.generator.data_existential_restriction(
                        owl_datatype_max_inclusive_restriction(splits[next_idx]), ce.get_property())
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
