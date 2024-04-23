"""Utils for mapping to and from owlready2."""
from datetime import date, datetime
from functools import singledispatchmethod
from types import MappingProxyType
from typing import Union

import owlready2
from owlapy.class_expression import OWLObjectOneOf, OWLClass, OWLObjectUnionOf, OWLObjectIntersectionOf, \
    OWLObjectSomeValuesFrom, OWLObjectComplementOf, OWLObjectAllValuesFrom, OWLDataSomeValuesFrom, \
    OWLDatatypeRestriction, OWLClassExpression, OWLDataAllValuesFrom, OWLDataHasValue, OWLDataOneOf, \
    OWLObjectMinCardinality, OWLObjectMaxCardinality, OWLObjectExactCardinality, \
    OWLObjectHasValue, OWLFacetRestriction, OWLObjectRestriction, OWLDataExactCardinality, OWLDataMaxCardinality, \
    OWLDataMinCardinality, OWLRestriction, OWLDataRestriction
from owlapy.iri import IRI
from owlapy.owl_axiom import OWLAnnotationProperty
from owlapy.owl_data_ranges import OWLDataRange, OWLDataComplementOf, OWLDataIntersectionOf, OWLDataUnionOf
from owlapy.owl_datatype import OWLDatatype
from owlapy.owl_individual import OWLNamedIndividual, OWLIndividual
from owlapy.owl_literal import OWLLiteral, IntegerOWLDatatype, DoubleOWLDatatype, BooleanOWLDatatype, DateOWLDatatype, \
    DateTimeOWLDatatype, DurationOWLDatatype, StringOWLDatatype
from owlapy.owl_object import OWLObject
from owlapy.owl_ontology import OWLOntology
from owlapy.owl_property import OWLObjectProperty, OWLDataProperty, OWLObjectPropertyExpression, OWLObjectInverseOf, \
    OWLDataPropertyExpression, OWLPropertyExpression

from pandas import Timedelta


from owlapy.vocab import OWLFacet

OWLREADY2_FACET_KEYS = MappingProxyType({
    OWLFacet.MIN_INCLUSIVE: "min_inclusive",
    OWLFacet.MIN_EXCLUSIVE: "min_exclusive",
    OWLFacet.MAX_INCLUSIVE: "max_inclusive",
    OWLFacet.MAX_EXCLUSIVE: "max_exclusive",
    OWLFacet.LENGTH: "length",
    OWLFacet.MIN_LENGTH: "min_length",
    OWLFacet.MAX_LENGTH: "max_length",
    OWLFacet.PATTERN: "pattern",
    OWLFacet.TOTAL_DIGITS: "total_digits",
    OWLFacet.FRACTION_DIGITS: "fraction_digits"
})


class ToOwlready2:
    __slots__ = '_world'

    _world: owlready2.World

    def __init__(self, world: owlready2.World):
        """Map owlapy model classes to owlready2.

        Args:
            world: Owlready2 World to use for mapping.
        """
        self._world = world

    @singledispatchmethod
    def map_object(self, o: OWLObject):
        """Map owlapy object classes."""
        raise NotImplementedError(f'don\'t know how to map {o}')

    @map_object.register
    def _(self, ce: OWLClassExpression) -> Union[owlready2.ClassConstruct, owlready2.ThingClass]:
        return self.map_concept(ce)

    @map_object.register
    def _(self, ont: OWLOntology) -> owlready2.namespace.Ontology:
        return self._world.get_ontology(
            ont.get_ontology_id().get_ontology_iri().as_str()
        )

    @map_object.register
    def _(self, ap: OWLAnnotationProperty) -> owlready2.annotation.AnnotationPropertyClass:
        return self._world[ap.str]

    # @TODO CD: map_object is buggy. and it can return None
    # single dispatch is still not implemented in mypy, see https://github.com/python/mypy/issues/2904
    @singledispatchmethod
    def map_concept(self, o: OWLClassExpression) \
            -> Union[owlready2.ClassConstruct, owlready2.ThingClass]:
        """Map owlapy concept classes."""
        raise NotImplementedError(o)

    @singledispatchmethod
    def _to_owlready2_property(self, p: OWLPropertyExpression) -> owlready2.Property:
        raise NotImplementedError(p)

    @_to_owlready2_property.register
    def _(self, p: OWLObjectInverseOf):
        p_x = self._to_owlready2_property(p.get_named_property())
        return owlready2.Inverse(p_x)

    @_to_owlready2_property.register
    def _(self, p: OWLObjectProperty) -> owlready2.prop.ObjectPropertyClass:
        return self._world[p.str]

    @_to_owlready2_property.register
    def _(self, p: OWLDataProperty) -> owlready2.prop.DataPropertyClass:
        return self._world[p.str]

    @singledispatchmethod
    def _to_owlready2_individual(self, i: OWLIndividual) -> owlready2.Thing:
        raise NotImplementedError(i)

    @_to_owlready2_individual.register
    def _(self, i: OWLNamedIndividual):
        return self._world[i.str]

    @map_concept.register
    def _(self, c: OWLClass) -> owlready2.ThingClass:
        x = self._world[c.str]
        try:
            assert x is not None
        except AssertionError:
            print(f"The world attribute{self._world} maps {c} into None")

        return x

    @map_concept.register
    def _(self, c: OWLObjectComplementOf) -> owlready2.class_construct.Not:
        return owlready2.Not(self.map_concept(c.get_operand()))

    @map_concept.register
    def _(self, ce: OWLObjectUnionOf) -> owlready2.class_construct.Or:
        return owlready2.Or(map(self.map_concept, ce.operands()))

    @map_concept.register
    def _(self, ce: OWLObjectIntersectionOf) -> owlready2.class_construct.And:
        return owlready2.And(map(self.map_concept, ce.operands()))

    @map_concept.register
    def _(self, ce: OWLObjectSomeValuesFrom) -> owlready2.class_construct.Restriction:
        prop = self._to_owlready2_property(ce.get_property())
        assert isinstance(ce.get_filler(),
                          OWLClassExpression), f"{ce.get_filler()} is not an OWL Class expression and cannot be serialized at the moment"
        return prop.some(self.map_concept(ce.get_filler()))

    @map_concept.register
    def _(self, ce: OWLObjectAllValuesFrom) -> owlready2.class_construct.Restriction:
        prop = self._to_owlready2_property(ce.get_property())
        return prop.only(self.map_concept(ce.get_filler()))

    @map_concept.register
    def _(self, ce: OWLObjectOneOf) -> owlready2.class_construct.OneOf:
        return owlready2.OneOf(list(map(self._to_owlready2_individual, ce.individuals())))

    @map_concept.register
    def _(self, ce: OWLObjectExactCardinality) -> owlready2.class_construct.Restriction:
        prop = self._to_owlready2_property(ce.get_property())
        return prop.exactly(ce.get_cardinality(), self.map_concept(ce.get_filler()))

    @map_concept.register
    def _(self, ce: OWLObjectMaxCardinality) -> owlready2.class_construct.Restriction:
        prop = self._to_owlready2_property(ce.get_property())
        return prop.max(ce.get_cardinality(), self.map_concept(ce.get_filler()))

    @map_concept.register
    def _(self, ce: OWLObjectMinCardinality) -> owlready2.class_construct.Restriction:
        prop = self._to_owlready2_property(ce.get_property())
        return prop.min(ce.get_cardinality(), self.map_concept(ce.get_filler()))

    @map_concept.register
    def _(self, ce: OWLObjectHasValue) -> owlready2.class_construct.Restriction:
        prop = self._to_owlready2_property(ce.get_property())
        return prop.value(self._to_owlready2_individual(ce.get_filler()))

    @map_concept.register
    def _(self, ce: OWLDataSomeValuesFrom) -> owlready2.class_construct.Restriction:
        prop = self._to_owlready2_property(ce.get_property())
        return prop.some(self.map_datarange(ce.get_filler()))

    @map_concept.register
    def _(self, ce: OWLDataAllValuesFrom) -> owlready2.class_construct.Restriction:
        prop = self._to_owlready2_property(ce.get_property())
        return prop.only(self.map_datarange(ce.get_filler()))

    @map_concept.register
    def _(self, ce: OWLDataExactCardinality) -> owlready2.class_construct.Restriction:
        prop = self._to_owlready2_property(ce.get_property())
        return prop.exactly(ce.get_cardinality(), self.map_datarange(ce.get_filler()))

    @map_concept.register
    def _(self, ce: OWLDataMaxCardinality) -> owlready2.class_construct.Restriction:
        prop = self._to_owlready2_property(ce.get_property())
        return prop.max(ce.get_cardinality(), self.map_datarange(ce.get_filler()))

    @map_concept.register
    def _(self, ce: OWLDataMinCardinality) -> owlready2.class_construct.Restriction:
        prop = self._to_owlready2_property(ce.get_property())
        return prop.min(ce.get_cardinality(), self.map_datarange(ce.get_filler()))

    @map_concept.register
    def _(self, ce: OWLDataHasValue) -> owlready2.class_construct.Restriction:
        prop = self._to_owlready2_property(ce.get_property())
        return prop.value(ce.get_filler().to_python())

    @singledispatchmethod
    def map_datarange(self, p: OWLDataRange) -> Union[owlready2.ClassConstruct, type]:
        """Map owlapy data range classes."""
        raise NotImplementedError(p)

    @map_datarange.register
    def _(self, p: OWLDataComplementOf) -> owlready2.class_construct.Not:
        return owlready2.Not(self.map_datarange(p.get_data_range()))

    @map_datarange.register
    def _(self, p: OWLDataUnionOf) -> owlready2.class_construct.Or:
        return owlready2.Or(map(self.map_datarange, p.operands()))

    @map_datarange.register
    def _(self, p: OWLDataIntersectionOf) -> owlready2.class_construct.And:
        return owlready2.And(map(self.map_datarange, p.operands()))

    @map_datarange.register
    def _(self, p: OWLDataOneOf) -> owlready2.class_construct.OneOf:
        return owlready2.OneOf([lit.to_python() for lit in p.operands()])

    @map_datarange.register
    def _(self, p: OWLDatatypeRestriction) -> owlready2.class_construct.ConstrainedDatatype:
        facet_args = dict()
        for facet_res in p.get_facet_restrictions():
            value = facet_res.get_facet_value().to_python()
            facet_key = OWLREADY2_FACET_KEYS[facet_res.get_facet()]
            facet_args[facet_key] = value
        return owlready2.ConstrainedDatatype(self.map_datarange(p.get_datatype()), **facet_args)

    @map_datarange.register
    def _(self, type_: OWLDatatype) -> type:
        if type_ == BooleanOWLDatatype:
            return bool
        elif type_ == DoubleOWLDatatype:
            return float
        elif type_ == IntegerOWLDatatype:
            return int
        elif type_ == StringOWLDatatype:
            return str
        elif type_ == DateOWLDatatype:
            return date
        elif type_ == DateTimeOWLDatatype:
            return datetime
        elif type_ == DurationOWLDatatype:
            return Timedelta
        else:
            raise ValueError(type_)


class FromOwlready2:
    """Map owlready2 classes to owlapy model classes."""
    __slots__ = ()

    @singledispatchmethod
    def map_concept(self, c: Union[owlready2.ClassConstruct, owlready2.ThingClass]) -> OWLClassExpression:
        """Map concept classes."""
        raise NotImplementedError(c)

    @singledispatchmethod
    def _from_owlready2_property(self, c: Union[owlready2.PropertyClass, owlready2.Inverse]) -> OWLPropertyExpression:
        raise NotImplementedError(c)

    @_from_owlready2_property.register
    def _(self, p: owlready2.ObjectPropertyClass) -> OWLObjectProperty:
        return OWLObjectProperty(IRI.create(p.iri))

    @_from_owlready2_property.register
    def _(self, p: owlready2.DataPropertyClass) -> OWLDataProperty:
        return OWLDataProperty(IRI.create(p.iri))

    @_from_owlready2_property.register
    def _(self, i: owlready2.Inverse) -> OWLObjectInverseOf:
        return OWLObjectInverseOf(self._from_owlready2_property(i.property))

    @map_concept.register
    def _(self, c: owlready2.ThingClass) -> OWLClass:
        return OWLClass(IRI.create(c.iri))

    @map_concept.register
    def _(self, c: owlready2.Not) -> OWLObjectComplementOf:
        return OWLObjectComplementOf(self.map_concept(c.Class))

    @map_concept.register
    def _(self, c: owlready2.And) -> OWLObjectIntersectionOf:
        return OWLObjectIntersectionOf(map(self.map_concept, c.Classes))

    @map_concept.register
    def _(self, c: owlready2.Or) -> OWLObjectUnionOf:
        return OWLObjectUnionOf(map(self.map_concept, c.Classes))

    @map_concept.register
    def _(self, c: owlready2.OneOf) -> OWLObjectOneOf:
        return OWLObjectOneOf([OWLNamedIndividual(IRI.create(ind.iri)) for ind in c.instances])

    @map_concept.register
    def _(self, c: owlready2.Restriction) -> OWLRestriction:
        if isinstance(c.property, owlready2.ObjectPropertyClass):
            return self._to_object_property(c)
        elif isinstance(c.property, owlready2.DataPropertyClass):
            return self._to_data_property(c)
        else:
            raise NotImplementedError(c)

    def _to_object_property(self, c: owlready2.Restriction) -> OWLObjectRestriction:
        p = self._from_owlready2_property(c.property)
        assert isinstance(p, OWLObjectPropertyExpression)

        if c.type == owlready2.VALUE:
            ind = OWLNamedIndividual(IRI.create(c.value.iri))
            return OWLObjectHasValue(p, ind)
        else:
            f = self.map_concept(c.value)
            if c.type == owlready2.SOME:
                return OWLObjectSomeValuesFrom(p, f)
            elif c.type == owlready2.ONLY:
                return OWLObjectAllValuesFrom(p, f)
            elif c.type == owlready2.EXACTLY:
                return OWLObjectExactCardinality(c.cardinality, p, f)
            elif c.type == owlready2.MIN:
                return OWLObjectMinCardinality(c.cardinality, p, f)
            elif c.type == owlready2.MAX:
                return OWLObjectMaxCardinality(c.cardinality, p, f)
            else:
                raise NotImplementedError(c)

    def _to_data_property(self, c: owlready2.Restriction) -> OWLDataRestriction:
        p = self._from_owlready2_property(c.property)
        assert isinstance(p, OWLDataPropertyExpression)

        if c.type == owlready2.VALUE:
            return OWLDataHasValue(p, OWLLiteral(c.value))
        else:
            f = self.map_datarange(c.value)
            if c.type == owlready2.SOME:
                return OWLDataSomeValuesFrom(p, f)
            elif c.type == owlready2.ONLY:
                return OWLDataAllValuesFrom(p, f)
            elif c.type == owlready2.EXACTLY:
                return OWLDataExactCardinality(c.cardinality, p, f)
            elif c.type == owlready2.MIN:
                return OWLDataMinCardinality(c.cardinality, p, f)
            elif c.type == owlready2.MAX:
                return OWLDataMaxCardinality(c.cardinality, p, f)
            else:
                raise NotImplementedError(c)

    @singledispatchmethod
    def map_datarange(self, p: owlready2.ClassConstruct) -> OWLDataRange:
        """Map data range classes."""
        raise NotImplementedError(p)

    @map_datarange.register
    def _(self, p: owlready2.Not) -> OWLDataComplementOf:
        return OWLDataComplementOf(self.map_datarange(p.Class))

    @map_datarange.register
    def _(self, p: owlready2.Or) -> OWLDataUnionOf:
        return OWLDataUnionOf(map(self.map_datarange, p.Classes))

    @map_datarange.register
    def _(self, p: owlready2.And) -> OWLDataIntersectionOf:
        return OWLDataIntersectionOf(map(self.map_datarange, p.Classes))

    @map_datarange.register
    def _(self, p: owlready2.OneOf) -> OWLDataOneOf:
        return OWLDataOneOf([OWLLiteral(i) for i in p.instances])

    @map_datarange.register
    def _(self, p: owlready2.ConstrainedDatatype) -> OWLDatatypeRestriction:
        restrictions = []
        for facet in OWLFacet:
            value = getattr(p, OWLREADY2_FACET_KEYS[facet], None)
            if value is not None:
                restrictions.append(OWLFacetRestriction(facet, OWLLiteral(value)))
        return OWLDatatypeRestriction(self.map_datarange(p.base_datatype), restrictions)

    @map_datarange.register
    def _(self, type_: type) -> OWLDatatype:
        if type_ == bool:
            return BooleanOWLDatatype
        elif type_ == float:
            return DoubleOWLDatatype
        elif type_ == int:
            return IntegerOWLDatatype
        elif type_ == str:
            return StringOWLDatatype
        elif type_ == date:
            return DateOWLDatatype
        elif type_ == datetime:
            return DateTimeOWLDatatype
        elif type_ == Timedelta:
            return DurationOWLDatatype
        else:
            raise ValueError(type_)
