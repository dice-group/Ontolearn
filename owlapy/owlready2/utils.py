from functools import singledispatchmethod
from typing import Type, Union

import owlready2

from owlapy.model import OWLDataOneOf, OWLObjectMinCardinality, OWLPropertyExpression, OWLObjectProperty, OWLClass, \
    OWLObjectComplementOf, OWLObjectUnionOf, OWLObjectIntersectionOf, OWLObjectSomeValuesFrom, OWLObjectAllValuesFrom, \
    OWLObjectPropertyExpression, OWLObject, OWLOntology, OWLAnnotationProperty, IRI, OWLObjectInverseOf, \
    DoubleOWLDatatype, IntegerOWLDatatype, OWLClassExpression, OWLDataAllValuesFrom, OWLDataComplementOf, \
    OWLDataIntersectionOf, OWLDataProperty, OWLDataRange, OWLDataSomeValuesFrom, OWLDataUnionOf, OWLDatatype, \
    BooleanOWLDatatype, OWLDataHasValue, OWLDataExactCardinality, OWLDataMaxCardinality, OWLDataMinCardinality, \
    OWLDataPropertyExpression, OWLDatatypeRestriction, OWLFacetRestriction, OWLLiteral, OWLObjectHasValue, \
    OWLNamedIndividual, OWLObjectExactCardinality, OWLObjectMaxCardinality \

from owlapy.vocab import OWLFacet


class ToOwlready2:
    __slots__ = '_world'

    _world: owlready2.World

    def __init__(self, world: owlready2.World):
        self._world = world

    @singledispatchmethod
    def map_object(self, o: OWLObject):
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
        return self._world[ap.get_iri().as_str()]

    # single dispatch is still not implemented in mypy, see https://github.com/python/mypy/issues/2904
    @singledispatchmethod
    def map_concept(self, o: OWLClassExpression) \
            -> Union[owlready2.ClassConstruct, owlready2.ThingClass]:
        raise NotImplementedError

    @singledispatchmethod
    def _to_owlready2_property(self, p: OWLPropertyExpression) -> owlready2.Property:
        raise NotImplementedError

    @_to_owlready2_property.register
    def _(self, p: OWLObjectInverseOf):
        p_x = self._to_owlready2_property(p.get_named_property())
        return owlready2.Inverse(p_x)

    @_to_owlready2_property.register
    def _(self, p: OWLObjectProperty) -> owlready2.prop.ObjectPropertyClass:
        return self._world[p.get_iri().as_str()]

    @_to_owlready2_property.register
    def _(self, p: OWLDataProperty) -> owlready2.prop.DataPropertyClass:
        return self._world[p.get_iri().as_str()]

    @map_concept.register
    def _(self, c: OWLClass) -> owlready2.ThingClass:
        return self._world[c.get_iri().as_str()]

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
        return prop.some(self.map_concept(ce.get_filler()))

    @map_concept.register
    def _(self, ce: OWLObjectAllValuesFrom) -> owlready2.class_construct.Restriction:
        prop = self._to_owlready2_property(ce.get_property())
        return prop.only(self.map_concept(ce.get_filler()))

    @map_concept.register
    def _(self, ce: OWLDataSomeValuesFrom) -> owlready2.class_construct.Restriction:
        prop = self._to_owlready2_property(ce.get_property())
        new_filler = self._map_datarange(ce.get_filler())
        return prop.some(new_filler)

    @map_concept.register
    def _(self, ce: OWLDataAllValuesFrom) -> owlready2.class_construct.Restriction:
        prop = self._to_owlready2_property(ce.get_property())
        new_filler = self._map_datarange(ce.get_filler())
        return prop.only(new_filler)

    @map_concept.register
    def _(self, ce: OWLDataHasValue) -> owlready2.class_construct.Restriction:
        prop = self._to_owlready2_property(ce.get_property())
        value = ce.get_filler().to_python()
        return prop.value(value)

    @singledispatchmethod
    def _map_datarange(self, p: OWLDataRange) -> Union[owlready2.ClassConstruct, Type]:
        raise NotImplementedError

    @_map_datarange.register
    def _(self, p: OWLDataComplementOf) -> owlready2.class_construct.Not:
        return owlready2.Not(self._map_datarange(p.get_data_range()))

    @_map_datarange.register
    def _(self, p: OWLDataUnionOf) -> owlready2.class_construct.Or:
        return owlready2.Or(map(self._map_datarange, p.operands()))

    @_map_datarange.register
    def _(self, p: OWLDataIntersectionOf) -> owlready2.class_construct.And:
        return owlready2.And(map(self._map_datarange, p.operands()))

    @_map_datarange.register
    def _(self, p: OWLDataOneOf) -> owlready2.class_construct.OneOf:
        return owlready2.OneOf([lit.to_python() for lit in p.operands()])

    @_map_datarange.register
    def _(self, p: OWLDatatypeRestriction) -> owlready2.class_construct.ConstrainedDatatype:
        args = dict()
        for facet_res in p.get_facet_restrictions():
            facet = facet_res.get_facet()
            op = facet.operator
            value = facet_res.get_facet_value().to_python()
            if facet.owlready2_key not in args or op(value, args[facet.owlready2_key]):
                args[facet.owlready2_key] = value

        return owlready2.ConstrainedDatatype(self._map_datarange(p.get_datatype()), **args)

    @_map_datarange.register
    def _(self, type_: OWLDatatype) -> type:
        if type_ == BooleanOWLDatatype:
            return bool
        elif type_ == DoubleOWLDatatype:
            return float
        elif type_ == IntegerOWLDatatype:
            return int
        else:
            raise ValueError


class FromOwlready2:
    __slots__ = ()

    @singledispatchmethod
    def map_concept(self, c: Union[owlready2.ClassConstruct, owlready2.ThingClass]) -> OWLClassExpression:
        raise NotImplementedError

    @singledispatchmethod
    def _from_owlready2_property(self, c: Union[owlready2.PropertyClass, owlready2.Inverse]) -> OWLPropertyExpression:
        raise NotImplementedError

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
    def _(self, c: owlready2.Restriction) -> OWLPropertyExpression:
        if isinstance(c.property, owlready2.ObjectPropertyClass):
            return self._to_object_property(c)
        elif isinstance(c.property, owlready2.DataPropertyClass):
            return self._to_data_property(c)
        else:
            raise NotImplementedError

    def _to_object_property(self, c: owlready2.Restriction) -> OWLObjectPropertyExpression:
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
                raise NotImplementedError

    def _to_data_property(self, c: owlready2.Restriction) -> OWLDataPropertyExpression:
        p = self._from_owlready2_property(c.property)
        assert isinstance(p, OWLDataPropertyExpression)

        if c.type == owlready2.VALUE:
            return OWLDataHasValue(p, OWLLiteral(c.value))
        else:
            f = self._map_datarange(c.value)
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
                raise NotImplementedError

    @singledispatchmethod
    def _map_datarange(self, p: owlready2.ClassConstruct) -> OWLDataRange:
        raise NotImplementedError

    @_map_datarange.register
    def _(self, p: owlready2.Not) -> OWLDataComplementOf:
        return OWLDataComplementOf(self._map_datarange(p.Class))

    @_map_datarange.register
    def _(self, p: owlready2.Or) -> OWLDataUnionOf:
        return OWLDataUnionOf(map(self._map_datarange, p.Classes))

    @_map_datarange.register
    def _(self, p: owlready2.And) -> OWLDataIntersectionOf:
        return OWLDataIntersectionOf(map(self._map_datarange, p.Classes))

    @_map_datarange.register
    def _(self, p: owlready2.OneOf) -> OWLDataOneOf:
        return OWLDataOneOf([OWLLiteral(i) for i in p.instances])

    @_map_datarange.register
    def _(self, p: owlready2.ConstrainedDatatype) -> OWLDatatypeRestriction:
        restrictions = []
        for f in [OWLFacet.MIN_EXCLUSIVE, OWLFacet.MIN_INCLUSIVE, OWLFacet.MAX_EXCLUSIVE, OWLFacet.MAX_INCLUSIVE]:
            value = getattr(p, f.owlready2_key, None)
            if value is not None:
                restrictions.append(OWLFacetRestriction(f, OWLLiteral(value)))

        return OWLDatatypeRestriction(self._map_datarange(p.base_datatype), restrictions)

    @_map_datarange.register
    def _(self, type_: type) -> OWLDatatype:
        if type_ == bool:
            return BooleanOWLDatatype
        elif type_ == float:
            return DoubleOWLDatatype
        elif type_ == int:
            return IntegerOWLDatatype
        else:
            raise ValueError
