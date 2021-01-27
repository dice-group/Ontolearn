from functools import singledispatchmethod
from typing import Union

import owlready2

from ontolearn.owlapy import IRI
from ontolearn.owlapy.model import OWLClassExpression, OWLPropertyExpression, OWLObjectProperty, OWLClass, \
    OWLObjectComplementOf, OWLObjectUnionOf, OWLObjectIntersectionOf, OWLObjectSomeValuesFrom, OWLObjectAllValuesFrom, \
    OWLObjectPropertyExpression


class ToOwlready2:
    __slots__ = '_world'

    _world: owlready2.World

    def __init__(self, world: owlready2.World):
        self._world = world

    @singledispatchmethod
    def map_concept(self, o: OWLClassExpression) \
            -> Union[owlready2.ClassConstruct, owlready2.ThingClass]:
        raise NotImplementedError

    @singledispatchmethod
    def _to_owlready2_property(self, p: OWLPropertyExpression) -> owlready2.Property:
        raise NotImplementedError

    @_to_owlready2_property.register
    def _(self, p: OWLObjectProperty) -> owlready2.prop.ObjectPropertyClass:
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


class FromOwlready2:
    __slots__ = ()

    @singledispatchmethod
    def map_concept(self, c: Union[owlready2.ClassConstruct, owlready2.ThingClass]) -> OWLClassExpression:
        raise NotImplementedError

    @singledispatchmethod
    def _from_owlready2_property(self, c: owlready2.PropertyClass) -> OWLPropertyExpression:
        raise NotImplementedError

    @_from_owlready2_property.register
    def _(self, p: owlready2.ObjectPropertyClass) -> OWLObjectProperty:
        return OWLObjectProperty(IRI.create(p.iri))

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
    def _(self, c: owlready2.Restriction) -> Union[OWLObjectSomeValuesFrom, OWLObjectAllValuesFrom]:
        if c.type == owlready2.SOME and c.cardinality is None:
            p = self._from_owlready2_property(c.property)
            assert isinstance(p, OWLObjectPropertyExpression)
            f = self.map_concept(c.value)
            return OWLObjectSomeValuesFrom(p, f)
        elif c.type == owlready2.ONLY and c.cardinality is None:
            p = self._from_owlready2_property(c.property)
            assert isinstance(p, OWLObjectPropertyExpression)
            f = self.map_concept(c.value)
            return OWLObjectAllValuesFrom(p, f)
        else:
            raise NotImplementedError
