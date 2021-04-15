from functools import singledispatchmethod
from typing import Union

import owlready2

from owlapy import IRI
from owlapy.model import OWLClassExpression, OWLPropertyExpression, OWLObjectProperty, OWLClass, \
    OWLObjectComplementOf, OWLObjectUnionOf, OWLObjectIntersectionOf, OWLObjectSomeValuesFrom, OWLObjectAllValuesFrom, \
    OWLObjectPropertyExpression, OWLObject, OWLOntology, OWLAnnotationProperty


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
