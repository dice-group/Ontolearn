"""OWL Reasoner Plus."""
from typing import Iterable, Set

import owlready2

from owlapy import namespaces
from owlapy.class_expression import OWLClassExpression, OWLClass, OWLThing
from owlapy.iri import IRI
from owlapy.owl_property import OWLObjectPropertyExpression, OWLObjectProperty
from ontolearn.base import OWLReasoner_Owlready2, OWLOntology_Owlready2


class OWLReasoner_Owlready2_Plus(OWLReasoner_Owlready2):
    """OWL Reasoner based on owlready2.

    Contains some behavioural fixes."""

    def __init__(self, ontology: OWLOntology_Owlready2, isolate: bool = False):
        super().__init__(ontology, isolate)

    def sub_classes(self, ce: OWLClassExpression, direct: bool = False, only_named: bool = True) -> Iterable[OWLClass]:
        if isinstance(ce, OWLClass):
            if direct:
                if ce.is_owl_thing():
                    thing_x = self._world[OWLThing.str]
                    for c in self._ontology.classes_in_signature():
                        c_x: owlready2.ThingClass = self._world[c.str]
                        super_classes_x = []
                        for super_class_x in c_x.is_a:
                            if isinstance(super_class_x, owlready2.ThingClass):
                                super_classes_x.append(super_class_x)
                            # elif isinstance(super_class_x, )
                        if super_classes_x == [thing_x]:
                            yield c
                else:
                    c_x: owlready2.ThingClass = self._world[ce.str]
                    sub_classes_x = set()
                    for sc_x in c_x.subclasses(world=self._world):
                        if isinstance(sc_x, owlready2.ThingClass):
                            if sc_x != c_x:
                                sub_classes_x.add(sc_x)
                    # filter out indirect sub-classes
                    for sc_x in sub_classes_x.copy():
                        for ssc_x in sc_x.subclasses(world=self._world):
                            if sc_x != ssc_x:
                                sub_classes_x.discard(ssc_x)
                    for sc_x in sub_classes_x:
                        yield OWLClass(IRI.create(sc_x.iri))
                    # Anonymous classes are ignored
            else:
                # indirect
                seen_set = set()
                yield from self._sub_classes_recursive(ce, seen_set, only_named=only_named)
        else:
            raise NotImplementedError("sub classes for complex class expressions not implemented", ce)

    def _sub_object_properties_recursive(self, op: OWLObjectProperty, seen_set: Set) -> Iterable[OWLObjectProperty]:
        if op.is_owl_top_object_property():
            yield from self._ontology.object_properties_in_signature()
        else:
            yield from super()._sup_or_sub_object_properties_recursive(op, seen_set, "sub")

    def sub_object_properties(self, op: OWLObjectPropertyExpression, direct: bool = False) -> Iterable[
            OWLObjectPropertyExpression]:
        assert isinstance(op, OWLObjectPropertyExpression)
        if isinstance(op, OWLObjectProperty):
            if direct:
                if op.is_owl_top_object_property():
                    owl_objectproperty_x: owlready2.ObjectPropertyClass = self._world[
                        IRI.create(namespaces.OWL, "ObjectProperty").as_str()]
                    for oop in self._ontology.object_properties_in_signature():
                        p_x: owlready2.ObjectPropertyClass = self._world[oop.str]
                        if p_x.is_a == [owl_objectproperty_x]:
                            yield oop
                else:
                    p_x: owlready2.ObjectPropertyClass = self._world[op.str]
                    for sp in p_x.subclasses(world=self._world):
                        if isinstance(sp, owlready2.ObjectPropertyClass):
                            yield OWLObjectProperty(IRI.create(sp.iri))
            else:
                seen_set = set()
                yield from self._sub_object_properties_recursive(op, seen_set)
        else:
            raise NotImplementedError("sub properties of inverse properties not yet implemented", op)
