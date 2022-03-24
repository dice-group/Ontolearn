from functools import singledispatch
import types
from typing import cast

import owlready2

from owlapy.model import OWLObjectPropertyAssertionAxiom, OWLOntology, OWLSubClassOfAxiom, OWLThing, OWLAxiom, \
    OWLAnnotationAssertionAxiom, OWLClass, OWLClassAssertionAxiom, OWLEquivalentClassesAxiom
from owlapy.owlready2.utils import ToOwlready2


@singledispatch
def _add_axiom(axiom: OWLAxiom):
    raise NotImplementedError(f'Axiom type {axiom} is not implemented yet.')


@_add_axiom.register
def _(axiom: OWLClassAssertionAxiom, ontology: OWLOntology, world: owlready2.namespace.World):
    conv = ToOwlready2(world)
    ontology_x = conv.map_object(ontology)

    individual = axiom.get_individual()
    cls_ = axiom.get_class_expression()
    with ontology_x:
        assert isinstance(cls_, OWLClass), f'Owlready2 only supports named classes ({cls_})'
        cls_x: owlready2.entity.ThingClass = world[cls_.to_string_id()]
        if cls_x is None:
            raise ValueError(f"Class {cls_} does not exist in ontology {ontology}.")
        ind_x: owlready2.individual.Thing = world[individual.to_string_id()]
        if ind_x is None:
            _ = cls_x(individual.get_iri().get_remainder())
        else:
            ind_x.is_a.append(cls_x)


@_add_axiom.register
def _(axiom: OWLObjectPropertyAssertionAxiom, ontology: OWLOntology, world: owlready2.namespace.World):
    conv = ToOwlready2(world)
    ontology_x = conv.map_object(ontology)

    subject = axiom.get_subject()
    property_ = axiom.get_property()
    object_ = axiom.get_object()
    thing_x: owlready2.entity.ThingClass = conv.map_concept(OWLThing)
    with ontology_x:
        subject_x = world[subject.to_string_id()] or thing_x(subject.get_iri().get_remainder())
        object_x = world[object_.to_string_id()] or thing_x(object_.get_iri().get_remainder())
        property_x = world[property_.to_string_id()]
        if property_x is None:
            raise ValueError(f"Property {property_} does not exist in ontology {ontology}.")
        property_x[subject_x].append(object_x)


@_add_axiom.register
def _(axiom: OWLSubClassOfAxiom, ontology: OWLOntology, world: owlready2.namespace.World):
    conv = ToOwlready2(world)
    ontology_x = conv.map_object(ontology)

    sub_class = axiom.get_sub_class()
    super_class = axiom.get_super_class()
    with ontology_x:
        assert isinstance(sub_class, OWLClass), f'Owlready2 only supports named classes ({sub_class})'
        sub_class_x: owlready2.entity.ThingClass = world[sub_class.to_string_id()]
        if sub_class_x is None:
            thing_x: owlready2.entity.ThingClass = conv.map_concept(OWLThing)
            sub_class_x: owlready2.entity.ThingClass = cast(thing_x,
                                                            types.new_class(
                                                                name=sub_class.get_iri().get_remainder(),
                                                                bases=(thing_x,)))
            sub_class_x.namespace = ontology_x.get_namespace(sub_class.get_iri().get_namespace())
            sub_class_x.is_a.remove(thing_x)
        sub_class_x.is_a.append(conv.map_concept(super_class))


@_add_axiom.register
def _(axiom: OWLEquivalentClassesAxiom, ontology: OWLOntology, world: owlready2.namespace.World):
    conv = ToOwlready2(world)
    ontology_x = conv.map_object(ontology)

    cls_a, cls_b = axiom.class_expressions()
    thing_x: owlready2.entity.ThingClass = conv.map_concept(OWLThing)
    with ontology_x:
        assert isinstance(cls_a, OWLClass), f'{cls_a} is no named class'
        w_x: owlready2.entity.ThingClass = cast(thing_x,
                                                types.new_class(name=cls_a.get_iri().get_remainder(),
                                                                bases=(thing_x,)))
        w_x.namespace = ontology_x.get_namespace(cls_a.get_iri().get_namespace())
        w_x.is_a.remove(thing_x)
        w_x.equivalent_to.append(conv.map_concept(cls_b))


@_add_axiom.register
def _(axiom: OWLAnnotationAssertionAxiom, ontology: OWLOntology, world: owlready2.namespace.World):
    conv = ToOwlready2(world)
    ontology_x: owlready2.Ontology = conv.map_object(ontology)

    prop_x = conv.map_object(axiom.get_property())
    if prop_x is None:
        with ontology_x:
            prop_x: owlready2.annotation.AnnotationPropertyClass = cast(
                    owlready2.AnnotationProperty,
                    types.new_class(
                        name=axiom.get_property().get_iri().get_remainder(),
                        bases=(owlready2.AnnotationProperty,)))
            prop_x.namespace = ontology_x.get_namespace(axiom.get_property().get_iri().get_namespace())
    sub_x = world[axiom.get_subject().as_iri().as_str()]
    assert sub_x is not None, f'{axiom.get_subject} not found in {ontology}'
    if axiom.get_value().is_literal():
        literal = axiom.get_value().as_literal()
        if literal.is_double():
            v = literal.parse_double()
        elif literal.is_integer():
            v = literal.parse_integer()
        elif literal.is_boolean():
            v = literal.parse_boolean()
        else:
            # TODO XXX
            raise NotImplementedError
        setattr(sub_x, prop_x.python_name, v)
    else:
        o_x = world[axiom.get_value().as_iri().as_str()]
        assert o_x is not None, f'{axiom.get_value()} not found in {ontology}'
        setattr(sub_x, prop_x.python_name, o_x)


@singledispatch
def _remove_axiom(axiom: OWLAxiom, ontology: OWLOntology, world: owlready2.namespace.World):
    raise NotImplementedError(f'Axiom type {axiom} is not implemented yet.')


@_remove_axiom.register
def _(axiom: OWLClassAssertionAxiom, ontology: OWLOntology, world: owlready2.namespace.World):
    conv = ToOwlready2(world)
    ontology_x = conv.map_object(ontology)

    ind = axiom.get_individual()
    cls_ = axiom.get_class_expression()
    with ontology_x:
        assert isinstance(cls_, OWLClass), f'Owlready2 only supports named classes ({cls_})'
        cls_x: owlready2.entity.ThingClass = world[cls_.to_string_id()]
        ind_x: owlready2.individual.Thing = world[ind.to_string_id()]
        if cls_x is None:
            raise ValueError(f"Class {cls_} does not exist in ontology {ontology}.")
        elif ind_x is None:
            raise ValueError(f"Individual {ind} does not exist in ontology {ontology}.")
        elif cls_x in ind_x.is_a:
            ind_x.is_a.remove(cls_x)


@_remove_axiom.register
def _(axiom: OWLObjectPropertyAssertionAxiom, ontology: OWLOntology, world: owlready2.namespace.World):
    conv = ToOwlready2(world)
    ontology_x = conv.map_object(ontology)

    subject = axiom.get_subject()
    property_ = axiom.get_property()
    object_ = axiom.get_object()
    with ontology_x:
        subject_x = world[subject.to_string_id()]
        property_x = world[property_.to_string_id()]
        object_x = world[object_.to_string_id()]
        if subject_x is None:
            raise ValueError(f"Individual {subject} does not exist in ontology {ontology}.")
        elif property_x is None:
            raise ValueError(f"Property {property_} does not exist in ontology {ontology}.")
        elif object_x is None:
            raise ValueError(f"Individual {object_} does not exist in ontology {ontology}.")
        elif object_x in property_x[subject_x]:
            property_x[subject_x].remove(object_x)


@_remove_axiom.register
def _(axiom: OWLSubClassOfAxiom, ontology: OWLOntology, world: owlready2.namespace.World):
    conv = ToOwlready2(world)
    ontology_x = conv.map_object(ontology)

    sub_class = axiom.get_sub_class()
    super_class = axiom.get_super_class()
    with ontology_x:
        assert isinstance(sub_class, OWLClass), f'Owlready2 only supports named classes ({sub_class})'
        assert isinstance(super_class, OWLClass), f'Owlready2 only supports named classes ({super_class})'
        sub_class_x: owlready2.entity.ThingClass = world[sub_class.to_string_id()]
        super_class_x: owlready2.entity.ThingClass = world[super_class.to_string_id()]
        if sub_class_x is None:
            raise ValueError(f"Class {sub_class} does not exist in ontology {ontology}")
        elif super_class_x is None:
            raise ValueError(f"Class {super_class} does not exist in ontology {ontology}")
        elif super_class_x in sub_class_x.is_a:
            sub_class_x.is_a.remove(super_class_x)
