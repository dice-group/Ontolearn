import logging
import types
from datetime import date, datetime
from enum import Enum, auto
from itertools import chain
from types import MappingProxyType
from typing import Iterable, Set, Final, cast

import owlready2
from owlready2 import declare_datatype
from pandas import Timedelta

from owlapy import namespaces
from owlapy.ext import OWLReasonerEx
from owlapy.model import OWLObjectPropertyRangeAxiom, OWLOntologyManager, OWLDataProperty, OWLObjectProperty, \
    OWLNamedIndividual, OWLClassExpression, OWLObjectPropertyExpression, OWLOntologyID, OWLAxiom, OWLOntology, \
    OWLOntologyChange, AddImport, OWLEquivalentClassesAxiom, OWLThing, OWLAnnotationAssertionAxiom, DoubleOWLDatatype, \
    OWLObjectInverseOf, BooleanOWLDatatype, IntegerOWLDatatype, DateOWLDatatype, DateTimeOWLDatatype, OWLClass, \
    DurationOWLDatatype, StringOWLDatatype, IRI, OWLDataPropertyRangeAxiom, OWLDataPropertyDomainAxiom, OWLLiteral, \
    OWLObjectPropertyDomainAxiom, OWLSubClassOfAxiom, OWLClassAssertionAxiom, OWLObjectPropertyAssertionAxiom
from owlapy.owlready2.utils import FromOwlready2, ToOwlready2

logger = logging.getLogger(__name__)

_Datatype_map: Final = MappingProxyType({
    int: IntegerOWLDatatype,
    float: DoubleOWLDatatype,
    bool: BooleanOWLDatatype,
    str: StringOWLDatatype,
    date: DateOWLDatatype,
    datetime: DateTimeOWLDatatype,
    Timedelta: DurationOWLDatatype,
})

_parse_concept_to_owlapy = FromOwlready2().map_concept
_parse_datarange_to_owlapy = FromOwlready2().map_datarange

_VERSION_IRI: Final = IRI.create(namespaces.OWL, "versionIRI")


def _parse_duration_datatype(literal: str):
    return Timedelta(literal)


def _unparse_duration_datatype(literal: Timedelta):
    return literal.isoformat()


declare_datatype(Timedelta, IRI.create(namespaces.XSD, "duration").as_str(),
                 _parse_duration_datatype, _unparse_duration_datatype)


class BaseReasoner_Owlready2(Enum):
    PELLET = auto()
    HERMIT = auto()


class OWLOntologyManager_Owlready2(OWLOntologyManager):
    __slots__ = '_world'

    _world: owlready2.namespace.World

    def __init__(self, world_store=None):
        if world_store is None:
            self._world = owlready2.World()
        else:
            self._world = owlready2.World(filename=world_store)

    def create_ontology(self, iri: IRI) -> 'OWLOntology_Owlready2':
        return OWLOntology_Owlready2(self, iri, load=False)

    def load_ontology(self, iri: IRI) -> 'OWLOntology_Owlready2':
        return OWLOntology_Owlready2(self, iri, load=True)

    def apply_change(self, change: OWLOntologyChange):
        if isinstance(change, AddImport):
            ont_x: owlready2.namespace.Ontology = self._world.get_ontology(
                change.get_ontology().get_ontology_id().get_ontology_iri().as_str())
            ont_x.imported_ontologies.append(
                self._world.get_ontology(change.get_import_declaration().get_iri().as_str()))
        else:
            # TODO XXX
            raise NotImplementedError

    # TODO: Refactor this method, use dispatch? Also lots of duplicated code.
    # TODO: Compare behavior to owlapi and adjust if necessary.
    def add_axiom(self, ontology: OWLOntology, axiom: OWLAxiom):
        conv = ToOwlready2(self._world)
        ont_x: owlready2.namespace.Ontology = conv.map_object(ontology)

        if isinstance(axiom, OWLClassAssertionAxiom):
            individual = axiom.get_individual()
            cls_ = axiom.get_class_expression()
            with ont_x:
                assert isinstance(cls_, OWLClass), f'Owlready2 only supports named classes ({cls_})'
                cls_x: owlready2.entity.ThingClass = self._world[cls_.to_string_id()]
                if cls_x is None:
                    raise ValueError(f"Class {cls_} does not exist in ontology {ontology.get_ontology_id()}.")
                ind_x: owlready2.individual.Thing = self._world[individual.to_string_id()]
                if ind_x is None:
                    _ = cls_x(individual.get_iri().get_remainder())
                else:
                    ind_x.is_a.append(cls_x)
        elif isinstance(axiom, OWLObjectPropertyAssertionAxiom):
            subject = axiom.get_subject()
            property_ = axiom.get_property()
            object_ = axiom.get_object()
            thing_x: owlready2.entity.ThingClass = conv.map_concept(OWLThing)
            with ont_x:
                subject_x = self._world[subject.to_string_id()] or thing_x(subject.get_iri().get_remainder())
                object_x = self._world[object_.to_string_id()] or thing_x(object_.get_iri().get_remainder())
                property_x = self._world[property_.to_string_id()]
                if property_x is None:
                    raise ValueError(f"Property {property_} does not exist in ontology {ontology.get_ontology_id()}.")
                property_x[subject_x].append(object_x)
        elif isinstance(axiom, OWLEquivalentClassesAxiom):
            cls_a, cls_b = axiom.class_expressions()
            thing_x: owlready2.entity.ThingClass = conv.map_concept(OWLThing)
            with ont_x:
                assert isinstance(cls_a, OWLClass), f'{cls_a} is no named class'
                w_x: owlready2.entity.ThingClass = cast(thing_x,
                                                        types.new_class(name=cls_a.get_iri().get_remainder(),
                                                                        bases=(thing_x,)))
                w_x.namespace = ont_x.get_namespace(cls_a.get_iri().get_namespace())
                w_x.is_a.remove(thing_x)
                w_x.equivalent_to.append(conv.map_concept(cls_b))
        elif isinstance(axiom, OWLSubClassOfAxiom):
            sub_class = axiom.get_sub_class()
            super_class = axiom.get_super_class()
            with ont_x:
                assert isinstance(sub_class, OWLClass), f'Owlready2 only supports named classes ({sub_class})'
                sub_class_x: owlready2.entity.ThingClass = self._world[sub_class.to_string_id()]
                if sub_class_x is None:
                    thing_x: owlready2.entity.ThingClass = conv.map_concept(OWLThing)
                    sub_class_x: owlready2.entity.ThingClass = cast(thing_x,
                                                                    types.new_class(
                                                                        name=sub_class.get_iri().get_remainder(),
                                                                        bases=(thing_x,)))
                    sub_class_x.namespace = ont_x.get_namespace(sub_class.get_iri().get_namespace())
                    sub_class_x.is_a.remove(thing_x)
                sub_class_x.is_a.append(conv.map_concept(super_class))
        elif isinstance(axiom, OWLAnnotationAssertionAxiom):
            prop_x = conv.map_object(axiom.get_property())
            if prop_x is None:
                with ont_x:
                    prop_x: owlready2.annotation.AnnotationPropertyClass = cast(
                        owlready2.AnnotationProperty,
                        types.new_class(
                            name=axiom.get_property().get_iri().get_remainder(),
                            bases=(owlready2.AnnotationProperty,)))
                    prop_x.namespace = ont_x.get_namespace(axiom.get_property().get_iri().get_namespace())
            sub_x = self._world[axiom.get_subject().as_iri().as_str()]
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
                o_x = self._world[axiom.get_value().as_iri().as_str()]
                assert o_x is not None, f'{axiom.get_value()} not found in {ontology}'
                setattr(sub_x, prop_x.python_name, o_x)
        else:
            # TODO XXX
            raise NotImplementedError(axiom)

    # TODO: Refactor this method, use dispatch? Also lots of duplicated code.
    # TODO: Compare behavior to owlapi and adjust if necessary.
    def remove_axiom(self, ontology: OWLOntology, axiom: OWLAxiom):
        conv = ToOwlready2(self._world)
        ont_x: owlready2.namespace.Ontology = conv.map_object(ontology)

        if isinstance(axiom, OWLClassAssertionAxiom):
            ind = axiom.get_individual()
            cls_ = axiom.get_class_expression()
            with ont_x:
                assert isinstance(cls_, OWLClass), f'Owlready2 only supports named classes ({cls_})'
                cls_x: owlready2.entity.ThingClass = self._world[cls_.to_string_id()]
                ind_x: owlready2.individual.Thing = self._world[ind.to_string_id()]
                if cls_x is None:
                    raise ValueError(f"Class {cls_} does not exist in ontology {ontology.get_ontology_id()}.")
                elif ind_x is None:
                    raise ValueError(f"Individual {ind} does not exist in ontology {ontology.get_ontology_id()}.")
                elif cls_x in ind_x.is_a:
                    ind_x.is_a.remove(cls_x)
        elif isinstance(axiom, OWLObjectPropertyAssertionAxiom):
            subject = axiom.get_subject()
            property_ = axiom.get_property()
            object_ = axiom.get_object()
            with ont_x:
                subject_x = self._world[subject.to_string_id()]
                property_x = self._world[property_.to_string_id()]
                object_x = self._world[object_.to_string_id()]
                if subject_x is None:
                    raise ValueError(f"Individual {subject} does not exist in ontology {ontology.get_ontology_id()}.")
                elif property_x is None:
                    raise ValueError(f"Property {property_} does not exist in ontology {ontology.get_ontology_id()}.")
                elif object_x is None:
                    raise ValueError(f"Individual {object_} does not exist in ontology {ontology.get_ontology_id()}.")
                elif object_x in property_x[subject_x]:
                    property_x[subject_x].remove(object_x)
        elif isinstance(axiom, OWLSubClassOfAxiom):
            sub_class = axiom.get_sub_class()
            super_class = axiom.get_super_class()
            with ont_x:
                assert isinstance(sub_class, OWLClass), f'Owlready2 only supports named classes ({sub_class})'
                assert isinstance(super_class, OWLClass), f'Owlready2 only supports named classes ({super_class})'
                sub_class_x: owlready2.entity.ThingClass = self._world[sub_class.to_string_id()]
                super_class_x: owlready2.entity.ThingClass = self._world[super_class.to_string_id()]
                if sub_class_x is None:
                    raise ValueError(f"Class {sub_class} does not exist in ontology {ontology.get_ontology_id()}")
                elif super_class_x is None:
                    raise ValueError(f"Class {super_class} does not exist in ontology {ontology.get_ontology_id()}")
                elif super_class_x in sub_class_x.is_a:
                    sub_class_x.is_a.remove(super_class_x)
        else:
            # TODO XXX
            raise NotImplementedError(axiom)

    def save_ontology(self, ontology: OWLOntology, document_iri: IRI):
        ont_x: owlready2.namespace.Ontology = self._world.get_ontology(
            ontology.get_ontology_id().get_ontology_iri().as_str()
        )
        if document_iri.get_namespace().startswith('file:/'):
            filename = document_iri.as_str()[len('file:/'):]
            ont_x.save(file=filename)
        else:
            # TODO XXX
            raise NotImplementedError

    def save_world(self):
        self._world.save()


class OWLOntology_Owlready2(OWLOntology):
    __slots__ = '_manager', '_world', '_onto'

    _manager: OWLOntologyManager_Owlready2
    _onto: owlready2.Ontology
    _world: owlready2.World

    def __init__(self, manager: OWLOntologyManager_Owlready2, ontology_iri: IRI, load: bool):
        self._manager = manager
        self._world = manager._world
        onto = self._world.get_ontology(ontology_iri.as_str())
        if load:
            onto = onto.load()
        self._onto = onto

    def classes_in_signature(self) -> Iterable[OWLClass]:
        for c in self._onto.classes():
            yield OWLClass(IRI.create(c.iri))

    def data_properties_in_signature(self) -> Iterable[OWLDataProperty]:
        for dp in self._onto.data_properties():
            yield OWLDataProperty(IRI.create(dp.iri))

    def object_properties_in_signature(self) -> Iterable[OWLObjectProperty]:
        for op in self._onto.object_properties():
            yield OWLObjectProperty(IRI.create(op.iri))

    def individuals_in_signature(self) -> Iterable[OWLNamedIndividual]:
        for i in self._onto.individuals():
            yield OWLNamedIndividual(IRI.create(i.iri))

    def get_owl_ontology_manager(self) -> OWLOntologyManager_Owlready2:
        return self._manager

    def get_ontology_id(self) -> OWLOntologyID:
        onto_iri = self._world._unabbreviate(self._onto.storid)
        look_version = self._world._get_obj_triple_sp_o(
            self._onto.storid,
            self._world._abbreviate(_VERSION_IRI.as_str()))
        if look_version is not None:
            version_iri = self._world._unabbreviate(look_version)
        else:
            version_iri = None

        return OWLOntologyID(IRI.create(onto_iri) if onto_iri is not None else None,
                             IRI.create(version_iri) if version_iri is not None else None)

    def data_property_domain_axioms(self, pe: OWLDataProperty) -> Iterable[OWLDataPropertyDomainAxiom]:
        p_x: owlready2.DataPropertyClass = self._world[pe.get_iri().as_str()]
        domains = set(p_x.domains_indirect())
        if len(domains) == 0:
            yield OWLDataPropertyDomainAxiom(pe, OWLThing)
        else:
            for dom in domains:
                if isinstance(dom, owlready2.ThingClass) or isinstance(dom, owlready2.ClassConstruct):
                    yield OWLDataPropertyDomainAxiom(pe, _parse_concept_to_owlapy(dom))
                else:
                    logger.warning("Construct %s not implemented at %s", dom, pe)
                    pass  # XXX TODO

    def data_property_range_axioms(self, pe: OWLDataProperty) -> Iterable[OWLDataPropertyRangeAxiom]:
        p_x: owlready2.DataPropertyClass = self._world[pe.get_iri().as_str()]
        ranges = set(chain.from_iterable(super_prop.range for super_prop in p_x.ancestors()))
        if len(ranges) == 0:
            pass
            # TODO
        else:
            for rng in ranges:
                if rng in _Datatype_map:
                    yield OWLDataPropertyRangeAxiom(pe, _Datatype_map[rng])
                elif isinstance(rng, owlready2.ClassConstruct):
                    yield OWLDataPropertyRangeAxiom(pe, _parse_datarange_to_owlapy(rng))
                else:
                    logger.warning("Datatype %s not implemented at %s", rng, pe)
                    pass  # XXX TODO

    def object_property_domain_axioms(self, pe: OWLObjectProperty) -> Iterable[OWLObjectPropertyDomainAxiom]:
        p_x: owlready2.ObjectPropertyClass = self._world[pe.get_iri().as_str()]
        domains = set(p_x.domains_indirect())
        if len(domains) == 0:
            yield OWLObjectPropertyDomainAxiom(pe, OWLThing)
        else:
            for dom in domains:
                if isinstance(dom, owlready2.ThingClass) or isinstance(dom, owlready2.ClassConstruct):
                    yield OWLObjectPropertyDomainAxiom(pe, _parse_concept_to_owlapy(dom))
                else:
                    logger.warning("Construct %s not implemented at %s", dom, pe)
                    pass  # XXX TODO

    def object_property_range_axioms(self, pe: OWLObjectProperty) -> Iterable[OWLObjectPropertyRangeAxiom]:
        p_x: owlready2.ObjectPropertyClass = self._world[pe.get_iri().as_str()]
        ranges = set(chain.from_iterable(super_prop.range for super_prop in p_x.ancestors()))
        if len(ranges) == 0:
            yield OWLObjectPropertyRangeAxiom(pe, OWLThing)
        else:
            for rng in ranges:
                if isinstance(rng, owlready2.ThingClass) or isinstance(rng, owlready2.ClassConstruct):
                    yield OWLObjectPropertyRangeAxiom(pe, _parse_concept_to_owlapy(rng))
                else:
                    logger.warning("Construct %s not implemented at %s", rng, pe)
                    pass  # XXX TODO

    def __eq__(self, other):
        if type(other) == type(self):
            return self._onto.loaded == other._onto.loaded and self._onto.base_iri == other._onto.base_iri
        return NotImplemented

    def __hash__(self):
        return hash(self._onto.base_iri)

    def __repr__(self):
        return f'OWLOntology_Owlready2({IRI.create(self._onto.base_iri)}, {self._onto.loaded})'


class OWLReasoner_Owlready2(OWLReasonerEx):
    __slots__ = '_ontology', '_world'

    _ontology: OWLOntology_Owlready2
    _world: owlready2.World

    def __init__(self, ontology: OWLOntology_Owlready2):
        super().__init__(ontology)
        assert isinstance(ontology, OWLOntology_Owlready2)
        self._ontology = ontology
        self._world = ontology._world

    def data_property_domains(self, pe: OWLDataProperty, direct: bool = False) -> Iterable[OWLClassExpression]:
        for ax in self.get_root_ontology().data_property_domain_axioms(pe):
            yield ax.get_domain()
            if not direct:
                yield from self.super_classes(ax.get_domain())

    def object_property_domains(self, pe: OWLObjectProperty, direct: bool = False) -> Iterable[OWLClassExpression]:
        for ax in self.get_root_ontology().object_property_domain_axioms(pe):
            yield ax.get_domain()
            if not direct:
                yield from self.super_classes(ax.get_domain())

    def object_property_ranges(self, pe: OWLObjectProperty, direct: bool = False) -> Iterable[OWLClassExpression]:
        for ax in self.get_root_ontology().object_property_range_axioms(pe):
            yield ax.get_range()
            if not direct:
                yield from self.super_classes(ax.get_range())

    def equivalent_classes(self, ce: OWLClassExpression) -> Iterable[OWLClass]:
        """Return the named classes that are directly equivalent to the class expression"""
        if isinstance(ce, OWLClass):
            c_x: owlready2.ThingClass = self._world[ce.get_iri().as_str()]
            for c in c_x.equivalent_to:
                if isinstance(c, owlready2.ThingClass):
                    yield OWLClass(IRI.create(c.iri))
                # Anonymous classes are ignored
        else:
            raise NotImplementedError("equivalent_classes for complex class expressions not implemented", ce)

    def data_property_values(self, ind: OWLNamedIndividual, pe: OWLDataProperty) -> Iterable[OWLLiteral]:
        i: owlready2.Thing = self._world[ind.get_iri().as_str()]
        p: owlready2.DataPropertyClass = self._world[pe.get_iri().as_str()]
        for val in p._get_values_for_individual(i):
            yield OWLLiteral(val)

    def all_data_property_values(self, pe: OWLDataProperty) -> Iterable[OWLLiteral]:
        p: owlready2.DataPropertyClass = self._world[pe.get_iri().as_str()]
        for _, val in p.get_relations():
            yield OWLLiteral(val)

    def object_property_values(self, ind: OWLNamedIndividual, pe: OWLObjectPropertyExpression) \
            -> Iterable[OWLNamedIndividual]:
        if isinstance(pe, OWLObjectProperty):
            i: owlready2.Thing = self._world[ind.get_iri().as_str()]
            p: owlready2.ObjectPropertyClass = self._world[pe.get_iri().as_str()]
            for val in p._get_values_for_individual(i):
                yield OWLNamedIndividual(IRI.create(val.iri))
        elif isinstance(pe, OWLObjectInverseOf):
            i: owlready2.Thing = self._world[ind.get_iri().as_str()]
            p: owlready2.ObjectPropertyClass = self._world[pe.get_named_property().get_iri().as_str()]
            for val in p._get_inverse_values_for_individual(i):
                yield OWLNamedIndividual(IRI.create(val.iri))
        else:
            raise NotImplementedError(pe)

    def flush(self) -> None:
        pass

    def instances(self, ce: OWLClassExpression, direct: bool = False) -> Iterable[OWLNamedIndividual]:
        if direct:
            if isinstance(ce, OWLClass):
                c_x: owlready2.ThingClass = self._world[ce.get_iri().as_str()]
                for i in self._ontology._onto.get_instances_of(c_x):
                    if isinstance(i, owlready2.Thing):
                        yield OWLNamedIndividual(IRI.create(i.iri))
            else:
                raise NotImplementedError("instances for complex class expressions not implemented", ce)
        else:
            if ce.is_owl_thing():
                yield from self._ontology.individuals_in_signature()
            elif isinstance(ce, OWLClass):
                c_x: owlready2.ThingClass = self._world[ce.get_iri().as_str()]
                for i in c_x.instances(world=self._world):
                    if isinstance(i, owlready2.Thing):
                        yield OWLNamedIndividual(IRI.create(i.iri))
            # elif isinstance(ce, OWLObjectSomeValuesFrom) and ce.get_filler().is_owl_thing()\
            #         and isinstance(ce.get_property(), OWLProperty):
            #     seen_set = set()
            #     p_x: owlready2.ObjectProperty = self._world[ce.get_property().get_named_property().get_iri().as_str()]
            #     for i, _ in p_x.get_relations():
            #         if isinstance(i, owlready2.Thing) and i not in seen_set:
            #             seen_set.add(i)
            #             yield OWLNamedIndividual(IRI.create(i.iri))
            else:
                raise NotImplementedError("instances for complex class expressions not implemented", ce)

    def _named_sub_classes_recursive(self, c: OWLClass, seen_set: Set) -> Iterable[OWLClass]:
        c_x: owlready2.EntityClass = self._world[c.get_iri().as_str()]
        # work around issue in class equivalence detection in Owlready2
        for c2 in [c_x, *c_x.equivalent_to]:
            if isinstance(c2, owlready2.ThingClass):
                for sc in c2.subclasses(world=self._world):
                    if isinstance(sc, owlready2.ThingClass) and sc not in seen_set:
                        seen_set.add(sc)
                        owl_sc = OWLClass(IRI.create(sc.iri))
                        yield owl_sc
                        yield from self._named_sub_classes_recursive(owl_sc, seen_set)

    def sub_classes(self, ce: OWLClassExpression, direct: bool = False) -> Iterable[OWLClass]:
        if isinstance(ce, OWLClass):
            if direct:
                c_x: owlready2.ThingClass = self._world[ce.get_iri().as_str()]
                for sc in c_x.subclasses(world=self._world):
                    if isinstance(sc, owlready2.ThingClass):
                        yield OWLClass(IRI.create(sc.iri))
                    # Anonymous classes are ignored
            else:
                # indirect
                seen_set = set()
                yield from self._named_sub_classes_recursive(ce, seen_set)
        else:
            raise NotImplementedError("sub classes for complex class expressions not implemented", ce)

    def super_classes(self, ce: OWLClassExpression, direct: bool = False) -> Iterable[OWLClass]:
        if isinstance(ce, OWLClass):
            c_x: owlready2.ThingClass = self._world[ce.get_iri().as_str()]
            if direct:
                for sc in c_x.is_a:
                    if isinstance(sc, owlready2.ThingClass):
                        yield OWLClass(IRI.create(sc.iri))
                    # Anonymous classes are ignored
            else:
                # indirect
                for sc in c_x.ancestors(include_self=False):
                    if isinstance(sc, owlready2.ThingClass):
                        yield OWLClass(IRI.create(sc.iri))
        else:
            raise NotImplementedError("super classes for complex class expressions not implemented", ce)

    def _sub_data_properties_recursive(self, dp: OWLDataProperty, seen_set: Set) -> Iterable[OWLDataProperty]:
        p_x: owlready2.DataPropertyClass = self._world[dp.get_iri().as_str()]
        assert isinstance(p_x, owlready2.DataPropertyClass)
        for sp_x in p_x.subclasses(world=self._world):
            if isinstance(sp_x, owlready2.DataPropertyClass) and sp_x not in seen_set:
                seen_set.add(sp_x)
                sp = OWLDataProperty(IRI.create(sp_x.iri))
                yield sp
                yield from self._sub_data_properties_recursive(sp, seen_set)

    def sub_data_properties(self, dp: OWLDataProperty, direct: bool = False) -> Iterable[OWLDataProperty]:
        assert isinstance(dp, OWLDataProperty)
        if direct:
            p_x: owlready2.DataPropertyClass = self._world[dp.get_iri().as_str()]
            for sp in p_x.subclasses(world=self._world):
                if isinstance(sp, owlready2.DataPropertyClass):
                    yield OWLDataProperty(IRI.create(sp.iri))
            else:
                seen_set = set()
                yield from self._sub_data_properties_recursive(dp, seen_set)

    def _sub_object_properties_recursive(self, op: OWLObjectProperty, seen_set: Set) -> Iterable[OWLObjectProperty]:
        p_x: owlready2.ObjectPropertyClass = self._world[op.get_iri().as_str()]
        assert isinstance(p_x, owlready2.ObjectPropertyClass)
        for sp_x in p_x.subclasses(world=self._world):
            if isinstance(sp_x, owlready2.ObjectPropertyClass) and sp_x not in seen_set:
                seen_set.add(sp_x)
                sp = OWLObjectProperty(IRI.create(sp_x.iri))
                yield sp
                yield from self._sub_object_properties_recursive(sp, seen_set)

    def sub_object_properties(self, op: OWLObjectPropertyExpression, direct: bool = False) \
            -> Iterable[OWLObjectPropertyExpression]:
        assert isinstance(op, OWLObjectPropertyExpression)
        if isinstance(op, OWLObjectProperty):
            if direct:
                p_x: owlready2.ObjectPropertyClass = self._world[op.get_iri().as_str()]
                for sp in p_x.subclasses(world=self._world):
                    if isinstance(sp, owlready2.ObjectPropertyClass):
                        yield OWLObjectProperty(IRI.create(sp.iri))
            else:
                seen_set = set()
                yield from self._sub_object_properties_recursive(op, seen_set)
        else:
            raise NotImplementedError("sub properties of inverse properties not yet implemented", op)

    def types(self, ind: OWLNamedIndividual, direct: bool = False) -> Iterable[OWLClass]:
        i: owlready2.Thing = self._world[ind.get_iri().as_str()]
        if direct:
            for c in i.is_a:
                if isinstance(c, owlready2.ThingClass):
                    yield OWLClass(IRI.create(c.iri))
                # Anonymous classes are ignored
        else:
            for c in i.INDIRECT_is_a:
                if isinstance(c, owlready2.ThingClass):
                    yield OWLClass(IRI.create(c.iri))
                # Anonymous classes are ignored

    def _sync_reasoner(self, other_reasoner: BaseReasoner_Owlready2 = None,
                       infer_property_values: bool = True,
                       infer_data_property_values: bool = True) -> None:
        """Call Owlready2's sync_reasoner method, which spawns a Java process on a temp file to infer more

        Args:
            other_reasoner: set to BaseReasoner.PELLET (default) or BaseReasoner.HERMIT
            infer_property_values: whether to infer property values
            infer_data_property_values: whether to infer data property values (only for PELLET)
        """
        assert other_reasoner is None or isinstance(other_reasoner, BaseReasoner_Owlready2)
        with self.get_root_ontology()._onto:
            if other_reasoner == BaseReasoner_Owlready2.HERMIT:
                owlready2.sync_reasoner_hermit(self._world, infer_property_values=infer_property_values)
            else:
                owlready2.sync_reasoner_pellet(self._world,
                                               infer_property_values=infer_property_values,
                                               infer_data_property_values=infer_data_property_values)

    def get_root_ontology(self) -> OWLOntology:
        return self._ontology
