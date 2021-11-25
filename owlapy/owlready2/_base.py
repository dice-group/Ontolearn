import logging
import types
from datetime import date, datetime
from enum import Enum, auto
from logging import warning
from types import MappingProxyType
from typing import Iterable, Set, Final, cast

import owlready2
from owlready2 import declare_datatype
from pandas import Timedelta

from owlapy import namespaces
from owlapy.ext import OWLReasonerEx
from owlapy.model import OWLLiteral, OWLOntologyManager, OWLOntology, OWLClass, OWLDataProperty, OWLObjectProperty, \
    OWLNamedIndividual, OWLClassExpression, OWLObjectPropertyExpression, OWLOntologyID, OWLAxiom, \
    OWLOntologyChange, AddImport, OWLEquivalentClassesAxiom, OWLThing, OWLAnnotationAssertionAxiom, DoubleOWLDatatype, \
    OWLObjectInverseOf, BooleanOWLDatatype, IntegerOWLDatatype, DateOWLDatatype, DateTimeOWLDatatype, \
    DurationOWLDatatype, IRI, OWLDataPropertyRangeAxiom
from owlapy.owlready2.utils import ToOwlready2

logger = logging.getLogger(__name__)

_Datatype_map: Final = MappingProxyType({
    int: IntegerOWLDatatype,
    float: DoubleOWLDatatype,
    bool: BooleanOWLDatatype,
    date: DateOWLDatatype,
    datetime: DateTimeOWLDatatype,
    Timedelta: DurationOWLDatatype,
})

_VERSION_IRI: Final = IRI.create(namespaces.OWL, "versionIRI")


def _parse_duration_datatype(literal: str):
    return Timedelta(literal)


def _unparse_duration_datatype(literal: Timedelta):
    return literal.isoformat()


declare_datatype(Timedelta, IRI.create(namespaces.XSD, "duration").as_str(),
                 _parse_duration_datatype, _unparse_duration_datatype)


class BaseReasoner_Owlready2(Enum):
    PELLET = auto()


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

    def add_axiom(self, ontology: OWLOntology, axiom: OWLAxiom):
        conv = ToOwlready2(self._world)
        ont_x: owlready2.namespace.Ontology = conv.map_object(ontology)

        if isinstance(axiom, OWLEquivalentClassesAxiom):
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
            raise NotImplementedError

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

    def data_property_range_axioms(self, property: OWLDataProperty) -> Iterable[OWLDataPropertyRangeAxiom]:
        p_x: owlready2.DataPropertyClass = self._world[property.get_iri().as_str()]
        for rng in p_x.range:
            if rng in _Datatype_map:
                yield OWLDataPropertyRangeAxiom(property, _Datatype_map[rng])
            else:
                logger.warning("Datatype %s not implemented at %s", rng, property)
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

    def data_property_domains(self, pe: OWLDataProperty, direct: bool = False) -> Iterable[OWLClass]:
        if direct:
            warning("direct not implemented")
        pe_x: owlready2.DataPropertyClass = self._world[pe.get_iri().as_str()]
        for dom in pe_x.domain:
            yield OWLClass(IRI.create(dom.iri))

    def object_property_domains(self, pe: OWLObjectProperty, direct: bool = False) -> Iterable[OWLClass]:
        if direct:
            warning("direct not implemented")
        pe_x: owlready2.ObjectPropertyClass = self._world[pe.get_iri().as_str()]
        for dom in pe_x.domain:
            if isinstance(dom, owlready2.ThingClass):
                yield OWLClass(IRI.create(dom.iri))
            else:
                pass  # XXX TODO

    def object_property_ranges(self, pe: OWLObjectProperty, direct: bool = False) -> Iterable[OWLClass]:
        if direct:
            warning("direct not implemented")
        pe_x: owlready2.ObjectPropertyClass = self._world[pe.get_iri().as_str()]
        for rng in pe_x.range:
            if isinstance(rng, owlready2.ThingClass):
                yield OWLClass(IRI.create(rng.iri))
            else:
                pass  # XXX TODO

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

    def _sync_reasoner(self, other_reasoner: BaseReasoner_Owlready2 = None, **kwargs) -> None:
        """Call Owlready2's sync_reasoner method, which spawns a Java process on a temp file to infer more

        Keyword arguments:
            other_reasoner -- set to BaseReasoner.PELLET to use pellet
        """
        assert other_reasoner is None or isinstance(other_reasoner, BaseReasoner_Owlready2)
        if other_reasoner == BaseReasoner_Owlready2.PELLET:
            owlready2.sync_reasoner_pellet(self._world, **kwargs)
        else:
            owlready2.sync_reasoner(self._world, **kwargs)

    def get_root_ontology(self) -> OWLOntology:
        return self._ontology
