import logging
from datetime import date, datetime
from enum import Enum, auto
from itertools import chain
from types import MappingProxyType
from typing import Iterable, Set, Final

import owlready2
from owlready2 import declare_datatype
from pandas import Timedelta

from owlapy.owlready2 import axioms
from owlapy import namespaces
from owlapy.ext import OWLReasonerEx
from owlapy.model import OWLObjectPropertyRangeAxiom, OWLOntologyManager, OWLDataProperty, OWLObjectProperty, \
    OWLNamedIndividual, OWLClassExpression, OWLObjectPropertyExpression, OWLOntologyID, OWLAxiom, OWLOntology, \
    OWLOntologyChange, AddImport, OWLThing, DoubleOWLDatatype, OWLObjectPropertyDomainAxiom, OWLLiteral, \
    OWLObjectInverseOf, BooleanOWLDatatype, IntegerOWLDatatype, DateOWLDatatype, DateTimeOWLDatatype, OWLClass, \
    DurationOWLDatatype, StringOWLDatatype, IRI, OWLDataPropertyRangeAxiom, OWLDataPropertyDomainAxiom
from owlapy.owlready2.utils import FromOwlready2

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

    def add_axiom(self, ontology: OWLOntology, axiom: OWLAxiom):
        axioms._add_axiom(axiom, ontology, self._world)

    def remove_axiom(self, ontology: OWLOntology, axiom: OWLAxiom):
        axioms._remove_axiom(axiom, ontology, self._world)

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

    def disjoint_classes(self, ce: OWLClassExpression) -> Iterable[OWLClass]:
        if isinstance(ce, OWLClass):
            c_x: owlready2.ThingClass = self._world[ce.get_iri().as_str()]
            for c in chain.from_iterable(map(lambda d: d.entities, c_x.disjoints())):
                if isinstance(c, owlready2.ThingClass) and c != c_x:
                    yield OWLClass(IRI.create(c.iri))
                # Anonymous classes are ignored
        else:
            raise NotImplementedError("disjoint_classes for complex class expressions not implemented", ce)

    def different_individuals(self, ind: OWLNamedIndividual) -> Iterable[OWLNamedIndividual]:
        i: owlready2.Thing = self._world[ind.get_iri().as_str()]
        yield from (OWLNamedIndividual(IRI.create(d_i.iri))
                    for d_i in chain.from_iterable(map(lambda x: x.entities, i.differents()))
                    if isinstance(d_i, owlready2.Thing) and i != d_i)

    def same_individuals(self, ind: OWLNamedIndividual) -> Iterable[OWLNamedIndividual]:
        i: owlready2.Thing = self._world[ind.get_iri().as_str()]
        yield from (OWLNamedIndividual(IRI.create(d_i.iri)) for d_i in i.equivalent_to
                    if isinstance(d_i, owlready2.Thing))

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

    def equivalent_object_properties(self, op: OWLObjectPropertyExpression) -> Iterable[OWLObjectPropertyExpression]:
        if isinstance(op, OWLObjectProperty):
            p_x: owlready2.ObjectPropertyClass = self._world[op.get_iri().as_str()]
            yield from (OWLObjectProperty(IRI.create(ep_x.iri)) for ep_x in p_x.equivalent_to
                        if isinstance(ep_x, owlready2.ObjectPropertyClass))
        else:
            raise NotImplementedError("equivalent properties of inverse properties not yet implemented", op)

    def equivalent_data_properties(self, dp: OWLDataProperty) -> Iterable[OWLDataProperty]:
        p_x: owlready2.DataPropertyClass = self._world[dp.get_iri().as_str()]
        yield from (OWLDataProperty(IRI.create(ep_x.iri)) for ep_x in p_x.equivalent_to
                    if isinstance(ep_x, owlready2.DataPropertyClass))

    def disjoint_object_properties(self, op: OWLObjectPropertyExpression) -> Iterable[OWLObjectPropertyExpression]:
        if isinstance(op, OWLObjectProperty):
            p_x: owlready2.ObjectPropertyClass = self._world[op.get_iri().as_str()]
            ont_x: owlready2.Ontology = self.get_root_ontology()._onto
            for disjoint in ont_x.disjoint_properties():
                if p_x in disjoint.entities:
                    yield from (OWLObjectProperty(IRI.create(o_p.iri)) for o_p in disjoint.entities
                                if isinstance(o_p, owlready2.ObjectPropertyClass) and o_p != p_x)
        else:
            raise NotImplementedError("disjoint object properties of inverse properties not yet implemented", op)

    def disjoint_data_properties(self, dp: OWLDataProperty) -> Iterable[OWLDataProperty]:
        p_x: owlready2.DataPropertyClass = self._world[dp.get_iri().as_str()]
        ont_x: owlready2.Ontology = self.get_root_ontology()._onto
        for disjoint in ont_x.disjoint_properties():
            if p_x in disjoint.entities:
                yield from (OWLDataProperty(IRI.create(o_p.iri)) for o_p in disjoint.entities
                            if isinstance(o_p, owlready2.DataPropertyClass) and o_p != p_x)

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
