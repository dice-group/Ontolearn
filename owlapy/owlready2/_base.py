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
    DurationOWLDatatype, StringOWLDatatype, IRI, OWLDataPropertyRangeAxiom, OWLDataPropertyDomainAxiom, OWLClassAxiom, \
    OWLSubClassOfAxiom, OWLEquivalentClassesAxiom
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
        """saves the actual state of the quadstore in the SQLite3 file
        """
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

    def equivalent_classes_axioms(self, c: OWLClass) -> Iterable[OWLEquivalentClassesAxiom]:
        c_x: owlready2.ThingClass = self._world[c.get_iri().as_str()]
        # TODO: Should this also return EquivalentClasses general class axioms? Compare to java owlapi
        for ec_x in c_x.equivalent_to:
            yield OWLEquivalentClassesAxiom([c, _parse_concept_to_owlapy(ec_x)])

    def general_class_axioms(self) -> Iterable[OWLClassAxiom]:
        # TODO: At the moment owlready2 only supports SubClassOf general class axioms. (18.02.2023)
        for ca in self._onto.general_class_axioms():
            yield from (OWLSubClassOfAxiom(_parse_concept_to_owlapy(ca.left_side), _parse_concept_to_owlapy(c))
                        for c in ca.is_a)

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
                if isinstance(dom, (owlready2.ThingClass, owlready2.ClassConstruct)):
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
                if isinstance(dom, (owlready2.ThingClass, owlready2.ClassConstruct)):
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
                if isinstance(rng, (owlready2.ThingClass, owlready2.ClassConstruct)):
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
        domains = {d.get_domain() for d in self.get_root_ontology().data_property_domain_axioms(pe)}
        # TODO: Remove if when super_classes is implemented for complex class expressions
        super_domains = set(chain.from_iterable([self.super_classes(d) for d in domains if isinstance(d, OWLClass)]))
        yield from domains - super_domains
        if not direct:
            yield from super_domains

    def object_property_domains(self, pe: OWLObjectProperty, direct: bool = False) -> Iterable[OWLClassExpression]:
        domains = {d.get_domain() for d in self.get_root_ontology().object_property_domain_axioms(pe)}
        # TODO: Remove if when super_classes is implemented for complex class expressions
        super_domains = set(chain.from_iterable([self.super_classes(d) for d in domains if isinstance(d, OWLClass)]))
        yield from domains - super_domains
        if not direct:
            yield from super_domains

    def object_property_ranges(self, pe: OWLObjectProperty, direct: bool = False) -> Iterable[OWLClassExpression]:
        ranges = {r.get_range() for r in self.get_root_ontology().object_property_range_axioms(pe)}
        # TODO: Remove if when super_classes is implemented for complex class expressions
        super_ranges = set(chain.from_iterable([self.super_classes(d) for d in ranges if isinstance(d, OWLClass)]))
        yield from ranges - super_ranges
        if not direct:
            yield from super_ranges

    def equivalent_classes(self, ce: OWLClassExpression, only_named: bool = True) -> Iterable[OWLClassExpression]:
        seen_set = {ce}
        if isinstance(ce, OWLClass):
            c_x: owlready2.ThingClass = self._world[ce.get_iri().as_str()]
            for eq_x in c_x.INDIRECT_equivalent_to:
                eq = _parse_concept_to_owlapy(eq_x)
                if (isinstance(eq, OWLClass) or
                        (isinstance(eq, OWLClassExpression) and not only_named)) and eq not in seen_set:
                    seen_set.add(eq)
                    yield eq
                # Workaround for problems in owlready2. It does not always recognize equivalent complex class
                # expressions through INDIRECT_equivalent_to. Maybe it will work as soon as owlready2 adds support for
                # EquivalentClasses general class axioms.
                if not only_named and isinstance(eq_x, owlready2.ThingClass):
                    for eq_2_x in eq_x.equivalent_to:
                        eq_2 = _parse_concept_to_owlapy(eq_2_x)
                        if eq_2 not in seen_set:
                            seen_set.add(eq_2)
                            yield eq_2
        elif isinstance(ce, OWLClassExpression):
            # Extend as soon as owlready2 supports EquivalentClasses general class axioms
            # Slow but works. No better way to do this in owlready2 without using the reasoners at the moment.
            # Might be able to change this when owlready2 supports general class axioms for EquivalentClasses.
            for c in self._ontology.classes_in_signature():
                if ce in self.equivalent_classes(c, only_named=False) and c not in seen_set:
                    seen_set.add(c)
                    yield c
                    for e_c in self.equivalent_classes(c, only_named=False):
                        if e_c not in seen_set and (not only_named or isinstance(e_c, OWLClass)):
                            seen_set.add(e_c)
                            yield e_c
        else:
            raise ValueError(f'Equivalent classes not implemented for: {ce}')

    def disjoint_classes(self, ce: OWLClassExpression, only_named: bool = True) -> Iterable[OWLClassExpression]:
        if isinstance(ce, OWLClass):
            c_x: owlready2.ThingClass = self._world[ce.get_iri().as_str()]
            for d_x in chain.from_iterable(map(lambda d: d.entities, c_x.disjoints())):
                if d_x != c_x and (isinstance(d_x, owlready2.ThingClass) or
                                   (isinstance(d_x, owlready2.ClassConstruct) and not only_named)):
                    yield _parse_concept_to_owlapy(d_x)
        elif isinstance(ce, OWLClassExpression):
            # Extend as soon as owlready2 supports DisjointClasses general class axioms
            # Slow but works. No better way to do this in owlready2 without using the reasoners at the moment.
            # Might be able to change this when owlready2 supports general class axioms for DjsjointClasses
            yield from (c for c in self._ontology.classes_in_signature() if ce in self.disjoint_classes(c, False))
        else:
            raise ValueError(f'Equivalent classes not implemented for: {ce}')

    def different_individuals(self, ind: OWLNamedIndividual) -> Iterable[OWLNamedIndividual]:
        i: owlready2.Thing = self._world[ind.get_iri().as_str()]
        yield from (OWLNamedIndividual(IRI.create(d_i.iri))
                    for d_i in chain.from_iterable(map(lambda x: x.entities, i.differents()))
                    if isinstance(d_i, owlready2.Thing) and i != d_i)

    def same_individuals(self, ind: OWLNamedIndividual) -> Iterable[OWLNamedIndividual]:
        i: owlready2.Thing = self._world[ind.get_iri().as_str()]
        yield from (OWLNamedIndividual(IRI.create(d_i.iri)) for d_i in i.equivalent_to
                    if isinstance(d_i, owlready2.Thing))

    def data_property_values(self, ind: OWLNamedIndividual, pe: OWLDataProperty, direct: bool = True) \
            -> Iterable[OWLLiteral]:
        i: owlready2.Thing = self._world[ind.get_iri().as_str()]
        p: owlready2.DataPropertyClass = self._world[pe.get_iri().as_str()]
        retrieval_func = p._get_values_for_individual if direct else p._get_indirect_values_for_individual
        for val in retrieval_func(i):
            yield OWLLiteral(val)

    def all_data_property_values(self, pe: OWLDataProperty, direct: bool = True) -> Iterable[OWLLiteral]:
        p: owlready2.DataPropertyClass = self._world[pe.get_iri().as_str()]
        relations = p.get_relations()
        if not direct:
            indirect_relations = chain.from_iterable(
                map(lambda x: self._world[x.get_iri().as_str()].get_relations(),
                    self.sub_data_properties(pe, direct=False)))
            relations = chain(relations, indirect_relations)
        for _, val in relations:
            yield OWLLiteral(val)

    def object_property_values(self, ind: OWLNamedIndividual, pe: OWLObjectPropertyExpression, direct: bool = True) \
            -> Iterable[OWLNamedIndividual]:
        if isinstance(pe, OWLObjectProperty):
            i: owlready2.Thing = self._world[ind.get_iri().as_str()]
            p: owlready2.ObjectPropertyClass = self._world[pe.get_iri().as_str()]
            retieval_func = p._get_values_for_individual if direct else p._get_indirect_values_for_individual
            for val in retieval_func(i):
                yield OWLNamedIndividual(IRI.create(val.iri))
        elif isinstance(pe, OWLObjectInverseOf):
            p: owlready2.ObjectPropertyClass = self._world[pe.get_named_property().get_iri().as_str()]
            inverse_p = p.inverse_property
            # If the inverse property is explicitly defined we can take shortcut
            if inverse_p is not None:
                yield from self.object_property_values(ind, OWLObjectProperty(IRI.create(inverse_p.iri)), direct)
            else:
                if not direct:
                    raise NotImplementedError('Indirect values of inverse properties are only implemented if the '
                                              'inverse property is explicitly defined in the ontology.'
                                              f'Property: {pe}')
                i: owlready2.Thing = self._world[ind.get_iri().as_str()]
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

    def _sub_classes_recursive(self, ce: OWLClassExpression, seen_set: Set, only_named: bool = True) \
            -> Iterable[OWLClassExpression]:

        # work around issue in class equivalence detection in Owlready2
        for c in [ce, *self.equivalent_classes(ce, only_named=False)]:
            # First go through all general class axioms, they should only have complex classes as sub_classes.
            # Done for OWLClass and OWLClassExpression.
            for axiom in self._ontology.general_class_axioms():
                if (isinstance(axiom, OWLSubClassOfAxiom) and axiom.get_super_class() == c
                        and axiom.get_sub_class() not in seen_set):
                    seen_set.add(axiom.get_sub_class())
                    if not only_named:
                        yield axiom.get_sub_class()
                    yield from self._sub_classes_recursive(axiom.get_sub_class(), seen_set, only_named)

            if isinstance(c, OWLClass):
                c_x: owlready2.EntityClass = self._world[c.get_iri().as_str()]
                # Subclasses will only return named classes
                for sc_x in c_x.subclasses(world=self._world):
                    sc = _parse_concept_to_owlapy(sc_x)
                    if isinstance(sc, OWLClass) and sc not in seen_set:
                        seen_set.add(sc)
                        yield sc
                        yield from self._sub_classes_recursive(sc, seen_set, only_named=only_named)
            elif isinstance(c, OWLClassExpression):
                # Slow but works. No better way to do this in owlready2 without using the reasoners at the moment.
                for atomic_c in self._ontology.classes_in_signature():
                    if c in self.super_classes(atomic_c, direct=True, only_named=False) and atomic_c not in seen_set:
                        seen_set.add(atomic_c)
                        yield atomic_c
                        yield from self._sub_classes_recursive(atomic_c, seen_set, only_named=only_named)
            else:
                raise ValueError(f'Sub classes retrieval not implemented for: {ce}')

    def sub_classes(self, ce: OWLClassExpression, direct: bool = False, only_named: bool = True) \
            -> Iterable[OWLClassExpression]:
        if not direct:
            seen_set = set()
            yield from self._sub_classes_recursive(ce, seen_set, only_named=only_named)
        else:
            # First go through all general class axioms, they should only have complex classes as sub_classes.
            # Done for OWLClass and OWLClassExpression.
            if not only_named:
                for axiom in self._ontology.general_class_axioms():
                    if isinstance(axiom, OWLSubClassOfAxiom) and axiom.get_super_class() == ce:
                        yield axiom.get_sub_class()

            if isinstance(ce, OWLClass):
                c_x: owlready2.ThingClass = self._world[ce.get_iri().as_str()]
                # Subclasses will only return named classes
                for sc in c_x.subclasses(world=self._world):
                    if isinstance(sc, owlready2.ThingClass):
                        yield OWLClass(IRI.create(sc.iri))
            elif isinstance(ce, OWLClassExpression):
                # Slow but works. No better way to do this in owlready2 without using the reasoners at the moment.
                for c in self._ontology.classes_in_signature():
                    if ce in self.super_classes(c, direct=True, only_named=False):
                        yield c
            else:
                raise ValueError(f'Sub classes retrieval not implemented for: {ce}')

    def _super_classes_recursive(self, ce: OWLClassExpression, seen_set: Set, only_named: bool = True) \
            -> Iterable[OWLClassExpression]:
        # work around issue in class equivalence detection in Owlready2
        for c in [ce, *self.equivalent_classes(ce, only_named=False)]:
            if isinstance(c, OWLClass):
                c_x: owlready2.EntityClass = self._world[c.get_iri().as_str()]
                for sc_x in c_x.is_a:
                    sc = _parse_concept_to_owlapy(sc_x)
                    if ((isinstance(sc, OWLClass) or isinstance(sc, OWLClassExpression)) and sc not in seen_set):
                        seen_set.add(sc)
                        # Return class expression if it is a named class or complex class expressions should be
                        # included
                        if isinstance(sc, OWLClass) or not only_named:
                            yield sc
                        yield from self._super_classes_recursive(sc, seen_set, only_named=only_named)
            elif isinstance(c, OWLClassExpression):
                for axiom in self._ontology.general_class_axioms():
                    if (isinstance(axiom, OWLSubClassOfAxiom) and axiom.get_sub_class() == c
                            and (axiom.get_super_class() not in seen_set)):
                        super_class = axiom.get_super_class()
                        seen_set.add(super_class)
                        # Return class expression if it is a named class or complex class expressions should be
                        # included
                        if isinstance(super_class, OWLClass) or not only_named:
                            yield super_class
                        yield from self._super_classes_recursive(super_class, seen_set, only_named=only_named)

                # Slow but works. No better way to do this in owlready2 without using the reasoners at the moment.
                for atomic_c in self._ontology.classes_in_signature():
                    if c in self.sub_classes(atomic_c, direct=True, only_named=False) and atomic_c not in seen_set:
                        seen_set.add(atomic_c)
                        yield atomic_c
                        yield from self._super_classes_recursive(atomic_c, seen_set, only_named=only_named)
            else:
                raise ValueError(f'Super classes retrieval not supported for: {ce}')

    def super_classes(self, ce: OWLClassExpression, direct: bool = False, only_named: bool = True) \
            -> Iterable[OWLClassExpression]:
        if not direct:
            seen_set = set()
            yield from self._super_classes_recursive(ce, seen_set, only_named=only_named)
        else:
            if isinstance(ce, OWLClass):
                c_x: owlready2.ThingClass = self._world[ce.get_iri().as_str()]
                for sc in c_x.is_a:
                    if (isinstance(sc, owlready2.ThingClass) or
                            (not only_named and isinstance(sc, owlready2.ClassConstruct))):
                        yield _parse_concept_to_owlapy(sc)
            elif isinstance(ce, OWLClassExpression):
                seen_set = set()
                for axiom in self._ontology.general_class_axioms():
                    if (isinstance(axiom, OWLSubClassOfAxiom) and axiom.get_sub_class() == ce
                            and (not only_named or isinstance(axiom.get_super_class(), OWLClass))):
                        seen_set.add(axiom.get_super_class())
                        yield axiom.get_super_class()
                # Slow but works. No better way to do this in owlready2 without using the reasoners at the moment.
                # TODO: Might not be needed, in theory the general class axioms above should cover all classes
                # that can be found here
                for c in self._ontology.classes_in_signature():
                    if ce in self.sub_classes(c, direct=True, only_named=False) and c not in seen_set:
                        seen_set.add(c)
                        yield c
            else:
                raise ValueError(f'Super classes retrieval not supported for {ce}')

    def equivalent_object_properties(self, op: OWLObjectPropertyExpression) -> Iterable[OWLObjectPropertyExpression]:
        if isinstance(op, OWLObjectProperty):
            p_x: owlready2.ObjectPropertyClass = self._world[op.get_iri().as_str()]
            yield from (OWLObjectProperty(IRI.create(ep_x.iri)) for ep_x in p_x.INDIRECT_equivalent_to
                        if isinstance(ep_x, owlready2.ObjectPropertyClass))
        else:
            raise NotImplementedError("equivalent properties of inverse properties not yet implemented", op)

    def equivalent_data_properties(self, dp: OWLDataProperty) -> Iterable[OWLDataProperty]:
        p_x: owlready2.DataPropertyClass = self._world[dp.get_iri().as_str()]
        yield from (OWLDataProperty(IRI.create(ep_x.iri)) for ep_x in p_x.INDIRECT_equivalent_to
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
        elif isinstance(op, OWLObjectInverseOf):
            p: owlready2.ObjectPropertyClass = self._world[op.get_named_property().get_iri().as_str()]
            inverse_p = p.inverse_property
            if inverse_p is not None:
                yield from self.sub_object_properties(OWLObjectProperty(IRI.create(inverse_p.iri)), direct)
            else:
                raise NotImplementedError('Sub properties of inverse properties are only implemented if the '
                                          'inverse property is explicitly defined in the ontology. '
                                          f'Property: {op}')
        else:
            raise NotImplementedError(op)

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
                       infer_data_property_values: bool = True, debug: bool = False) -> None:
        """Call Owlready2's sync_reasoner method, which spawns a Java process on a temp file to infer more

        Args:
            other_reasoner: set to BaseReasoner.PELLET (default) or BaseReasoner.HERMIT
            infer_property_values: whether to infer property values
            infer_data_property_values: whether to infer data property values (only for PELLET)
        """
        assert other_reasoner is None or isinstance(other_reasoner, BaseReasoner_Owlready2)
        with self.get_root_ontology()._onto:
            if other_reasoner == BaseReasoner_Owlready2.HERMIT:
                owlready2.sync_reasoner_hermit(self._world, infer_property_values=infer_property_values, debug=debug)
            else:
                owlready2.sync_reasoner_pellet(self._world,
                                               infer_property_values=infer_property_values,
                                               infer_data_property_values=infer_data_property_values,
                                               debug=debug)

    def get_root_ontology(self) -> OWLOntology:
        return self._ontology
