from enum import Enum, auto
from logging import warning
from typing import Iterable, Set, Final

import owlready2

from owlapy import IRI, namespaces
from owlapy.model import OWLOntologyManager, OWLOntology, OWLClass, OWLDataProperty, OWLObjectProperty, \
    OWLNamedIndividual, OWLReasoner, OWLClassExpression, OWLObjectPropertyExpression, OWLOntologyID \
    # OWLObjectSomeValuesFrom, OWLProperty, \
from owlapy.namespaces import Namespaces


class BaseReasoner(Enum):
    PELLET = auto()


class OWLOntologyManager_Owlready2(OWLOntologyManager):
    __slots__ = ()

    def create_ontology(self, iri: IRI) -> 'OWLOntology_Owlready2':
        return OWLOntology_Owlready2(self, iri, load=False)

    def load_ontology(self, iri: IRI) -> 'OWLOntology_Owlready2':
        return OWLOntology_Owlready2(self, iri, load=True)


_VERSION_IRI: Final = IRI.create(namespaces.OWL, "versionIRI")


class OWLOntology_Owlready2(OWLOntology):
    __slots__ = '_manager', '_world', '_onto'

    _manager: OWLOntologyManager_Owlready2
    _onto: owlready2.Ontology
    _world: owlready2.World

    def __init__(self, manager: OWLOntologyManager_Owlready2, ontology_iri: IRI, load: bool):
        self._manager = manager
        self._world = owlready2.World()
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

    def get_manager(self) -> OWLOntologyManager_Owlready2:
        return self._manager

    def get_ontology_id(self) -> OWLOntologyID:
        onto_iri = self._world._unabbreviate(self._onto.storid)
        look_version =  self._world.get_triples(
            self._onto.storid,
            self._world._abbreviate(_VERSION_IRI.as_str()))
        if look_version:  # look_version = [ (s_id, p_id, o_id), ... ]
            version_iri = self._world._unabbreviate(look_version[0][2])
        else:
            version_iri = None

        return OWLOntologyID(IRI.create(onto_iri) if onto_iri is not None else None,
                             IRI.create(version_iri) if version_iri is not None else None)

    def __eq__(self, other):
        if type(other) == type(self):
            return self._onto.loaded == other._onto.loaded and self._onto.base_iri == other._onto.base_iri
        return NotImplemented

    def __hash__(self):
        return hash(self._onto.base_iri)

    def __repr__(self):
        return f'OWLOntology_Owlready2({IRI.create(self._onto.base_iri)}, {self._onto.loaded})'


class OWLReasoner_Owlready2(OWLReasoner):
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
            yield OWLClass(IRI.create(dom.iri))

    def object_property_ranges(self, pe: OWLObjectProperty, direct: bool = False) -> Iterable[OWLClass]:
        if direct:
            warning("direct not implemented")
        pe_x: owlready2.ObjectPropertyClass = self._world[pe.get_iri().as_str()]
        for rng in pe_x.range:
            yield OWLClass(IRI.create(rng.iri))

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

    def data_property_values(self, ind: OWLNamedIndividual, pe: OWLDataProperty) -> Iterable:
        i: owlready2.Thing = self._world[ind.get_iri().as_str()]
        p: owlready2.DataPropertyClass = self._world[pe.get_iri().as_str()]
        for val in p._get_values_for_individual(i):
            yield val

    def object_property_values(self, ind: OWLNamedIndividual, pe: OWLObjectProperty) -> Iterable[OWLNamedIndividual]:
        i: owlready2.Thing = self._world[ind.get_iri().as_str()]
        p: owlready2.ObjectPropertyClass = self._world[pe.get_iri().as_str()]
        for val in p._get_values_for_individual(i):
            yield OWLNamedIndividual(IRI.create(val.iri))

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

    def sub_object_properties(self, op: OWLObjectPropertyExpression, direct: bool = False)\
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

    def _sync_reasoner(self, other_reasoner: BaseReasoner = None, **kwargs) -> None:
        """Call Owlready2's sync_reasoner method, which spawns a Java process on a temp file to infer more

        Keyword arguments:
            other_reasoner -- set to BaseReasoner.PELLET to use pellet
        """
        assert other_reasoner is None or isinstance(other_reasoner, BaseReasoner)
        if other_reasoner == BaseReasoner.PELLET:
            owlready2.sync_reasoner_pellet(self._world, **kwargs)
        else:
            owlready2.sync_reasoner(self._world, **kwargs)

    def get_root_ontology(self) -> OWLOntology:
        return self._ontology
