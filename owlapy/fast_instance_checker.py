import operator
from functools import singledispatchmethod, reduce
from logging import warning
from types import MappingProxyType
from typing import Iterable, Dict, Mapping

from owlapy import IRI
from owlapy.model import OWLReasoner, OWLOntology, OWLNamedIndividual, OWLClass, OWLClassExpression, \
    OWLObjectProperty, OWLDataProperty, OWLObjectUnionOf, OWLObjectIntersectionOf, OWLObjectSomeValuesFrom, \
    OWLObjectPropertyExpression, OWLObjectComplementOf, OWLObjectAllValuesFrom
from owlapy.util import NamedFixedSet


class OWLReasoner_FastInstanceChecker(OWLReasoner):
    """Tries to check instances fast (but maybe incomplete)"""
    __slots__ = '_ontology', '_base_reasoner', \
                '_ind_enc', '_cls_to_ind', '_obj_prop', '_objectsomevalues_cache', \
                '_negation_default'

    _ontology: OWLOntology
    _base_reasoner: OWLReasoner
    _cls_to_ind: Dict[OWLClass, int]  # Class => individuals
    _obj_prop: Dict[OWLObjectProperty, Mapping[int, int]]  # ObjectProperty => { individual => individuals }
    _ind_enc: NamedFixedSet[OWLNamedIndividual]
    _objectsomevalues_cache: Dict[OWLClassExpression, int]  # ObjectSomeValuesFrom => individuals

    def __init__(self, ontology: OWLOntology, base_reasoner: OWLReasoner, *, negation_default=False):
        """Fast instance checker

        Args:
            ontology: Ontology to use
            base_reasoner: Reasoner to get instances/types from"""
        super().__init__(ontology)
        self._ontology = ontology
        self._base_reasoner = base_reasoner
        self._negation_default = negation_default
        self._init()

    def _init(self):
        self._cls_to_ind = dict()
        self._obj_prop = dict()
        individuals = self._ontology.individuals_in_signature()
        self._ind_enc = NamedFixedSet(OWLNamedIndividual, individuals)
        self._objectsomevalues_cache = dict()

    def reset(self):
        """The reset method shall reset any cached state"""
        self._init()

    def data_property_domains(self, pe: OWLDataProperty, direct: bool = False) -> Iterable[OWLClass]:
        yield from self._base_reasoner.data_property_domains(pe, direct=direct)

    def object_property_domains(self, pe: OWLObjectProperty, direct: bool = False) -> Iterable[OWLClass]:
        yield from self._base_reasoner.object_property_domains(pe, direct=direct)

    def object_property_ranges(self, pe: OWLObjectProperty, direct: bool = False) -> Iterable[OWLClass]:
        yield from self._base_reasoner.object_property_ranges(pe, direct=direct)

    def equivalent_classes(self, ce: OWLClassExpression) -> Iterable[OWLClass]:
        yield from self._base_reasoner.equivalent_classes(ce)

    def data_property_values(self, ind: OWLNamedIndividual, pe: OWLDataProperty) -> Iterable:
        yield from self._base_reasoner.data_property_values(ind, pe)

    def object_property_values(self, ind: OWLNamedIndividual, pe: OWLObjectProperty) -> Iterable[OWLNamedIndividual]:
        self._lazy_cache_obj_prop(pe)
        ind_enc = self._ind_enc(ind)
        yield from self._ind_enc(self._obj_prop[pe][ind_enc])

    def flush(self) -> None:
        self._base_reasoner.flush()

    def instances(self, ce: OWLClassExpression, direct: bool = False) -> Iterable[OWLNamedIndividual]:
        if direct:
            warning("direct not implemented")
        temp = self._find_instances(ce)
        yield from self._ind_enc(temp)

    def sub_classes(self, ce: OWLClassExpression, direct: bool = False) -> Iterable[OWLClass]:
        yield from self._base_reasoner.sub_classes(ce, direct=direct)

    def types(self, ind: OWLNamedIndividual, direct: bool = False) -> Iterable[OWLClass]:
        yield from self._base_reasoner.types(ind, direct=direct)

    def sub_data_properties(self, dp: OWLDataProperty, direct: bool = False) -> Iterable[OWLDataProperty]:
        yield from self._base_reasoner.sub_data_properties(dp=dp, direct=direct)

    def sub_object_properties(self, op: OWLObjectPropertyExpression, direct: bool = False)\
            -> Iterable[OWLObjectPropertyExpression]:
        yield from self._base_reasoner.sub_object_properties(op=op, direct=direct)

    def get_root_ontology(self) -> OWLOntology:
        return self._ontology

    def _lazy_cache_obj_prop(self, pe: OWLObjectProperty) -> None:
        """Get all individuals involved in this object property and put them in a Dict"""
        if pe in self._obj_prop:
            return

        # Dict with Individual => Set[Individual]
        opc: Dict[int, int] = dict()

        # shortcut for owlready2
        from owlapy.owlready2 import OWLOntology_Owlready2
        if isinstance(self._ontology, OWLOntology_Owlready2):
            import owlready2
            # _x => owlready2 objects
            p_x: owlready2.ObjectProperty = self._ontology._world[pe.get_iri().as_str()]
            for s_x, o_x in p_x.get_relations():
                if isinstance(s_x, owlready2.Thing) and isinstance(o_x, owlready2.Thing):
                    s_enc = self._ind_enc(OWLNamedIndividual(IRI.create(s_x.iri)))
                    o_enc = self._ind_enc(OWLNamedIndividual(IRI.create(o_x.iri)))
                    if s_enc in opc:
                        opc[s_enc] |= o_enc
                    else:
                        opc[s_enc] = o_enc
        else:
            for s_enc, s in self._ind_enc.items():
                opc[s_enc] = self._ind_enc(self._base_reasoner.object_property_values(s, pe))

        self._obj_prop[pe] = MappingProxyType(opc)

    # single dispatch is still not implemented in mypy, see https://github.com/python/mypy/issues/2904
    @singledispatchmethod
    def _find_instances(self, ce: OWLClassExpression) -> int:
        raise NotImplementedError(ce)

    @_find_instances.register
    def _(self, c: OWLClass) -> int:
        self._lazy_cache_class(c)
        return self._cls_to_ind[c]

    @_find_instances.register
    def _(self, ce: OWLObjectUnionOf):
        return reduce(operator.or_, map(self._find_instances, ce.operands()))

    @_find_instances.register
    def _(self, ce: OWLObjectIntersectionOf):
        return reduce(operator.and_, map(self._find_instances, ce.operands()))

    @_find_instances.register
    def _(self, ce: OWLObjectSomeValuesFrom):
        if ce in self._objectsomevalues_cache:
            return self._objectsomevalues_cache[ce]

        p = ce.get_property()
        assert isinstance(p, OWLObjectProperty)
        self._lazy_cache_obj_prop(p)

        filler_ind_enc = self._find_instances(ce.get_filler())
        ind_enc = 0
        for s_enc, o_set_enc in self._obj_prop[p].items():
            if o_set_enc & filler_ind_enc:
                ind_enc |= s_enc

        self._objectsomevalues_cache[ce] = ind_enc
        return ind_enc

    @_find_instances.register
    def _(self, ce: OWLObjectComplementOf):
        if self._negation_default:
            all = (1 << len(self._ind_enc)) - 1
            complement_ind_enc = self._find_instances(ce.get_operand())
            return all ^ complement_ind_enc
        else:
            # TODO! XXX
            return 0
            # if self.complement_as_negation:
            #     ...
            # else:
            #     self._lazy_cache_negation

    @_find_instances.register
    def _(self, ce: OWLObjectAllValuesFrom):
        # TODO! XXX
        return 0

    def _lazy_cache_class(self, c: OWLClass) -> None:
        if c in self._cls_to_ind:
            return
        temp = self._base_reasoner.instances(c)
        self._cls_to_ind[c] = self._ind_enc(temp)
