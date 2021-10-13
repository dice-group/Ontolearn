import logging
import operator
from functools import singledispatchmethod, reduce
from itertools import repeat
from logging import warning
from types import MappingProxyType
from typing import Callable, Iterable, Dict, Mapping, Set

from owlapy.model import OWLObjectOneOf, OWLOntology, OWLNamedIndividual, OWLClass, OWLClassExpression, \
    OWLObjectProperty, OWLDataProperty, OWLObjectUnionOf, OWLObjectIntersectionOf, OWLObjectSomeValuesFrom, \
    OWLObjectPropertyExpression, OWLObjectComplementOf, OWLObjectAllValuesFrom, IRI, OWLObjectInverseOf, \
    OWLDataSomeValuesFrom, OWLDataPropertyExpression, OWLDatatypeRestriction, OWLLiteral, \
    OWLDataComplementOf, OWLDataAllValuesFrom, OWLDatatype, OWLDataHasValue, OWLDataOneOf, OWLReasoner, \
    OWLDataIntersectionOf, OWLDataUnionOf, OWLObjectCardinalityRestriction, OWLObjectMinCardinality, \
    OWLObjectMaxCardinality, OWLObjectExactCardinality, OWLObjectHasValue
from owlapy.util import NamedFixedSet, LRUCache

logger = logging.getLogger(__name__)


class OWLReasoner_FastInstanceChecker(OWLReasoner):
    """Tries to check instances fast (but maybe incomplete)"""
    __slots__ = '_ontology', '_base_reasoner', \
                '_ind_enc', '_cls_to_ind', '_obj_prop', '_obj_prop_inv', '_data_prop', '_objectsomevalues_cache', \
                '_negation_default', '_datasomevalues_cache', '_objectcardinality_cache'

    _ontology: OWLOntology
    _base_reasoner: OWLReasoner
    _cls_to_ind: Dict[OWLClass, int]  # Class => individuals
    _obj_prop: Dict[OWLObjectProperty, Mapping[int, int]]  # ObjectProperty => { individual => individuals }
    _obj_prop_inv: Dict[OWLObjectProperty, Mapping[int, int]]  # ObjectProperty => { individual => individuals }
    _data_prop: Dict[OWLDataProperty, Mapping[int, Set[OWLLiteral]]]  # DataProperty => { individual => literals }
    _ind_enc: NamedFixedSet[OWLNamedIndividual]
    _objectsomevalues_cache: LRUCache[OWLClassExpression, int]  # ObjectSomeValuesFrom => individuals
    _datasomevalues_cache: Dict[OWLClassExpression, int]  # DataSomeValuesFrom => individuals
    _objectcardinality_cache: Dict[OWLClassExpression, int]  # ObjectCardinalityRestriction => individuals

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

    def _init(self, osv_cache_size=128):
        self._cls_to_ind = dict()
        self._obj_prop = dict()
        self._obj_prop_inv = dict()
        self._data_prop = dict()
        individuals = self._ontology.individuals_in_signature()
        self._ind_enc = NamedFixedSet(OWLNamedIndividual, individuals)
        self._objectsomevalues_cache = LRUCache(maxsize=osv_cache_size)
        self._datasomevalues_cache = dict()
        self._objectcardinality_cache = dict()

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

    def object_property_values(self, ind: OWLNamedIndividual, pe: OWLObjectPropertyExpression) \
            -> Iterable[OWLNamedIndividual]:
        self._lazy_cache_obj_prop(pe)
        ind_enc = self._ind_enc(ind)
        if isinstance(pe, OWLObjectProperty):
            yield from self._ind_enc(self._obj_prop[pe][ind_enc])
        elif isinstance(pe, OWLObjectInverseOf):
            yield from self._ind_enc(self._obj_prop_inv[pe.get_named_property()][ind_enc])
        else:
            raise NotImplementedError

    def flush(self) -> None:
        self._base_reasoner.flush()

    def instances(self, ce: OWLClassExpression, direct: bool = False) -> Iterable[OWLNamedIndividual]:
        if direct:
            warning("direct not implemented")
        temp = self._find_instances(ce)
        yield from self._ind_enc(temp)

    def sub_classes(self, ce: OWLClassExpression, direct: bool = False) -> Iterable[OWLClass]:
        yield from self._base_reasoner.sub_classes(ce, direct=direct)

    def super_classes(self, ce: OWLClassExpression, direct: bool = False) -> Iterable[OWLClass]:
        yield from self._base_reasoner.super_classes(ce, direct=direct)

    def types(self, ind: OWLNamedIndividual, direct: bool = False) -> Iterable[OWLClass]:
        yield from self._base_reasoner.types(ind, direct=direct)

    def sub_data_properties(self, dp: OWLDataProperty, direct: bool = False) -> Iterable[OWLDataProperty]:
        yield from self._base_reasoner.sub_data_properties(dp=dp, direct=direct)

    def sub_object_properties(self, op: OWLObjectPropertyExpression, direct: bool = False)\
            -> Iterable[OWLObjectPropertyExpression]:
        yield from self._base_reasoner.sub_object_properties(op=op, direct=direct)

    def get_root_ontology(self) -> OWLOntology:
        return self._ontology

    def _lazy_cache_obj_prop(self, pe: OWLObjectPropertyExpression) -> None:
        """Get all individuals involved in this object property and put them in a Dict"""
        if isinstance(pe, OWLObjectInverseOf):
            inverse = True
            if pe.get_named_property() in self._obj_prop_inv:
                return
        elif isinstance(pe, OWLObjectProperty):
            inverse = False
            if pe in self._obj_prop:
                return
        else:
            raise NotImplementedError

        # Dict with Individual => Set[Individual]
        opc: Dict[int, int] = dict()

        # shortcut for owlready2
        from owlapy.owlready2 import OWLOntology_Owlready2
        if isinstance(self._ontology, OWLOntology_Owlready2):
            import owlready2
            # _x => owlready2 objects
            p_x: owlready2.ObjectProperty = self._ontology._world[pe.get_named_property().get_iri().as_str()]
            for l_x, r_x in p_x.get_relations():
                if inverse:
                    o_x = l_x
                    s_x = r_x
                else:
                    s_x = l_x
                    o_x = r_x
                if isinstance(s_x, owlready2.Thing) and isinstance(o_x, owlready2.Thing):
                    s_enc = self._ind_enc(OWLNamedIndividual(IRI.create(s_x.iri)))
                    o_enc = self._ind_enc(OWLNamedIndividual(IRI.create(o_x.iri)))
                    if s_enc in opc:
                        opc[s_enc] |= o_enc
                    else:
                        opc[s_enc] = o_enc
        else:
            for s_enc, s in self._ind_enc.items():
                individuals = self._ind_enc(self._base_reasoner.object_property_values(s, pe))
                if individuals:
                    opc[s_enc] = individuals

        if inverse:
            self._obj_prop_inv[pe.get_named_property()] = MappingProxyType(opc)
        else:
            self._obj_prop[pe] = MappingProxyType(opc)

    def _lazy_cache_data_prop(self, pe: OWLDataPropertyExpression) -> None:
        """Get all individuals and values involved in this data property and put them in a Dict"""
        assert(isinstance(pe, OWLDataProperty))
        if pe in self._data_prop:
            return

        opc: Dict[int, Set[OWLLiteral]] = dict()

        # shortcut for owlready2
        from owlapy.owlready2 import OWLOntology_Owlready2
        if isinstance(self._ontology, OWLOntology_Owlready2):
            import owlready2
            # _x => owlready2 objects
            p_x: owlready2.DataProperty = self._ontology._world[pe.get_iri().as_str()]
            for s_x, o_x in p_x.get_relations():
                if isinstance(s_x, owlready2.Thing):
                    o_literal = OWLLiteral(o_x)
                    s_enc = self._ind_enc(OWLNamedIndividual(IRI.create(s_x.iri)))
                    if s_enc in opc:
                        opc[s_enc].add(o_literal)
                    else:
                        opc[s_enc] = {o_literal}
        else:
            for s_enc, s in self._ind_enc.items():
                values = set(self._base_reasoner.data_property_values(s, pe))
                if len(values) > 0:
                    opc[s_enc] = values

        self._data_prop[pe] = MappingProxyType(opc)

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
        assert isinstance(p, OWLObjectPropertyExpression)
        self._lazy_cache_obj_prop(p)

        filler_ind_enc = self._find_instances(ce.get_filler())
        ind_enc = 0
        if isinstance(p, OWLObjectInverseOf):
            ops = self._obj_prop_inv[p.get_named_property()]
        elif isinstance(p, OWLObjectProperty):
            ops = self._obj_prop[p]
        else:
            raise ValueError

        for s_enc, o_set_enc in ops.items():
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
            logger.warning("Object Complement Of not implemented at %s", ce)
            return 0
            # if self.complement_as_negation:
            #     ...
            # else:
            #     self._lazy_cache_negation

    @_find_instances.register
    def _(self, ce: OWLObjectAllValuesFrom):
        return self._find_instances(
            OWLObjectSomeValuesFrom(
                property=ce.get_property(),
                filler=ce.get_filler().get_object_complement_of().get_nnf()
            ).get_object_complement_of())

    @_find_instances.register
    def _(self, ce: OWLObjectOneOf):
        return self._ind_enc(ce.individuals())

    @_find_instances.register
    def _(self, ce: OWLObjectHasValue):
        return self._find_instances(ce.as_some_values_from())

    @_find_instances.register
    def _(self, ce: OWLObjectMinCardinality):
        return self._get_instances_object_card_restriction(ce, operator.ge)

    @_find_instances.register
    def _(self, ce: OWLObjectMaxCardinality):
        all_ = (1 << len(self._ind_enc)) - 1
        min_ind_enc = self._find_instances(OWLObjectMinCardinality(cardinality=ce.get_cardinality()+1,
                                                                   property=ce.get_property(),
                                                                   filler=ce.get_filler()))
        return all_ ^ min_ind_enc

    @_find_instances.register
    def _(self, ce: OWLObjectExactCardinality):
        return self._get_instances_object_card_restriction(ce, operator.eq)

    def _get_instances_object_card_restriction(self, ce: OWLObjectCardinalityRestriction,
                                               operator_: Callable):
        if ce in self._objectcardinality_cache:
            return self._objectcardinality_cache[ce]

        p = ce.get_property()
        card = ce.get_cardinality()
        assert isinstance(p, OWLObjectPropertyExpression)
        assert card >= 0
        self._lazy_cache_obj_prop(p)

        filler_ind_enc = self._find_instances(ce.get_filler())
        ind_enc = 0
        if isinstance(p, OWLObjectInverseOf):
            ops = self._obj_prop_inv[p.get_named_property()]
        elif isinstance(p, OWLObjectProperty):
            ops = self._obj_prop[p]
        else:
            raise ValueError

        for s_enc, o_set_enc in ops.items():
            if operator_(bin(o_set_enc & filler_ind_enc).count('1'), card):
                ind_enc |= s_enc

        self._objectcardinality_cache[ce] = ind_enc
        return ind_enc

    @_find_instances.register
    def _(self, ce: OWLDataSomeValuesFrom):
        if ce in self._datasomevalues_cache:
            return self._datasomevalues_cache[ce]

        p = ce.get_property()
        filler = ce.get_filler()
        assert isinstance(p, OWLDataPropertyExpression)
        self._lazy_cache_data_prop(p)
        ops = self._data_prop[p]

        ind_enc = 0
        if isinstance(filler, OWLDatatype):
            # TODO: Currently we just assume that the values are of the given type (also
            # done in DLLearner)
            ind_enc = reduce(operator.or_, ops.keys())
        elif isinstance(filler, OWLDataOneOf):
            values = set(filler.values())
            for s_enc, literals in ops.items():
                if literals & values:
                    ind_enc |= s_enc
        elif isinstance(filler, OWLDataComplementOf):
            ind_enc = reduce(operator.or_, ops.keys())
            temp_enc = self._find_instances(
                OWLDataSomeValuesFrom(property=p, filler=filler.get_data_range()))
            ind_enc ^= temp_enc
        elif isinstance(filler, OWLDataUnionOf):
            operands = [OWLDataSomeValuesFrom(p, op) for op in filler.operands()]
            ind_enc = reduce(operator.or_, map(self._find_instances, operands))
        elif isinstance(filler, OWLDataIntersectionOf):
            operands = [OWLDataSomeValuesFrom(p, op) for op in filler.operands()]
            ind_enc = reduce(operator.and_, map(self._find_instances, operands))
        elif isinstance(filler, OWLDatatypeRestriction):
            # TODO: Parse the facet restrictions once before?
            facet_restrictions = filler.get_facet_restrictions()
            for s_enc, literals in ops.items():
                for o_literal in literals:
                    if o_literal.get_datatype() == filler.get_datatype() and \
                            all(map(lambda res, l: res.get_facet().operator(l, res.get_facet_value()),
                                    facet_restrictions,
                                    repeat(o_literal))):
                        ind_enc |= s_enc
                        break
        else:
            raise ValueError

        self._datasomevalues_cache[ce] = ind_enc
        return ind_enc

    @_find_instances.register
    def _(self, ce: OWLDataAllValuesFrom):
        filler = ce.get_filler()
        if isinstance(filler, OWLDataComplementOf):
            filler = filler.get_data_range()
        else:
            filler = OWLDataComplementOf(filler)
        return self._find_instances(
            OWLDataSomeValuesFrom(
                property=ce.get_property(),
                filler=filler
            ).get_object_complement_of())

    @_find_instances.register
    def _(self, ce: OWLDataHasValue):
        return self._find_instances(ce.as_some_values_from())

    def _lazy_cache_class(self, c: OWLClass) -> None:
        if c in self._cls_to_ind:
            return
        temp = self._base_reasoner.instances(c)
        self._cls_to_ind[c] = self._ind_enc(temp)
