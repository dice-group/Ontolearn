from collections import Counter
from typing import Iterable, List, Optional, AbstractSet, Dict

from ontolearn.core.owl.hierarchy import ClassHierarchy, ObjectPropertyHierarchy, DatatypePropertyHierarchy
from ontolearn.core.owl.utils import OrderedOWLObject
from owlapy.model import OWLClass, OWLClassExpression, OWLObjectComplementOf, OWLObjectSomeValuesFrom, \
    OWLObjectAllValuesFrom, OWLObjectIntersectionOf, OWLObjectUnionOf, OWLObjectPropertyExpression, OWLThing, \
    OWLNothing, OWLReasoner, OWLObjectProperty
from owlapy.utils import as_index
from ontolearn.utils import parametrized_performance_debugger


class ConceptGenerator:
    """A class that can generate some sorts of OWL Class Expressions"""
    __slots__ = '_class_hierarchy', '_object_property_hierarchy', '_data_property_hierarchy', '_reasoner', '_op_domains'

    _class_hierarchy: ClassHierarchy
    _object_property_hierarchy: ObjectPropertyHierarchy
    _data_property_hierarchy: DatatypePropertyHierarchy
    _reasoner: OWLReasoner
    _op_domains: Dict[OWLObjectProperty, AbstractSet[OWLClass]]

    def __init__(self, reasoner: OWLReasoner,
                 class_hierarchy: Optional[ClassHierarchy] = None,
                 object_property_hierarchy: Optional[ObjectPropertyHierarchy] = None,
                 data_property_hierarchy: Optional[DatatypePropertyHierarchy] = None):
        self._reasoner = reasoner

        if class_hierarchy is None:
            class_hierarchy = ClassHierarchy(self._reasoner)

        if object_property_hierarchy is None:
            object_property_hierarchy = ObjectPropertyHierarchy(self._reasoner)

        if data_property_hierarchy is None:
            data_property_hierarchy = DatatypePropertyHierarchy(self._reasoner)

        self._class_hierarchy = class_hierarchy
        self._object_property_hierarchy = object_property_hierarchy
        self._data_property_hierarchy = data_property_hierarchy

        self._op_domains = dict()

    def get_leaf_concepts(self, concept: OWLClass):
        """ Return : { x | (x subClassOf concept) AND not exist y: y subClassOf x )} """
        assert isinstance(concept, OWLClass)
        yield from self._class_hierarchy.leaves(of=concept)

    @parametrized_performance_debugger()
    def negation_from_iterables(self, s: Iterable[OWLClassExpression]) -> Iterable[OWLObjectComplementOf]:
        """ Return : { x | ( x \\equv not s} """
        for item in s:
            assert isinstance(item, OWLClassExpression)
            yield self.negation(item)

    @parametrized_performance_debugger()
    def get_direct_sub_concepts(self, concept: OWLClass) -> Iterable[OWLClass]:
        """ Return : { x | ( x subClassOf concept )} """
        assert isinstance(concept, OWLClass)
        yield from self._class_hierarchy.sub_classes(concept, direct=True)

    def _object_property_domain(self, prop: OWLObjectProperty):
        if prop not in self._op_domains:
            self._op_domains[prop] = frozenset(self._reasoner.object_property_domains(prop))
        return self._op_domains[prop]

    def most_general_existential_restrictions(self, *,
                                              domain: OWLClassExpression, filler: Optional[OWLClassExpression] = None) \
            -> Iterable[OWLObjectSomeValuesFrom]:
        if filler is None:
            filler = self.thing
        assert isinstance(domain, OWLClass)  # for now, only named classes supported
        assert isinstance(filler, OWLClassExpression)

        for prop in self._object_property_hierarchy.most_general_roles():
            if domain.is_owl_thing() or domain in self._object_property_domain(prop):
                yield OWLObjectSomeValuesFrom(property=prop, filler=filler)

    def most_general_universal_restrictions(self, *,
                                            domain: OWLClassExpression, filler: Optional[OWLClassExpression] = None) \
            -> Iterable[OWLObjectAllValuesFrom]:
        if filler is None:
            filler = self.thing
        assert isinstance(domain, OWLClass)  # for now, only named classes supported
        assert isinstance(filler, OWLClassExpression)

        for prop in self._object_property_hierarchy.most_general_roles():
            if domain.is_owl_thing() or domain in self._object_property_domain(prop):
                yield OWLObjectAllValuesFrom(property=prop, filler=filler)

    # noinspection PyMethodMayBeStatic
    def intersection(self, ops: Iterable[OWLClassExpression]) -> OWLObjectIntersectionOf:
        operands = []
        for c in ops:
            if isinstance(c, OWLObjectIntersectionOf):
                operands.extend(c.operands())
            else:
                assert isinstance(c, OWLClassExpression)
                operands.append(c)
        # operands = _avoid_overly_redundand_operands(operands)
        return OWLObjectIntersectionOf(operands)

    # noinspection PyMethodMayBeStatic
    def union(self, ops: Iterable[OWLClassExpression]) -> OWLObjectUnionOf:
        operands = []
        for c in ops:
            if isinstance(c, OWLObjectUnionOf):
                operands.extend(c.operands())
            else:
                assert isinstance(c, OWLClassExpression)
                operands.append(c)
        # operands = _avoid_overly_redundand_operands(operands)
        return OWLObjectUnionOf(operands)

    def get_direct_parents(self, concept: OWLClassExpression) -> Iterable[OWLClass]:
        assert isinstance(concept, OWLClass)
        yield from self._class_hierarchy.super_classes(concept, direct=True)

    def get_all_direct_sub_concepts(self, concept: OWLClassExpression) -> Iterable[OWLClassExpression]:
        assert isinstance(concept, OWLClass)
        yield from self._class_hierarchy.sub_classes(concept, direct=True)

    def get_all_sub_concepts(self, concept: OWLClassExpression) -> Iterable[OWLClassExpression]:
        assert isinstance(concept, OWLClass)
        yield from self._class_hierarchy.sub_classes(concept, direct=False)

    # noinspection PyMethodMayBeStatic
    def existential_restriction(self, concept: OWLClassExpression, property: OWLObjectPropertyExpression) \
            -> OWLObjectSomeValuesFrom:
        assert isinstance(property, OWLObjectPropertyExpression)
        return OWLObjectSomeValuesFrom(property=property, filler=concept)

    # noinspection PyMethodMayBeStatic
    def universal_restriction(self, concept: OWLClassExpression, property: OWLObjectPropertyExpression) \
            -> OWLObjectAllValuesFrom:
        assert isinstance(property, OWLObjectPropertyExpression)
        return OWLObjectAllValuesFrom(property=property, filler=concept)

    def negation(self, concept: OWLClassExpression) -> OWLClassExpression:
        if concept.is_owl_thing():
            return self.nothing
        elif isinstance(concept, OWLObjectComplementOf):
            return concept.get_operand()
        else:
            return concept.get_object_complement_of()

    def contains_class(self, concept: OWLClassExpression) -> bool:
        assert isinstance(concept, OWLClass)
        return concept in self._class_hierarchy

    def class_hierarchy(self) -> ClassHierarchy:
        return self._class_hierarchy

    @property
    def thing(self) -> OWLClass:
        return OWLThing

    @property
    def nothing(self) -> OWLClass:
        return OWLNothing

    def clean(self):
        self._op_domains.clear()
