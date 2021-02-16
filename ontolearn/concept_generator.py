from typing import Iterable

from ontolearn.core.owl.hierarchy import ClassHierarchy, ObjectPropertyHierarchy, DatatypePropertyHierarchy
from ontolearn.owlapy.model import OWLClass, OWLClassExpression, OWLObjectComplementOf, OWLObjectSomeValuesFrom, \
    OWLObjectAllValuesFrom, OWLObjectIntersectionOf, OWLObjectUnionOf, OWLObjectPropertyExpression, OWLThing, OWLNothing
from ontolearn.utils import parametrized_performance_debugger


class ConceptGenerator:
    """A class that can generate some sorts of OWL Class Expressions"""
    __slots__ = '_class_hierarchy', '_object_property_hierarchy', '_data_property_hierarchy'

    _class_hierarchy: ClassHierarchy
    _object_property_hierarchy: ObjectPropertyHierarchy
    _data_property_hierarchy: DatatypePropertyHierarchy

    def __init__(self, class_hierarchy: ClassHierarchy, object_property_hierarchy: ObjectPropertyHierarchy,
                 data_property_hierarchy: DatatypePropertyHierarchy):
        self._class_hierarchy = class_hierarchy
        self._object_property_hierarchy = object_property_hierarchy
        self._data_property_hierarchy = data_property_hierarchy

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

    def most_general_existential_restrictions(self, concept: OWLClassExpression) -> Iterable[OWLObjectSomeValuesFrom]:
        assert isinstance(concept, OWLClassExpression)
        for prop in self._object_property_hierarchy.most_general_roles():
            yield OWLObjectSomeValuesFrom(property=prop, filler=concept)

    def most_general_universal_restriction(self, concept: OWLClassExpression) -> Iterable[OWLObjectAllValuesFrom]:
        assert isinstance(concept, OWLClassExpression)
        for prop in self._object_property_hierarchy.most_general_roles():
            yield OWLObjectAllValuesFrom(property=prop, filler=concept)

    # noinspection PyMethodMayBeStatic
    def intersection(self, ops: Iterable[OWLClassExpression]) -> OWLObjectIntersectionOf:
        operands = []
        for c in ops:
            if isinstance(c, OWLObjectIntersectionOf):
                operands.extend(c.operands())
            else:
                assert isinstance(c, OWLClassExpression)
                operands.append(c)
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
        return OWLObjectUnionOf(operands)

    def get_direct_parents(self, concept: OWLClassExpression) -> Iterable[OWLClass]:
        assert isinstance(concept, OWLClass)
        yield from self._class_hierarchy.super_classes(concept, direct=True)

    def get_all_direct_sub_concepts(self, concept: OWLClassExpression) -> Iterable[OWLClassExpression]:
        assert isinstance(concept, OWLClass)
        yield from self._class_hierarchy.sub_classes(concept, direct=True)

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
        pass
