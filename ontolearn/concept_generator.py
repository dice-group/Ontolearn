from typing import Iterable, Optional, AbstractSet, Dict, Generator

from ontolearn.core.owl.hierarchy import ClassHierarchy, ObjectPropertyHierarchy, DatatypePropertyHierarchy
from ontolearn.utils import parametrized_performance_debugger
from owlapy.model import OWLClass, OWLClassExpression, OWLObjectComplementOf, OWLObjectSomeValuesFrom, \
    OWLObjectAllValuesFrom, OWLObjectIntersectionOf, OWLObjectUnionOf, OWLObjectPropertyExpression, OWLThing, \
    OWLNothing, OWLReasoner, OWLObjectProperty


class ConceptGenerator:
    """A class that can generate some sorts of OWL Class Expressions"""
    __slots__ = '_class_hierarchy', '_object_property_hierarchy', '_data_property_hierarchy', '_reasoner', \
                '_op_domains', '_op_ranges'

    _class_hierarchy: ClassHierarchy
    _object_property_hierarchy: ObjectPropertyHierarchy
    _data_property_hierarchy: DatatypePropertyHierarchy
    _reasoner: OWLReasoner
    _op_domains: Dict[OWLObjectProperty, AbstractSet[OWLClass]]
    _op_ranges: Dict[OWLObjectProperty, AbstractSet[OWLClass]]

    def __init__(self, reasoner: OWLReasoner,
                 class_hierarchy: Optional[ClassHierarchy] = None,
                 object_property_hierarchy: Optional[ObjectPropertyHierarchy] = None,
                 data_property_hierarchy: Optional[DatatypePropertyHierarchy] = None):
        """Create a new Concept Generator

        Args:
            reasoner: OWL reasoner with ontology loaded
            class_hierarchy: Class hierarchy to answer subclass queries. Created from the root ontology loaded in the
                reasoner if not given
            object_property_hierarchy: Object property hierarchy. Created from the root ontology loaded in the reasoner
                if not given
            data_property_hierarchy: Data property hierarchy. Created from the root ontology loaded in the reasoner
                if not given
        """
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
        self._op_ranges = dict()

    def get_leaf_concepts(self, concept: OWLClass):
        """Get leaf classes

        Args:
            concept: atomic class for which to find leaf classes

        Returns:
            Leaf classes

                { x \\| (x subClassOf concept) AND not exist y: y subClassOf x )} """
        assert isinstance(concept, OWLClass)
        yield from self._class_hierarchy.leaves(of=concept)

    @parametrized_performance_debugger()
    def negation_from_iterables(self, class_expressions: Iterable[OWLClassExpression]):
        """Negate a sequence of Class Expressions

        Args:
            class_expressions: iterable of class expressions to negate

        Returns:
            negated form of input

                { x \\| ( x \\equv not s} """
        for item in class_expressions:
            assert isinstance(item, OWLClassExpression)
            yield self.negation(item)

    @staticmethod
    def intersect_from_iterables(a_operands: Iterable[OWLClassExpression], b_operands: Iterable[OWLClassExpression]) -> \
            Iterable[OWLObjectComplementOf]:
        """ Create an intersection of each class expression in a_operands with each class expression in b_operands"""
        assert isinstance(a_operands, Generator) is False and isinstance(b_operands, Generator) is False
        seen = set()
        # TODO: if input sizes say 10^4, we can employ multiprocessing
        for i in a_operands:
            for j in b_operands:
                if (i, j) in seen:
                    continue
                i_and_j = OWLObjectIntersectionOf((i, j))
                seen.add((i, j))
                seen.add((j, i))
                yield i_and_j

    @staticmethod
    def union_from_iterables(a_operands: Iterable[OWLClassExpression],
                             b_operands: Iterable[OWLClassExpression]) -> Iterable[OWLObjectUnionOf]:
        """ Create an union of each class expression in a_operands with each class expression in b_operands"""
        assert (isinstance(a_operands, Generator) is False) and (isinstance(b_operands, Generator) is False)
        # TODO: if input sizes say 10^4, we can employ multiprocessing
        seen = set()
        for i in a_operands:
            for j in b_operands:
                if (i, j) in seen:
                    continue
                i_and_j = OWLObjectUnionOf((i, j))
                seen.add((i, j))
                seen.add((j, i))
                yield i_and_j

    @parametrized_performance_debugger()
    def get_direct_sub_concepts(self, concept: OWLClass) -> Iterable[OWLClass]:
        """Direct sub classes of atomic class

        Args:
            concept: atomic concept

        Returns:
            direct sub classes of concept

                { x \\| ( x subClassOf concept )} """
        assert isinstance(concept, OWLClass)
        yield from self._class_hierarchy.sub_classes(concept, direct=True)

    def _object_property_domain(self, prop: OWLObjectProperty):
        """Get the domain of a property

        Args:
            prop: object property

        Returns:
            domain of the property
        """
        if prop not in self._op_domains:
            self._op_domains[prop] = frozenset(self._reasoner.object_property_domains(prop))
        return self._op_domains[prop]

    def _object_property_range(self, prop: OWLObjectProperty):
        """Get the range of a property

        Args:
            prop: object property

        Returns:
            range of the property
        """
        if prop not in self._op_ranges:
            self._op_ranges[prop] = frozenset(self._reasoner.object_property_ranges(prop))
        return self._op_ranges[prop]

    def most_general_existential_restrictions(self, *,
                                              domain: OWLClassExpression, filler: Optional[OWLClassExpression] = None) \
            -> Iterable[OWLObjectSomeValuesFrom]:
        """Find most general restrictions that are applicable to a domain

        Args:
            domain: domain for which to search properties
            filler: optional filler to put in the restriction (not normally used)

        Returns:
            existential restrictions
        """
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
        """Find most general restrictions that are applicable to a domain

        Args:
            domain: domain for which to search properties
            filler: optional filler to put in the restriction (not normally used)

        Returns:
            universal restrictions
        """
        if filler is None:
            filler = self.thing
        assert isinstance(domain, OWLClass)  # for now, only named classes supported
        assert isinstance(filler, OWLClassExpression)

        for prop in self._object_property_hierarchy.most_general_roles():
            if domain.is_owl_thing() or domain in self._object_property_domain(prop):
                yield OWLObjectAllValuesFrom(property=prop, filler=filler)

    def most_general_existential_restrictions_inverse(self, *,
                                                      domain: OWLClassExpression,
                                                      filler: Optional[OWLClassExpression] = None) \
            -> Iterable[OWLObjectSomeValuesFrom]:
        """Find most general inverse existential restrictions that are applicable to a domain

        Args:
            domain: domain for which to search properties
            filler: optional filler to put in the restriction (not normally used)

        Returns:
            existential restrictions over inverse property
        """
        if filler is None:
            filler = self.thing
        assert isinstance(domain, OWLClass)  # for now, only named classes supported
        assert isinstance(filler, OWLClassExpression)

        for prop in self._object_property_hierarchy.most_general_roles():
            if domain.is_owl_thing() or domain in self._object_property_range(prop):
                yield OWLObjectSomeValuesFrom(property=prop.get_inverse_property(), filler=filler)

    def most_general_universal_restrictions_inverse(self, *,
                                                    domain: OWLClassExpression,
                                                    filler: Optional[OWLClassExpression] = None) \
            -> Iterable[OWLObjectAllValuesFrom]:
        """Find most general universal inverse restrictions that are applicable to a domain

        Args:
            domain: domain for which to search properties
            filler: optional filler to put in the restriction (not normally used)

        Returns:
            universal restrictions over inverse property
        """
        if filler is None:
            filler = self.thing
        assert isinstance(domain, OWLClass)  # for now, only named classes supported
        assert isinstance(filler, OWLClassExpression)

        for prop in self._object_property_hierarchy.most_general_roles():
            if domain.is_owl_thing() or domain in self._object_property_range(prop):
                yield OWLObjectAllValuesFrom(property=prop.get_inverse_property(), filler=filler)

    # noinspection PyMethodMayBeStatic
    def intersection(self, ops: Iterable[OWLClassExpression]) -> OWLObjectIntersectionOf:
        """Create intersection of class expression

        Args:
            ops: operands of the intersection

        Returns:
            intersection with all operands (intersections are merged)
        """
        # TODO CD: I would rather prefer def intersection(self, a: OWLClassExpression, b: OWLClassExpression)
        # TODO: This is more advantages as one does not need to create a tuple of a list before intersection two expressions.
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
        """Create union of class expressions

        Args:
            ops: operands of the union

        Returns:
            union with all operands (unions are merged)
        """
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
        """Direct parent concepts

        Args:
            concept: concept to find super concepts of

        Returns:
            direct parent concepts
        """
        assert isinstance(concept, OWLClass)
        yield from self._class_hierarchy.super_classes(concept, direct=True)

    def get_all_direct_sub_concepts(self, concept: OWLClassExpression) -> Iterable[OWLClassExpression]:
        """All direct sub concepts of a concept

        Args:
            concept: parent concept for which to get sub concepts

        Returns:
            direct sub concepts
        """
        assert isinstance(concept, OWLClass)
        yield from self._class_hierarchy.sub_classes(concept, direct=True)

    def get_all_sub_concepts(self, concept: OWLClassExpression) -> Iterable[OWLClassExpression]:
        """All sub concepts of a concept

        Args:
            concept: parent concept for which to get sub concepts

        Returns:
            sub concepts
        """
        assert isinstance(concept, OWLClass)
        yield from self._class_hierarchy.sub_classes(concept, direct=False)

    # noinspection PyMethodMayBeStatic
    def existential_restriction(self, filler: OWLClassExpression, property: OWLObjectPropertyExpression) \
            -> OWLObjectSomeValuesFrom:
        """Create existential restriction

        Args:
            property: property
            filler: filler of the restriction

        Returns:
            existential restriction
        """
        assert isinstance(property, OWLObjectPropertyExpression)
        return OWLObjectSomeValuesFrom(property=property, filler=filler)

    # noinspection PyMethodMayBeStatic
    def universal_restriction(self, filler: OWLClassExpression, property: OWLObjectPropertyExpression) \
            -> OWLObjectAllValuesFrom:
        """Create universal restriction

        Args:
            property: property
            filler: filler of the restriction

        Returns:
            universal restriction
        """
        assert isinstance(property, OWLObjectPropertyExpression)
        return OWLObjectAllValuesFrom(property=property, filler=filler)

    def negation(self, concept: OWLClassExpression) -> OWLClassExpression:
        """Create negation of a concept

        Args:
            concept: class expression

        Returns:
            negation of concept
        """
        if concept.is_owl_thing():
            return self.nothing
        elif isinstance(concept, OWLObjectComplementOf):
            return concept.get_operand()
        else:
            return concept.get_object_complement_of()

    def contains_class(self, concept: OWLClassExpression) -> bool:
        """Check if an atomic class is contained within this concept generator

        Args:
            concept: atomic class

        Returns:
            whether the class is contained in the concept generator
        """
        assert isinstance(concept, OWLClass)
        return concept in self._class_hierarchy

    def class_hierarchy(self) -> ClassHierarchy:
        """Access the Class Hierarchy of this Concept Generator

        Returns:
            class hierarchy
        """
        return self._class_hierarchy

    def object_property_hierarchy(self) -> ObjectPropertyHierarchy:
        """Access the Object property hierarchy of this concept generator

        Returns:
            object property hierarchy
        """
        return self._object_property_hierarchy

    def data_property_hierarchy(self) -> DatatypePropertyHierarchy:
        """Access the Datatype property hierarchy of this concept generator

        Returns:
            data property hierarchy
        """
        return self._data_property_hierarchy

    @property
    def thing(self) -> OWLClass:
        """OWL Thing"""
        return OWLThing

    @property
    def nothing(self) -> OWLClass:
        """OWL Nothing"""
        return OWLNothing

    def clean(self):
        """Clear any state and caches"""
        self._op_domains.clear()
