from typing import Dict, Iterable, Optional, Callable, overload, cast, Union

from .abstracts import AbstractKnowledgeBase
from .core.owl.hierarchy import ClassHierarchy, ObjectPropertyHierarchy, DatatypePropertyHierarchy
from .core.owl.utils import OWLClassExpressionLengthMetric, BitSet
from .owlapy import IRI
from .owlapy.model import OWLOntologyManager, OWLOntology, OWLReasoner, OWLClassExpression, OWLObjectComplementOf, \
    OWLClass, OWLObjectSomeValuesFrom, OWLObjectAllValuesFrom, OWLNamedIndividual, OWLObjectIntersectionOf, \
    OWLObjectUnionOf, OWLObjectPropertyExpression, OWLThing, OWLNothing
from .owlapy.utils import NamedFixedSet, popcount, iter_count
from .utils import parametrized_performance_debugger

# warnings.filterwarnings("ignore")

Factory = Callable


def _Default_OntologyManagerFactory() -> OWLOntologyManager:
    from ontolearn.owlapy.owlready2 import OWLOntologyManager_Owlready2

    return OWLOntologyManager_Owlready2()


def _Default_ReasonerFactory(onto: OWLOntology) -> OWLReasoner:
    from ontolearn.owlapy.owlready2.base import OWLOntology_Owlready2
    from ontolearn.owlapy.owlready2.temp_classes import OWLReasoner_Owlready2_TempClasses
    from ontolearn.owlapy.fast_instance_checker import OWLReasoner_FastInstanceChecker

    assert isinstance(onto, OWLOntology_Owlready2)
    base_reasoner = OWLReasoner_Owlready2_TempClasses(ontology=onto)
    reasoner = OWLReasoner_FastInstanceChecker(ontology=onto,
                                               base_reasoner=base_reasoner)
    return reasoner


def _Default_ClassExpressionLengthMetricFactory() -> OWLClassExpressionLengthMetric:
    return OWLClassExpressionLengthMetric.get_default()


class KnowledgeBase(AbstractKnowledgeBase):
    """Knowledge Base Class representing Tbox and Abox along with concept hierarchies

    Args:
        path: path to an ontology file that is to be loaded
        ontologymanager_factory: factory that creates an ontology manager to be used to load the file
        ontology: OWL ontology object
        reasoner_factory: factory that creates a reasoner to reason about the ontology
        reasoner: reasoner over the ontology
        length_metric_factory: see `length_metric`
        length_metric: length metric that is used in calculation of class expresion lengths
    """
    __slots__ = '_manager', '_ontology', '_reasoner', '_length_metric', \
                '_class_hierarchy', '_object_property_hierarchy', '_data_property_hierarchy', '_ind_enc', \
                '_ind_cache', 'path', 'use_individuals_cache'

    _manager: OWLOntologyManager
    _ontology: OWLOntology
    _reasoner: OWLReasoner

    _class_hierarchy: ClassHierarchy
    _object_property_hierarchy: ObjectPropertyHierarchy
    _data_property_hierarchy: DatatypePropertyHierarchy

    _ind_enc: NamedFixedSet[OWLNamedIndividual]
    _ind_cache: Dict[OWLClassExpression, int]  # class expression => individuals

    path: str
    use_individuals_cache: bool

    @overload
    def __init__(self, *,
                 path: str,
                 ontologymanager_factory: Factory[[], OWLOntologyManager] = _Default_OntologyManagerFactory,
                 reasoner_factory: Factory[[OWLOntology], OWLReasoner] = _Default_ReasonerFactory,
                 length_metric: Optional[OWLClassExpressionLengthMetric] = None,
                 length_metric_factory: Optional[Factory[[], OWLClassExpressionLengthMetric]] = None,
                 use_individuals_cache: bool = True):
        ...

    @overload
    def __init__(self, *,
                 ontology: OWLOntology,
                 reasoner: OWLReasoner,
                 length_metric: Optional[OWLClassExpressionLengthMetric] = None,
                 length_metric_factory: Optional[Factory[[], OWLClassExpressionLengthMetric]] = None,
                 use_individuals_cache: bool = True):
        ...

    def __init__(self, *,
                 path: Optional[str] = None,

                 ontologymanager_factory: Optional[Factory[[], OWLOntologyManager]] = None,
                 reasoner_factory: Optional[Factory[[OWLOntology], OWLReasoner]] = None,
                 length_metric_factory: Optional[Factory[[], OWLClassExpressionLengthMetric]] = None,

                 ontology: Optional[OWLOntology] = None,
                 reasoner: Optional[OWLReasoner] = None,
                 length_metric: Optional[OWLClassExpressionLengthMetric] = None,

                 use_individuals_cache: bool = True):
        super().__init__()
        self.path = path
        if ontology is not None:
            self._manager = ontology.get_manager()
            self._ontology = ontology
        elif ontologymanager_factory is not None:
            self._manager = ontologymanager_factory()
        else:  # default to Owlready2 implementation
            self._manager = _Default_OntologyManagerFactory()
            # raise TypeError("neither ontology nor manager factory given")

        if ontology is None:
            if path is None:
                raise TypeError("path missing")
            self._ontology = self._manager.load_ontology(IRI.create('file://' + self.path))

        if reasoner is not None:
            self._reasoner = reasoner
        elif reasoner_factory is not None:
            self._reasoner = reasoner_factory(self._ontology)
        else:  # default to fast instance checker
            self._reasoner = _Default_ReasonerFactory(self._ontology)
            # raise TypeError("neither reasoner nor reasoner factory given")

        if length_metric is not None:
            self._length_metric = length_metric
        elif length_metric_factory is not None:
            self._length_metric = length_metric_factory()
        else:
            self._length_metric = _Default_ClassExpressionLengthMetricFactory()

        self._class_hierarchy = ClassHierarchy(self._reasoner)
        self._object_property_hierarchy = ObjectPropertyHierarchy(self._reasoner)
        self._data_property_hierarchy = DatatypePropertyHierarchy(self._reasoner)

        individuals = self._ontology.individuals_in_signature()
        self._ind_enc = NamedFixedSet(OWLNamedIndividual, individuals)

        self.use_individuals_cache = use_individuals_cache
        self.clean()

        self.describe()

    def ontology(self) -> OWLOntology:
        return self._ontology

    def cl(self, ce: OWLClassExpression) -> int:
        """Calculate the length of a concept

        Args:
            ce: concept

        Returns:
            length of the concept
        """
        return self._length_metric.length(ce)

    def clean(self):
        """Clean all stored values if there is any.
        """
        if self.use_individuals_cache:
            self._ind_cache = dict()

    def get_leaf_concepts(self, concept: OWLClass) -> Iterable[OWLClass]:
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
        """ Return : { \\exist.r.x | r \\in MostGeneral r} """
        assert isinstance(concept, OWLClassExpression)
        for prop in self._object_property_hierarchy.most_general_roles():
            yield OWLObjectSomeValuesFrom(property=prop, filler=concept)

    def most_general_universal_restrictions(self, concept: OWLClassExpression) -> Iterable[OWLObjectAllValuesFrom]:
        """ Return : { \\forall.r.x | r \\in MostGeneral r} """
        assert isinstance(concept, OWLClassExpression)
        for prop in self._object_property_hierarchy.most_general_roles():
            yield OWLObjectAllValuesFrom(property=prop, filler=concept)

    def _cache_individuals(self, ce: OWLClassExpression) -> None:
        if not self.use_individuals_cache:
            raise TypeError
        if ce in self._ind_cache:
            return
        temp = self._reasoner.instances(ce)
        self._ind_cache[ce] = self._ind_enc(temp)

    def _maybe_cache_individuals(self, ce: OWLClassExpression) -> Iterable[OWLNamedIndividual]:
        if self.use_individuals_cache:
            self._cache_individuals(ce)
            yield from self._ind_enc(self._ind_cache[ce])
        else:
            yield from self._reasoner.instances(ce)

    def _maybe_cache_individuals_count(self, ce: OWLClassExpression) -> int:
        if self.use_individuals_cache:
            self._cache_individuals(ce)
            return popcount(self._ind_cache[ce])
        else:
            return iter_count(self._reasoner.instances(ce))

    def individuals(self, concept: Optional[OWLClassExpression] = None) -> Iterable[OWLNamedIndividual]:
        """All named individuals belonging to the concept in the ontology

        Args:
            concept: class expression of which to list individuals

        Returns:
            individuals belonging to the given class
        """
        if concept is None or concept.is_owl_thing():
            for _, i in self._ind_enc.items():
                yield i
        else:
            yield from self._maybe_cache_individuals(concept)

    def individuals_count(self, concept: Optional[OWLClassExpression] = None) -> int:
        """Number of individuals"""
        if concept is None or concept.is_owl_thing():
            return len(self._ind_enc)
        else:
            return self._maybe_cache_individuals_count(concept)

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

    @overload
    def individuals_set(self, concept: OWLClassExpression):
        ...

    @overload
    def individuals_set(self, individuals: Iterable[OWLNamedIndividual]):
        ...

    def individuals_set(self, arg: Union[Iterable[OWLNamedIndividual], OWLClassExpression]):
        if isinstance(arg, OWLClassExpression):
            return self.individuals_set(self.individuals(arg))
        else:
            if self._ind_enc:
                return BitSet(self._ind_enc(arg))
            else:
                return frozenset(arg)

    def all_individuals_set(self):
        if self._ind_enc:
            return BitSet((1 << len(self._ind_enc)) - 1)
        else:
            return frozenset(self._ontology.individuals_in_signature())

    @property
    def thing(self) -> OWLClass:
        return OWLThing

    @property
    def nothing(self) -> OWLClass:
        return OWLNothing
