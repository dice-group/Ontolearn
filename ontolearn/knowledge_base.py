import logging
from typing import Dict, Iterable, Optional, Callable, overload, Union

from .abstracts import AbstractKnowledgeBase
from .concept_generator import ConceptGenerator
from .core.owl.hierarchy import ClassHierarchy, ObjectPropertyHierarchy, DatatypePropertyHierarchy
from .core.owl.utils import OWLClassExpressionLengthMetric
from .core.utils import BitSet
from .owlapy import IRI
from .owlapy.model import OWLOntologyManager, OWLOntology, OWLReasoner, OWLClassExpression, OWLNamedIndividual
from .owlapy.utils import NamedFixedSet, popcount, iter_count

# warnings.filterwarnings("ignore")

Factory = Callable

logger = logging.getLogger(__name__)


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


class KnowledgeBase(AbstractKnowledgeBase, ConceptGenerator):
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
                '_ind_enc', '_ind_cache', 'path', 'use_individuals_cache'

    _manager: OWLOntologyManager
    _ontology: OWLOntology
    _reasoner: OWLReasoner

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
        AbstractKnowledgeBase.__init__(self)
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

        ConceptGenerator.__init__(self,
                                  class_hierarchy=ClassHierarchy(self._reasoner),
                                  object_property_hierarchy=ObjectPropertyHierarchy(self._reasoner),
                                  data_property_hierarchy=DatatypePropertyHierarchy(self._reasoner))

        individuals = self._ontology.individuals_in_signature()
        self._ind_enc = NamedFixedSet(OWLNamedIndividual, individuals)

        self.use_individuals_cache = use_individuals_cache
        self.clean()

        self.describe()

    def ontology(self) -> OWLOntology:
        return self._ontology

    def restrict_and_copy(self, ):

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
        ConceptGenerator.clean(self)
        if self.use_individuals_cache:
            self._ind_cache.clear()

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

    @overload
    def individuals_set(self, concept: OWLClassExpression):
        ...

    @overload
    def individuals_set(self, individual: OWLNamedIndividual):
        ...

    @overload
    def individuals_set(self, individuals: Iterable[OWLNamedIndividual]):
        ...

    def individuals_set(self, arg: Union[Iterable[OWLNamedIndividual], OWLNamedIndividual, OWLClassExpression]):
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
