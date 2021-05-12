import logging
from typing import Dict, Iterable, Optional, Callable, overload, Union

from .abstracts import AbstractKnowledgeBase
from .concept_generator import ConceptGenerator
from .core.owl.utils import OWLClassExpressionLengthMetric
from .core.utils import BitSet
from owlapy import IRI
from owlapy.model import OWLOntologyManager, OWLOntology, OWLReasoner, OWLClassExpression, OWLNamedIndividual, \
    OWLObjectProperty, OWLClass, OWLDataProperty
from owlapy.render import DLSyntaxObjectRenderer
from owlapy.util import NamedFixedSet, popcount, iter_count

Factory = Callable

logger = logging.getLogger(__name__)


def _Default_OntologyManagerFactory() -> OWLOntologyManager:
    from owlapy.owlready2 import OWLOntologyManager_Owlready2

    return OWLOntologyManager_Owlready2()


def _Default_ReasonerFactory(onto: OWLOntology) -> OWLReasoner:
    from owlapy.owlready2 import OWLOntology_Owlready2
    from owlapy.owlready2.temp_classes import OWLReasoner_Owlready2_TempClasses
    from owlapy.fast_instance_checker import OWLReasoner_FastInstanceChecker

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

    _length_metric: OWLClassExpressionLengthMetric

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

        ConceptGenerator.__init__(self, reasoner=self._reasoner)

        individuals = self._ontology.individuals_in_signature()
        from owlapy.fast_instance_checker import OWLReasoner_FastInstanceChecker
        if isinstance(self._reasoner, OWLReasoner_FastInstanceChecker):
            self._ind_enc = self._reasoner._ind_enc  # performance hack
        else:
            self._ind_enc = NamedFixedSet(OWLNamedIndividual, individuals)

        self.use_individuals_cache = use_individuals_cache
        if use_individuals_cache:
            self._ind_cache = dict()

        self.describe()

    def ontology(self) -> OWLOntology:
        """Root Ontology loaded in this knowledge base

        Returns:
            Ontology
        """
        return self._ontology

    def reasoner(self) -> OWLReasoner:
        """Reasoner loaded in this knowledge base

        Returns:
            reasoner
        """
        return self._reasoner

    def ignore_and_copy(self, ignored_classes: Optional[Iterable[OWLClass]] = None,
                        ignored_object_properties: Optional[Iterable[OWLObjectProperty]] = None,
                        ignored_data_properties: Optional[Iterable[OWLDataProperty]] = None) -> 'KnowledgeBase':
        """Make a copy of the knowledge base while ignoring specified concepts and properties

        Args:
            ignored_classes: classes to ignore
            ignored_object_properties: object properties to ignore
            ignored_data_properties: data properties to ignore

        Returns:
            a new KnowledgeBase with the hierarchies restricted as requested
        """
        new = object.__new__(KnowledgeBase)

        AbstractKnowledgeBase.__init__(new)
        new._manager = self._manager
        new._ontology = self._ontology
        new._reasoner = self._reasoner
        new._length_metric = self._length_metric
        new._ind_enc = self._ind_enc
        new.path = self.path
        new.use_individuals_cache = self.use_individuals_cache

        if self.use_individuals_cache:
            new._ind_cache = self._ind_cache.copy()

        if ignored_classes is not None:
            owl_concepts_to_ignore = set()
            for i in ignored_classes:
                if self.contains_class(i):
                    owl_concepts_to_ignore.add(i)
                else:
                    raise ValueError(
                        f'{i} could not found in \n{self} \n'
                        f'{[_ for _ in self.ontology().classes_in_signature()]}.')
            if logger.isEnabledFor(logging.INFO):
                r = DLSyntaxObjectRenderer()
                logger.info('Concepts to ignore: {0}'.format(' '.join(map(r.render, owl_concepts_to_ignore))))
            class_hierarchy = self._class_hierarchy.restrict_and_copy(remove=owl_concepts_to_ignore)
        else:
            class_hierarchy = self._class_hierarchy

        if ignored_object_properties is not None:
            object_property_hierarchy = self._object_property_hierarchy.restrict_and_copy(
                remove=ignored_object_properties)
        else:
            object_property_hierarchy = self._object_property_hierarchy

        if ignored_data_properties is not None:
            data_property_hierarchy = self._data_property_hierarchy.restrict_and_copy(remove=ignored_data_properties)
        else:
            data_property_hierarchy = self._data_property_hierarchy

        ConceptGenerator.__init__(new,
                                  reasoner=self._reasoner,
                                  class_hierarchy=class_hierarchy,
                                  object_property_hierarchy=object_property_hierarchy,
                                  data_property_hierarchy=data_property_hierarchy)

        return new

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
        from owlapy.fast_instance_checker import OWLReasoner_FastInstanceChecker
        if isinstance(self._reasoner, OWLReasoner_FastInstanceChecker):
            self._ind_cache[ce] = self._reasoner._find_instances(ce)  # performance hack
        else:
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
            if self.use_individuals_cache:
                self._cache_individuals(arg)
                return BitSet(self._ind_cache[arg])
            else:
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
