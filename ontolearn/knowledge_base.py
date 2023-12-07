""" Knowledge Base."""

import logging
import random
from itertools import chain
from typing import Iterable, Optional, Callable, overload, Union, FrozenSet, Set, Dict
from ontolearn.base import OWLOntology_Owlready2, OWLOntologyManager_Owlready2, OWLReasoner_Owlready2
from ontolearn.base.fast_instance_checker import OWLReasoner_FastInstanceChecker
from owlapy.model import OWLOntologyManager, OWLOntology, OWLReasoner, OWLClassExpression, \
    OWLNamedIndividual, OWLObjectProperty, OWLClass, OWLDataProperty, IRI, OWLDataRange, OWLObjectSomeValuesFrom, \
    OWLObjectAllValuesFrom, OWLDatatype, BooleanOWLDatatype, NUMERIC_DATATYPES, TIME_DATATYPES, OWLThing, \
    OWLObjectPropertyExpression, OWLLiteral, OWLDataPropertyExpression
from owlapy.render import DLSyntaxObjectRenderer
from ontolearn.search import EvaluatedConcept
from owlapy.util import iter_count, LRUCache
from .abstracts import AbstractKnowledgeBase, AbstractScorer, EncodedLearningProblem
from .concept_generator import ConceptGenerator
from ontolearn.base.owl.utils import OWLClassExpressionLengthMetric
from .learning_problem import PosNegLPStandard, EncodedPosNegLPStandard
from ontolearn.base.owl.hierarchy import ClassHierarchy, ObjectPropertyHierarchy, DatatypePropertyHierarchy

Factory = Callable

logger = logging.getLogger(__name__)


# TODO:CD: To many non pythonic functions
def _Default_OntologyManagerFactory(world_store=None) -> OWLOntologyManager:
    return OWLOntologyManager_Owlready2(world_store=world_store)


def _Default_ReasonerFactory(onto: OWLOntology, triplestore_address: str) -> OWLReasoner:

    assert isinstance(onto, OWLOntology_Owlready2)
    base_reasoner = OWLReasoner_Owlready2(ontology=onto, triplestore_address=triplestore_address)

    if triplestore_address is not None:
        return base_reasoner
    else:
        return OWLReasoner_FastInstanceChecker(ontology=onto, base_reasoner=base_reasoner)


def _Default_ClassExpressionLengthMetricFactory() -> OWLClassExpressionLengthMetric:
    return OWLClassExpressionLengthMetric.get_default()

# TODO:CD: __init__ is overcrowded. This bit can/should be simplified to few lines
# TODO:CD: Namings are not self-explanatory: User does not need to know
#  a) factory programming pattern b) Manager Classes etc inadvertently increases cognitive load


class KnowledgeBase(AbstractKnowledgeBase):
    """Representation of an OWL knowledge base in Ontolearn.

    Args:
        path: Path to an ontology file that is to be loaded.
        ontologymanager_factory: Factory that creates an ontology manager to be used to load the file.
        ontology: OWL ontology object.
        reasoner_factory: Factory that creates a reasoner to reason about the ontology.
        reasoner: reasoner Over the ontology.
        length_metric_factory: See :attr:`length_metric`.
        length_metric: Length metric that is used in calculation of class expression lengths.
        individuals_cache_size: How many individuals of class expressions to cache.
        backend_store: Whether to sync the world to backend store.
            reasoner of this object, if you enter a reasoner using :arg:`reasoner_factory` or :arg:`reasoner`
            argument it will override this setting.
        triplestore_address: The address where the triplestore is hosted.
        include_implicit_individuals: Whether to identify and consider instances which are not set as OWL Named
            Individuals (does not contain this type) as individuals.

    Attributes:
        generator (ConceptGenerator): Instance of concept generator.
        path (str): Path of the ontology file.
        use_individuals_cache (bool): Whether to use individuals cache to store individuals for method efficiency.
    """
    __slots__ = '_manager', '_ontology', '_reasoner', '_length_metric', \
                '_ind_set', '_ind_cache', 'path', 'use_individuals_cache', 'generator', '_class_hierarchy', \
                '_object_property_hierarchy', '_data_property_hierarchy', '_op_domains', '_op_ranges', '_dp_domains', \
                '_dp_ranges'

    _manager: OWLOntologyManager
    _ontology: OWLOntology
    _reasoner: OWLReasoner

    _length_metric: OWLClassExpressionLengthMetric

    _ind_set: FrozenSet[OWLNamedIndividual]
    _ind_cache: LRUCache[OWLClassExpression, FrozenSet[OWLNamedIndividual]]  # class expression => individuals

    path: str
    use_individuals_cache: bool
    generator: ConceptGenerator

    _class_hierarchy: ClassHierarchy
    _object_property_hierarchy: ObjectPropertyHierarchy
    _data_property_hierarchy: DatatypePropertyHierarchy
    _op_domains: Dict[OWLObjectProperty, OWLClassExpression]
    _op_ranges: Dict[OWLObjectProperty, OWLClassExpression]
    _dp_domains: Dict[OWLDataProperty, OWLClassExpression]
    _dp_ranges: Dict[OWLDataProperty, FrozenSet[OWLDataRange]]

    @overload
    def __init__(self, *,
                 path: str,
                 ontologymanager_factory: Factory[[], OWLOntologyManager] = _Default_OntologyManagerFactory,
                 reasoner_factory: Factory[[OWLOntology], OWLReasoner] = _Default_ReasonerFactory,
                 length_metric: Optional[OWLClassExpressionLengthMetric] = None,
                 length_metric_factory: Optional[Factory[[], OWLClassExpressionLengthMetric]] = None,
                 individuals_cache_size=128,
                 triplestore_address: str = None,
                 backend_store: bool = False,
                 include_implicit_individuals=False):
        ...

    @overload
    def __init__(self, *,
                 ontology: OWLOntology,
                 reasoner: OWLReasoner,
                 length_metric: Optional[OWLClassExpressionLengthMetric] = None,
                 length_metric_factory: Optional[Factory[[], OWLClassExpressionLengthMetric]] = None,
                 individuals_cache_size=128):
        ...

    @overload
    def __init__(self, *, triplestore_address: str = None):
        ...

    def __init__(self, *,
                 path: Optional[str] = None,

                 ontologymanager_factory: Optional[Factory[[], OWLOntologyManager]] = None,
                 reasoner_factory: Optional[Factory[[OWLOntology], OWLReasoner]] = None,
                 length_metric_factory: Optional[Factory[[], OWLClassExpressionLengthMetric]] = None,

                 ontology: Optional[OWLOntology] = None,
                 reasoner: Optional[OWLReasoner] = None,
                 triplestore_address: str = None,
                 length_metric: Optional[OWLClassExpressionLengthMetric] = None,

                 individuals_cache_size=128,
                 backend_store: bool = False,
                 class_hierarchy: Optional[ClassHierarchy] = None,
                 object_property_hierarchy: Optional[ObjectPropertyHierarchy] = None,
                 data_property_hierarchy: Optional[DatatypePropertyHierarchy] = None,
                 include_implicit_individuals=False
                 ):
        AbstractKnowledgeBase.__init__(self)
        self.path = path

        if triplestore_address is not None:
            self._manager = _Default_OntologyManagerFactory()
            if path is None:
                # create a dummy ontology, so we can avoid making tons of changes.
                self._ontology = OWLOntology_Owlready2(self._manager, IRI.create("dummy_ontology#onto"), load=False,
                                                       triplestore_address=triplestore_address)
            else:
                # why not create a real ontology if the user gives the path :) (triplestore will be used anyway)
                self._ontology = OWLOntology_Owlready2(self._manager, IRI.create('file://' + self.path), load=True,
                                                       triplestore_address=triplestore_address)
        else:
            if ontology is not None:
                self._manager = ontology.get_owl_ontology_manager()
                self._ontology = ontology
            elif ontologymanager_factory is not None:
                self._manager = ontologymanager_factory()
            else:  # default to Owlready2 implementation
                if path is not None and backend_store:
                    self._manager = _Default_OntologyManagerFactory(world_store=path + ".or2")
                else:
                    self._manager = _Default_OntologyManagerFactory()
                # raise TypeError("neither ontology nor manager factory given")

            if ontology is None:
                if path is None:
                    raise TypeError("path missing")
                else:
                    self._ontology = self._manager.load_ontology(IRI.create('file://' + self.path))
                    if isinstance(self._manager, OWLOntologyManager_Owlready2) and backend_store:
                        self._manager.save_world()
                        logger.debug("Synced world to backend store")

        is_using_triplestore = True if triplestore_address is not None else False
        if reasoner is not None and reasoner.is_using_triplestore() == is_using_triplestore:
            self._reasoner = reasoner
        elif reasoner_factory is not None and triplestore_address is None:
            self._reasoner = reasoner_factory(self._ontology)
        else:
            self._reasoner = _Default_ReasonerFactory(self._ontology, triplestore_address)

        if length_metric is not None:
            self._length_metric = length_metric
        elif length_metric_factory is not None:
            self._length_metric = length_metric_factory()
        else:
            self._length_metric = _Default_ClassExpressionLengthMetricFactory()

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
        self._dp_domains = dict()
        self._dp_ranges = dict()
        self.generator = ConceptGenerator()

        self.use_individuals_cache = individuals_cache_size > 0
        if self.use_individuals_cache:
            self._ind_cache = LRUCache(maxsize=individuals_cache_size)

        if include_implicit_individuals:
            self._ind_set = frozenset(chain.from_iterable(self.individuals(i) for i in self.get_concepts()))
        elif isinstance(self._reasoner, OWLReasoner_FastInstanceChecker) and triplestore_address is None:
            self._ind_set = self._reasoner._ind_set  # performance hack:
        else:
            individuals = self._ontology.individuals_in_signature()
            self._ind_set = frozenset(individuals)

        self.describe()

    def ontology(self) -> OWLOntology:
        """Get the root Ontology loaded in this knowledge base.

        Returns:
            The Ontology.
        """

        return self._ontology

    def reasoner(self) -> OWLReasoner:
        """Get the Reasoner loaded in this knowledge base.

        Returns:
            The Reasoner.
        """

        return self._reasoner

    def ignore_and_copy(self, ignored_classes: Optional[Iterable[OWLClass]] = None,
                        ignored_object_properties: Optional[Iterable[OWLObjectProperty]] = None,
                        ignored_data_properties: Optional[Iterable[OWLDataProperty]] = None) -> 'KnowledgeBase':
        """Makes a copy of the knowledge base while ignoring specified concepts and properties.

        Args:
            ignored_classes: Classes to ignore.
            ignored_object_properties: Object properties to ignore.
            ignored_data_properties: Data properties to ignore.
        Returns:
            A new KnowledgeBase with the hierarchies restricted as requested.
        """

        new = object.__new__(KnowledgeBase)

        AbstractKnowledgeBase.__init__(new)
        new._manager = self._manager
        new._ontology = self._ontology
        new._reasoner = self._reasoner
        new._length_metric = self._length_metric
        new._ind_set = self._ind_set
        new.path = self.path
        new.use_individuals_cache = self.use_individuals_cache
        new.generator = self.generator
        new._op_domains = self._op_domains
        new._op_ranges = self._op_ranges
        new._dp_domains = self._dp_domains
        new._dp_ranges = self._dp_ranges

        if self.use_individuals_cache:
            new._ind_cache = LRUCache(maxsize=self._ind_cache.maxsize)

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
            new._class_hierarchy = self._class_hierarchy.restrict_and_copy(remove=owl_concepts_to_ignore)
        else:
            new._class_hierarchy = self._class_hierarchy

        if ignored_object_properties is not None:
            new._object_property_hierarchy = self._object_property_hierarchy.restrict_and_copy(
                remove=ignored_object_properties)
        else:
            new._object_property_hierarchy = self._object_property_hierarchy

        if ignored_data_properties is not None:
            new._data_property_hierarchy = self._data_property_hierarchy.restrict_and_copy(
                remove=ignored_data_properties)
        else:
            new._data_property_hierarchy = self._data_property_hierarchy

        return new

    def concept_len(self, ce: OWLClassExpression) -> int:
        """Calculates the length of a concept and is used by some concept learning algorithms to
        find the best results considering also the length of the concepts.

        Args:
            ce: The concept to be measured.
        Returns:
            Length of the concept.
        """

        return self._length_metric.length(ce)

    def clean(self):
        """Clean all stored values (states and caches) if there is any.

        Note:
            1. If you have more than one learning problem that you want to fit to the same model (i.e. to learn the
            concept using the same concept learner model) use this method to make sure that you have cleared every
            previous stored value.
            2. If you store another KnowledgeBase instance using the same variable name as before, it is recommended to
            use this method before the initialization to avoid data mismatch.
        """

        self._op_domains.clear()
        if self.use_individuals_cache:
            self._ind_cache.cache_clear()

    def _cache_individuals(self, ce: OWLClassExpression) -> None:
        if not self.use_individuals_cache:
            raise TypeError
        if ce in self._ind_cache:
            return
        if isinstance(self._reasoner, OWLReasoner_FastInstanceChecker):
            self._ind_cache[ce] = self._reasoner._find_instances(ce)  # performance hack
        else:
            temp = self._reasoner.instances(ce)
            self._ind_cache[ce] = frozenset(temp)

    def _maybe_cache_individuals(self, ce: OWLClassExpression) -> Iterable[OWLNamedIndividual]:
        if self.use_individuals_cache:
            self._cache_individuals(ce)
            yield from self._ind_cache[ce]
        else:
            yield from self._reasoner.instances(ce)

    def _maybe_cache_individuals_count(self, ce: OWLClassExpression) -> int:
        if self.use_individuals_cache:
            self._cache_individuals(ce)
            r = self._ind_cache[ce]
            return len(r)
        else:
            return iter_count(self._reasoner.instances(ce))

    def individuals(self, concept: Optional[OWLClassExpression] = None) -> Iterable[OWLNamedIndividual]:
        """Retrieve all individuals belonging to the concept in the ontology. If the concept property is not
        specified then it returns all the individuals.

        Args:
            concept: Class expression of which to list individuals.
        Returns:
            Individuals belonging to the given class.
        """

        if concept is None or concept.is_owl_thing():
            for i in self._ind_set:
                yield i
        else:
            yield from self._maybe_cache_individuals(concept)

    def individuals_count(self, concept: Optional[OWLClassExpression] = None) -> int:
        """Returns the number of all individuals belonging to the concept in the ontology.

        Args:
            concept: Class expression of the individuals to count.
        Returns:
            Number of the individuals belonging to the given class.
        """

        if concept is None or concept.is_owl_thing():
            return len(self._ind_set)
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
        """Retrieve the individuals specified in the arg as a frozenset. If `arg` is an OWLClassExpression then this
        method behaves as the method "individuals" but will return the final result as a frozenset.

        Args:
            arg: more than one individual/ single individual/ class expression of which to list individuals.
        Returns:
            Frozenset of the individuals depending on the arg type.
        """

        if isinstance(arg, OWLClassExpression):
            if self.use_individuals_cache:
                self._cache_individuals(arg)
                r = self._ind_cache[arg]
                return r
            else:
                return frozenset(self.individuals(arg))
        elif isinstance(arg, OWLNamedIndividual):
            return frozenset({arg})
        else:
            return frozenset(arg)

    def all_individuals_set(self):
        """Retrieve all the individuals of the knowledge base.

        Returns:
            Frozenset of the all individuals.
        """

        if self._ind_set is not None:
            return self._ind_set
        else:
            return frozenset(self._ontology.individuals_in_signature())

    def most_general_object_properties(self, *, domain: OWLClassExpression, inverse: bool = False) \
            -> Iterable[OWLObjectProperty]:
        """Find the most general object property.

        Args:
            domain: Domain for which to search properties.
            inverse: Inverse order?
        """
        assert isinstance(domain, OWLClassExpression)

        func = self.get_object_property_ranges if inverse else self.get_object_property_domains

        inds_domain = self.individuals_set(domain)
        for prop in self._object_property_hierarchy.most_general_roles():
            if domain.is_owl_thing() or inds_domain <= self.individuals_set(func(prop)):
                yield prop

    def _data_properties_for_domain(self, domain: OWLClassExpression, data_properties: Iterable[OWLDataProperty]) \
            -> Iterable[OWLDataProperty]:
        assert isinstance(domain, OWLClassExpression)

        inds_domain = self.individuals_set(domain)
        for prop in data_properties:
            if domain.is_owl_thing() or inds_domain <= self.individuals_set(self.get_data_property_domains(prop)):
                yield prop

    def __repr__(self):
        properties_count = iter_count(self.ontology().object_properties_in_signature()) + iter_count(
            self.ontology().data_properties_in_signature())
        class_count = iter_count(self.ontology().classes_in_signature())
        individuals_count = self.individuals_count()

        return f'KnowledgeBase(path={repr(self.path)} <{class_count} classes, {properties_count} properties, ' \
               f'{individuals_count} individuals)'

    # in case more types of AbstractLearningProblem are introduced to the project uncomment the method below and use
    # decorators
    # @singledispatchmethod
    # def encode_learning_problem(self, lp: AbstractLearningProblem):
    #     raise NotImplementedError(lp)

    def encode_learning_problem(self, lp: PosNegLPStandard):
        """Provides the encoded learning problem (lp), i.e. the class containing the set of OWLNamedIndividuals
        as follows:
            kb_pos --> the positive examples set,
            kb_neg --> the negative examples set,
            kb_all --> all lp individuals / all individuals set,
            kb_diff --> kb_all - (kb_pos + kb_neg).
        Note:
            Simple access of the learning problem individuals divided in respective sets.
            You will need the encoded learning problem to use the method evaluate_concept of this class.
        Args:
            lp (PosNegLPStandard): The learning problem.
        Return:
            EncodedPosNegLPStandard: The encoded learning problem.
        """

        assert len(self.class_hierarchy()) > 0

        if lp.all is None:
            kb_all = self.all_individuals_set()
        else:
            kb_all = self.individuals_set(lp.all)

        assert 0 < len(lp.pos) < len(kb_all) and len(kb_all) > len(lp.neg)
        if logger.isEnabledFor(logging.INFO):
            r = DLSyntaxObjectRenderer()
            logger.info('E^+:[ {0} ]'.format(', '.join(map(r.render, lp.pos))))
            logger.info('E^-:[ {0} ]'.format(', '.join(map(r.render, lp.neg))))

        kb_pos = self.individuals_set(lp.pos)
        if len(lp.neg) == 0:  # if negatives are not provided, randomly sample.
            kb_neg = type(kb_all)(random.sample(list(kb_all), len(kb_pos)))
        else:
            kb_neg = self.individuals_set(lp.neg)

        try:
            assert len(kb_pos) == len(lp.pos)
        except AssertionError:
            print(lp.pos)
            print(kb_pos)
            print(kb_all)
            print('Assertion error. Exiting.')
            raise
        if lp.neg:
            assert len(kb_neg) == len(lp.neg)

        return EncodedPosNegLPStandard(
            kb_pos=kb_pos,
            kb_neg=kb_neg,
            kb_all=kb_all,
            kb_diff=kb_all.difference(kb_pos.union(kb_neg)))

    def evaluate_concept(self, concept: OWLClassExpression, quality_func: AbstractScorer,
                         encoded_learning_problem: EncodedLearningProblem) -> EvaluatedConcept:
        """Evaluates a concept by using the encoded learning problem examples, in terms of Accuracy or F1-score.

        Note:
            This method is useful to tell the quality (e.q) of a generated concept by the concept learners, to get
            the set of individuals (e.inds) that are classified by this concept and the amount of them (e.ic).
        Args:
            concept: The concept to be evaluated.
            quality_func: Quality measurement in terms of Accuracy or F1-score.
            encoded_learning_problem: The encoded learning problem.
        Return:
            The evaluated concept.
        """

        e = EvaluatedConcept()
        e.inds = self.individuals_set(concept)
        e.ic = len(e.inds)
        _, e.q = quality_func.score_elp(e.inds, encoded_learning_problem)
        return e

    async def evaluate_concept_async(self, concept: OWLClassExpression, quality_func: AbstractScorer,
                                     encoded_learning_problem: EncodedLearningProblem) -> EvaluatedConcept:
        """The asynchronous version of evaluate_concept.

        Raises:
            NotImplementedError: This method is not implemented yet.
        """
        raise NotImplementedError

    def get_leaf_concepts(self, concept: OWLClass):
        """Get leaf classes.

        Args:
            concept: Atomic class for which to find leaf classes.

        Returns:
            Leaf classes { x \\| (x subClassOf concept) AND not exist y: y subClassOf x )}. """
        assert isinstance(concept, OWLClass)
        yield from self._class_hierarchy.leaves(of=concept)

    def get_direct_sub_concepts(self, concept: OWLClass) -> Iterable[OWLClass]:
        """Direct sub-classes of atomic class.

        Args:
            concept: Atomic concept.

        Returns:
            Direct sub classes of concept { x \\| ( x subClassOf concept )}."""
        assert isinstance(concept, OWLClass)
        yield from self._class_hierarchy.sub_classes(concept, direct=True)

    def get_object_property_domains(self, prop: OWLObjectProperty) -> OWLClassExpression:
        """Get the domains of an object property.

        Args:
            prop: Object property.

        Returns:
            Domains of the property.
        """
        if prop not in self._op_domains:
            domains = list(self._reasoner.object_property_domains(prop, direct=True))
            self._op_domains[prop] = self.generator.intersection(domains) if len(domains) > 1 else domains[0]
        return self._op_domains[prop]

    def get_object_property_ranges(self, prop: OWLObjectProperty) -> OWLClassExpression:
        """Get the ranges of an object property.

        Args:
            prop: Object property.

        Returns:
            Ranges of the property.
        """
        if prop not in self._op_ranges:
            ranges = list(self._reasoner.object_property_ranges(prop, direct=True))
            self._op_ranges[prop] = self.generator.intersection(ranges) if len(ranges) > 1 else ranges[0]
        return self._op_ranges[prop]

    def get_data_property_domains(self, prop: OWLDataProperty) -> OWLClassExpression:
        """Get the domains of a data property.

        Args:
            prop: Data property.

        Returns:
            Domains of the property.
        """
        if prop not in self._dp_domains:
            domains = list(self._reasoner.data_property_domains(prop, direct=True))
            self._dp_domains[prop] = self.generator.intersection(domains) if len(domains) > 1 else domains[0]
        return self._dp_domains[prop]

    def get_data_property_ranges(self, prop: OWLDataProperty) -> FrozenSet[OWLDataRange]:
        """Get the ranges of a data property.

        Args:
            prop: Data property.

        Returns:
            Ranges of the property.
        """
        if prop not in self._dp_ranges:
            self._dp_ranges[prop] = frozenset(self._reasoner.data_property_ranges(prop, direct=True))
        return self._dp_ranges[prop]

    def most_general_data_properties(self, *, domain: OWLClassExpression) -> Iterable[OWLDataProperty]:
        """Find most general data properties that are applicable to a domain.

        Args:
            domain: Domain for which to search properties.

        Returns:
            Most general data properties for the given domain.
        """
        yield from self._data_properties_for_domain(domain, self.get_data_properties())

    def most_general_boolean_data_properties(self, *, domain: OWLClassExpression) -> Iterable[OWLDataProperty]:
        """Find most general boolean data properties that are applicable to a domain.

        Args:
            domain: Domain for which to search properties.

        Returns:
            Most general boolean data properties for the given domain.
        """
        yield from self._data_properties_for_domain(domain, self.get_boolean_data_properties())

    def most_general_numeric_data_properties(self, *, domain: OWLClassExpression) -> Iterable[OWLDataProperty]:
        """Find most general numeric data properties that are applicable to a domain.

        Args:
            domain: Domain for which to search properties.

        Returns:
            Most general numeric data properties for the given domain.
        """
        yield from self._data_properties_for_domain(domain, self.get_numeric_data_properties())

    def most_general_time_data_properties(self, *, domain: OWLClassExpression) -> Iterable[OWLDataProperty]:
        """Find most general time data properties that are applicable to a domain.

        Args:
            domain: Domain for which to search properties.

        Returns:
            Most general time data properties for the given domain.
        """
        yield from self._data_properties_for_domain(domain, self.get_time_data_properties())

    def most_general_existential_restrictions(self, *,
                                              domain: OWLClassExpression, filler: Optional[OWLClassExpression] = None) \
            -> Iterable[OWLObjectSomeValuesFrom]:
        """Find most general existential restrictions that are applicable to a domain.

        Args:
            domain: Domain for which to search properties.
            filler: Optional filler to put in the restriction (not normally used).

        Returns:
           Most general existential restrictions for the given domain.
        """
        if filler is None:
            filler = self.generator.thing
        assert isinstance(filler, OWLClassExpression)

        for prop in self.most_general_object_properties(domain=domain):
            yield OWLObjectSomeValuesFrom(property=prop, filler=filler)

    def most_general_universal_restrictions(self, *,
                                            domain: OWLClassExpression, filler: Optional[OWLClassExpression] = None) \
            -> Iterable[OWLObjectAllValuesFrom]:
        """Find most general universal restrictions that are applicable to a domain.

        Args:
            domain: Domain for which to search properties.
            filler: Optional filler to put in the restriction (not normally used).

        Returns:
            Most general universal restrictions for the given domain.
        """
        if filler is None:
            filler = self.generator.thing
        assert isinstance(filler, OWLClassExpression)

        for prop in self.most_general_object_properties(domain=domain):
            yield OWLObjectAllValuesFrom(property=prop, filler=filler)

    def most_general_existential_restrictions_inverse(self, *,
                                                      domain: OWLClassExpression,
                                                      filler: Optional[OWLClassExpression] = None) \
            -> Iterable[OWLObjectSomeValuesFrom]:
        """Find most general inverse existential restrictions that are applicable to a domain.

        Args:
            domain: Domain for which to search properties.
            filler: Optional filler to put in the restriction (not normally used).

        Returns:
            Most general existential restrictions over inverse property.
        """
        if filler is None:
            filler = self.generator.thing
        assert isinstance(filler, OWLClassExpression)

        for prop in self.most_general_object_properties(domain=domain, inverse=True):
            yield OWLObjectSomeValuesFrom(property=prop.get_inverse_property(), filler=filler)

    def most_general_universal_restrictions_inverse(self, *,
                                                    domain: OWLClassExpression,
                                                    filler: Optional[OWLClassExpression] = None) \
            -> Iterable[OWLObjectAllValuesFrom]:
        """Find most general inverse universal restrictions that are applicable to a domain.

        Args:
            domain: Domain for which to search properties.
            filler: Optional filler to put in the restriction (not normally used).

        Returns:
            Most general universal restrictions over inverse property.
        """
        if filler is None:
            filler = self.generator.thing
        assert isinstance(filler, OWLClassExpression)

        for prop in self.most_general_object_properties(domain=domain, inverse=True):
            yield OWLObjectAllValuesFrom(property=prop.get_inverse_property(), filler=filler)

    def get_direct_parents(self, concept: OWLClassExpression) -> Iterable[OWLClass]:
        """Direct parent concepts.

        Args:
            concept: Concept to find super concepts of.

        Returns:
            Direct parent concepts.
        """
        assert isinstance(concept, OWLClass)
        yield from self._class_hierarchy.super_classes(concept, direct=True)

    def get_all_direct_sub_concepts(self, concept: OWLClassExpression) -> Iterable[OWLClassExpression]:
        """All direct sub concepts of a concept.

        Args:
            concept: Parent concept for which to get sub concepts.

        Returns:
            Direct sub concepts.
        """
        assert isinstance(concept, OWLClass)
        yield from self._class_hierarchy.sub_classes(concept, direct=True)

    def get_all_sub_concepts(self, concept: OWLClassExpression) -> Iterable[OWLClassExpression]:
        """All sub concepts of a concept.

        Args:
            concept: Parent concept for which to get sub concepts.

        Returns:
            Sub concepts.
        """
        assert isinstance(concept, OWLClass)
        yield from self._class_hierarchy.sub_classes(concept, direct=False)

    def get_concepts(self) -> Iterable[OWLClass]:
        """Get all concepts of this concept generator.

        Returns:
            Concepts.
        """
        yield from self._class_hierarchy.items()

    def get_object_properties(self) -> Iterable[OWLObjectProperty]:
        """Get all object properties of this concept generator.

        Returns:
            Object properties.
        """

        yield from self._object_property_hierarchy.items()

    def get_data_properties(self, ranges: Set[OWLDatatype] = None) -> Iterable[OWLDataProperty]:
        """Get all data properties of this concept generator for the given ranges.

        Args:
           ranges: Ranges for which to extract the data properties.

        Returns:
            Data properties for the given range.
        """
        if ranges is not None:
            for dp in self._data_property_hierarchy.items():
                if self.get_data_property_ranges(dp) & ranges:
                    yield dp
        else:
            yield from self._data_property_hierarchy.items()

    def get_boolean_data_properties(self) -> Iterable[OWLDataProperty]:
        """Get all boolean data properties of this concept generator.

        Returns:
            Boolean data properties.
        """
        yield from self.get_data_properties({BooleanOWLDatatype})

    def get_numeric_data_properties(self) -> Iterable[OWLDataProperty]:
        """Get all numeric data properties of this concept generator.

        Returns:
            Numeric data properties.
        """
        yield from self.get_data_properties(NUMERIC_DATATYPES)

    def get_time_data_properties(self) -> Iterable[OWLDataProperty]:
        """Get all time data properties of this concept generator.

        Returns:
            Time data properties.
        """
        yield from self.get_data_properties(TIME_DATATYPES)

    def get_types(self, ind: OWLNamedIndividual, direct: bool = False) -> Iterable[OWLClass]:
        """Get the named classes which are (direct) types of the specified individual.

        Args:
            ind: Individual.
            direct: Whether to consider direct types.

        Returns:
            Types of the given individual.
        """
        all_types = set(self.get_concepts())
        for type_ in self._reasoner.types(ind, direct):
            if type_ in all_types or type_ == OWLThing:
                yield type_

    def get_object_properties_for_ind(self, ind: OWLNamedIndividual, direct: bool = True) \
            -> Iterable[OWLObjectProperty]:
        """Get the object properties for the given individual.

        Args:
            ind: Individual
            direct: Whether only direct properties should be considered (True), or if also
                    indirect properties should be considered (False). Indirect properties
                    would be super properties super_p of properties p with ObjectPropertyAssertion(p ind obj).

        Returns:
            Object properties.
        """
        properties = set(self.get_object_properties())
        yield from (pe for pe in self._reasoner.ind_object_properties(ind, direct) if pe in properties)

    def get_data_properties_for_ind(self, ind: OWLNamedIndividual, direct: bool = True) -> Iterable[OWLDataProperty]:
        """Get the data properties for the given individual

        Args:
            ind: Individual
            direct: Whether only direct properties should be considered (True), or if also
                    indirect properties should be considered (False). Indirect properties
                    would be super properties super_p of properties p with ObjectPropertyAssertion(p ind obj).

        Returns:
            Data properties.
        """
        properties = set(self.get_data_properties())
        yield from (pe for pe in self._reasoner.ind_data_properties(ind, direct) if pe in properties)

    def get_object_property_values(self, ind: OWLNamedIndividual,
                                   property_: OWLObjectPropertyExpression,
                                   direct: bool = True) -> Iterable[OWLNamedIndividual]:
        """Get the object property values for the given individual and property.

        Args:
            ind: Individual.
            property_: Object property.
            direct: Whether only the property property_ should be considered (True), or if also
                    the values of sub properties of property_ should be considered (False).

        Returns:
            Individuals.
        """
        yield from self._reasoner.object_property_values(ind, property_, direct)

    def get_data_property_values(self, ind: OWLNamedIndividual,
                                 property_: OWLDataPropertyExpression,
                                 direct: bool = True) -> Iterable[OWLLiteral]:
        """Get the data property values for the given individual and property.

        Args:
            ind: Individual.
            property_: Data property.
            direct: Whether only the property property_ should be considered (True), or if also
                    the values of sub properties of property_ should be considered (False).

        Returns:
            Literals.
        """
        yield from self._reasoner.data_property_values(ind, property_, direct)

    def contains_class(self, concept: OWLClassExpression) -> bool:
        """Check if an atomic class is contained within this concept generator.

        Args:
            concept: Atomic class.

        Returns:
            Whether the class is contained in the concept generator.
        """
        assert isinstance(concept, OWLClass)
        return concept in self._class_hierarchy

    def class_hierarchy(self) -> ClassHierarchy:
        """Access the Class Hierarchy of this Concept Generator.

        Returns:
            Class hierarchy.
        """
        return self._class_hierarchy

    def object_property_hierarchy(self) -> ObjectPropertyHierarchy:
        """Access the Object property hierarchy of this concept generator.

        Returns:
            Object property hierarchy.
        """
        return self._object_property_hierarchy

    def data_property_hierarchy(self) -> DatatypePropertyHierarchy:
        """Access the Datatype property hierarchy of this concept generator.

        Returns:
            Data property hierarchy.
        """
        return self._data_property_hierarchy
