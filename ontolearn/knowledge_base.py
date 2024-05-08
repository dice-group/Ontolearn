""" Knowledge Base."""

import logging
import random
from collections import Counter
from typing import Iterable, Optional, Callable, overload, Union, FrozenSet, Set, Dict, cast, Generator

import owlapy
from owlapy.class_expression import OWLClassExpression, OWLClass, OWLObjectSomeValuesFrom, OWLObjectAllValuesFrom, \
    OWLThing, OWLObjectMinCardinality, OWLObjectOneOf
from owlapy.iri import IRI
from owlapy.owl_axiom import OWLClassAssertionAxiom, OWLObjectPropertyAssertionAxiom, OWLDataPropertyAssertionAxiom, \
    OWLSubClassOfAxiom, OWLEquivalentClassesAxiom
from owlapy.owl_data_ranges import OWLDataRange
from owlapy.owl_datatype import OWLDatatype
from owlapy.owl_individual import OWLNamedIndividual
from owlapy.owl_literal import BooleanOWLDatatype, NUMERIC_DATATYPES, DoubleOWLDatatype, TIME_DATATYPES, OWLLiteral
from owlapy.owl_ontology import OWLOntology
from owlapy.owl_ontology_manager import OWLOntologyManager
from owlapy.owl_property import OWLObjectProperty, OWLDataProperty, OWLObjectPropertyExpression, \
    OWLDataPropertyExpression
from owlapy.owl_reasoner import OWLReasoner

from ontolearn.base import OWLOntology_Owlready2, OWLOntologyManager_Owlready2, OWLReasoner_Owlready2
from ontolearn.base.fast_instance_checker import OWLReasoner_FastInstanceChecker

from owlapy.render import DLSyntaxObjectRenderer
from ontolearn.search import EvaluatedConcept
from owlapy.util import iter_count, LRUCache
from .abstracts import AbstractKnowledgeBase, AbstractScorer, EncodedLearningProblem
from .concept_generator import ConceptGenerator
from ontolearn.base.owl.utils import OWLClassExpressionLengthMetric
from .learning_problem import PosNegLPStandard, EncodedPosNegLPStandard
from ontolearn.base.owl.hierarchy import ClassHierarchy, ObjectPropertyHierarchy, DatatypePropertyHierarchy

from .utils.static_funcs import (init_length_metric, init_hierarchy_instances,
                                 init_named_individuals, init_individuals_from_concepts)

from owlapy.class_expression import OWLDataMaxCardinality, OWLDataSomeValuesFrom
from owlapy import owl_expression_to_sparql, owl_expression_to_dl
from owlapy.owl_data_ranges import OWLDataRange
from owlapy.class_expression import OWLDataOneOf

logger = logging.getLogger(__name__)


def depth_Default_ReasonerFactory(onto: OWLOntology) -> OWLReasoner:
    assert isinstance(onto, OWLOntology_Owlready2)
    base_reasoner = OWLReasoner_Owlready2(ontology=onto)
    return OWLReasoner_FastInstanceChecker(ontology=onto, base_reasoner=base_reasoner)


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
        include_implicit_individuals: Whether to identify and consider instances which are not set as OWL Named
            Individuals (does not contain this type) as individuals.

    Attributes:
        generator (ConceptGenerator): Instance of concept generator.
        path (str): Path of the ontology file.
        use_individuals_cache (bool): Whether to use individuals cache to store individuals for method efficiency.
    """
    # __slots__ = '_manager', '_ontology', '_reasoner', '_length_metric', \
    #    '_ind_set', '_ind_cache', 'path', 'use_individuals_cache', 'generator', '_class_hierarchy', \
    #    '_object_property_hierarchy', '_data_property_hierarchy', '_op_domains', '_op_ranges', '_dp_domains', \
    #    '_dp_ranges'

    length_metric: OWLClassExpressionLengthMetric

    ind_set: FrozenSet[OWLNamedIndividual]
    ind_cache: LRUCache[OWLClassExpression, FrozenSet[OWLNamedIndividual]]  # class expression => individuals

    path: str
    use_individuals_cache: bool
    generator: ConceptGenerator

    @overload
    def __init__(self, *,
                 path: str,
                 ontologymanager_factory: Callable[[], OWLOntologyManager] = OWLOntologyManager_Owlready2(
                     world_store=None),
                 reasoner_factory: Callable[[OWLOntology], OWLReasoner] = None,
                 length_metric: Optional[OWLClassExpressionLengthMetric] = None,
                 length_metric_factory: Optional[Callable[[], OWLClassExpressionLengthMetric]] = None,
                 individuals_cache_size=128,
                 backend_store: bool = False,
                 include_implicit_individuals=False):
        ...

    @overload
    def __init__(self, *,
                 ontology: OWLOntology,
                 reasoner: OWLReasoner,
                 length_metric: Optional[OWLClassExpressionLengthMetric] = None,
                 length_metric_factory: Optional[Callable[[], OWLClassExpressionLengthMetric]] = None,
                 individuals_cache_size=128):
        ...

    def __init__(self, *,
                 path: Optional[str] = None,

                 ontologymanager_factory: Optional[Callable[[], OWLOntologyManager]] = None,
                 reasoner_factory: Optional[Callable[[OWLOntology], OWLReasoner]] = None,
                 length_metric_factory: Optional[Callable[[], OWLClassExpressionLengthMetric]] = None,

                 ontology: Optional[OWLOntology] = None,
                 reasoner: Optional[OWLReasoner] = None,
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

        if ontology is not None:
            self.manager = ontology.get_owl_ontology_manager()
            self.ontology = ontology
        elif ontologymanager_factory is not None:
            self.manager = ontologymanager_factory()
        else:  # default to Owlready2 implementation
            if path is not None and backend_store:
                self.manager = OWLOntologyManager_Owlready2(world_store=path + ".or2")
            else:
                self.manager = OWLOntologyManager_Owlready2(world_store=None)
            # raise TypeError("neither ontology nor manager factory given")

        if ontology is None:
            if path is None:
                raise TypeError("path missing")
            else:
                self.ontology = self.manager.load_ontology(IRI.create('file://' + self.path))
                if isinstance(self.manager, OWLOntologyManager_Owlready2) and backend_store:
                    self.manager.save_world()
                    logger.debug("Synced world to backend store")

        reasoner: OWLReasoner
        if reasoner is not None:
            self.reasoner = reasoner
        elif reasoner_factory is not None:
            self.reasoner = reasoner_factory(self.ontology)
        else:
            self.reasoner = OWLReasoner_FastInstanceChecker(ontology=self.ontology,
                                                            base_reasoner=OWLReasoner_Owlready2(
                                                                ontology=self.ontology))

        self.length_metric = init_length_metric(length_metric, length_metric_factory)

        self.class_hierarchy: ClassHierarchy
        self.object_property_hierarchy: ObjectPropertyHierarchy
        self.data_property_hierarchy: DatatypePropertyHierarchy
        (self.class_hierarchy,
         self.object_property_hierarchy,
         self.data_property_hierarchy) = init_hierarchy_instances(self.reasoner,
                                                                  class_hierarchy=class_hierarchy,
                                                                  object_property_hierarchy=object_property_hierarchy,
                                                                  data_property_hierarchy=data_property_hierarchy)
        # Object property domain and range:
        self.op_domains: Dict[OWLObjectProperty, OWLClassExpression]
        self.op_domains = dict()
        self.op_ranges: Dict[OWLObjectProperty, OWLClassExpression]
        self.op_ranges = dict()
        # Data property domain and range:g
        self.dp_domains: Dict[OWLDataProperty, OWLClassExpression]
        self.dp_domains = dict()
        self.dp_ranges: Dict[OWLDataProperty, FrozenSet[OWLDataRange]]
        self.dp_ranges = dict()
        # OWL class expression generator
        self.generator = ConceptGenerator()

        self.use_individuals_cache, self.ind_cache = init_named_individuals(individuals_cache_size)
        self.ind_set = init_individuals_from_concepts(include_implicit_individuals,
                                                      reasoner=self.reasoner,
                                                      ontology=self.ontology,
                                                      individuals_per_concept=(self.individuals(i) for i in
                                                                               self.get_concepts()))

        self.describe()

    def individuals(self, concept: Optional[OWLClassExpression] = None) -> Iterable[OWLNamedIndividual]:
        """Given an OWL class expression, retrieve all individuals belonging to it.


        Args:
            concept: Class expression of which to list individuals.
        Returns:
            Individuals belonging to the given class.
        """

        if concept is None or concept.is_owl_thing():
            for i in self.ind_set:
                yield i
        else:
            yield from self.maybe_cache_individuals(concept)

    def abox(self, individual: Union[OWLNamedIndividual, Iterable[OWLNamedIndividual]] = None, mode='native'):
        """
        Get all the abox axioms for a given individual. If no individual is given, get all abox axioms

        Args:
            individual (OWLNamedIndividual): Individual/s to get the abox axioms from.
            mode (str): The return format.
             1) 'native' -> returns triples as tuples of owlapy objects,
             2) 'iri' -> returns triples as tuples of IRIs as string,
             3) 'axiom' -> triples are represented by owlapy axioms.

        Returns: Iterable of tuples or owlapy axiom, depending on the mode.
        """

        assert mode in ['native', 'iri', 'axiom',
                        "expression"], "Valid modes are: 'native', 'iri' ,'expression' or 'axiom'"

        if isinstance(individual, OWLNamedIndividual):
            inds = [individual]
        elif isinstance(individual, Iterable):
            inds = individual
        else:
            inds = self.individuals()

        for i in inds:
            if mode == "native":
                # Obtain all class assertion triples/axioms
                # For now, 'rdfs:type' predicate will be represented as an IRI
                yield from ((i, IRI.create("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"), t) for t in
                            self.get_types(ind=i, direct=True))

                # Obtain all property assertion triples/axioms
                for dp in self.get_data_properties_for_ind(ind=i):
                    yield from ((i, dp, literal) for literal in self.get_data_property_values(i, dp))

                for op in self.get_object_properties_for_ind(ind=i):
                    yield from ((i, op, ind) for ind in self.get_object_property_values(i, op))
            elif mode == "iri":
                yield from ((i.str, "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
                             t.str) for t in self.get_types(ind=i, direct=True))
                for dp in self.get_data_properties_for_ind(ind=i):
                    yield from ((i.str, dp.str, literal.get_literal()) for literal in
                                self.get_data_property_values(i, dp))
                for op in self.get_object_properties_for_ind(ind=i):
                    yield from ((i.str, op.str, ind.str) for ind in
                                self.get_object_property_values(i, op))
            elif mode == "axiom":
                yield from (OWLClassAssertionAxiom(i, t) for t in self.get_types(ind=i, direct=True))
                for dp in self.get_data_properties_for_ind(ind=i):
                    yield from (OWLDataPropertyAssertionAxiom(i, dp, literal) for literal in
                                self.get_data_property_values(i, dp))
                for op in self.get_object_properties_for_ind(ind=i):
                    yield from (OWLObjectPropertyAssertionAxiom(i, op, ind) for ind in
                                self.get_object_property_values(i, op))
            elif mode == "expression":
                object_restrictions_quantifiers = dict()
                # To no return duplicate objects.
                quantifier_gate = set()
                # (1) Iterate over triples where individual is in the subject position. Recursion
                for s, p, o in self.abox(individual=individual, mode="native"):
                    if isinstance(p, IRI) and isinstance(o, OWLClass):
                        """ Return OWLClass """
                        yield o
                    elif isinstance(p, OWLObjectProperty) and isinstance(o, OWLNamedIndividual):
                        """ STORE: ObjectSomeValuesFrom with ObjectOneOf over OWLNamedIndividual"""
                        object_restrictions_quantifiers.setdefault(p, []).append(o)
                    elif isinstance(p, OWLDataProperty) and isinstance(o, OWLLiteral):
                        """ RETURN: OWLDataSomeValuesFrom with OWLDataOneOf over OWLLiteral"""
                        yield OWLDataSomeValuesFrom(property=p, filler=OWLDataOneOf(o))
                    else:
                        raise RuntimeError("Unrecognized triples to expression mappings")

                for k, iter_inds in object_restrictions_quantifiers.items():
                    # RETURN Existential Quantifiers over Nominals: \exists r. {x....y}
                    for x in iter_inds:
                        yield OWLObjectSomeValuesFrom(property=k, filler=OWLObjectOneOf(values=x))
                    type_: OWLClass
                    count: int
                    for type_, count in Counter(
                            [type_i for i in iter_inds for type_i in self.get_types(ind=i, direct=True)]).items():
                        existential_quantifier = OWLObjectSomeValuesFrom(property=k, filler=type_)
                        if existential_quantifier in quantifier_gate:
                            continue
                        else:
                            # RETURN Existential Quantifiers over Concepts: \exists r. C
                            quantifier_gate.add(existential_quantifier)
                            yield existential_quantifier
                        if count > 1:
                            min_cardinality_item = OWLObjectMinCardinality(cardinality=count, property=k, filler=type_)
                            if min_cardinality_item in quantifier_gate:
                                continue
                            else:
                                quantifier_gate.add(min_cardinality_item)
                                # RETURN \ge number r. C
                                yield min_cardinality_item


            else:
                raise RuntimeError(f"Unrecognized mode:{mode}")

    def tbox(self, entities: Union[Iterable[OWLClass], Iterable[OWLDataProperty], Iterable[OWLObjectProperty], OWLClass,
    OWLDataProperty, OWLObjectProperty, None] = None, mode='native'):
        """Get all the tbox axioms for the given concept-s|propert-y/ies.
         If no concept-s|propert-y/ies are given, get all tbox axioms.

         Args:
             @TODO: entities or namedindividuals ?!
             entities: Entities to obtain tbox axioms from. This can be a single
              OWLClass/OWLDataProperty/OWLObjectProperty object, a list of those objects or None. If you enter a list
              that combines classes and properties (which we don't recommend doing), only axioms for one type will be
              returned, whichever comes first (classes or props).
             mode (str): The return format.
              1) 'native' -> returns triples as tuples of owlapy objects,
              2) 'iri' -> returns triples as tuples of IRIs as string,
              3) 'axiom' -> triples are represented by owlapy axioms.

         Returns: Iterable of tuples or owlapy axiom, depending on the mode.

        """
        assert mode in ['native', 'iri', 'axiom'], "Valid modes are: 'native', 'iri' or 'axiom'"
        if mode == "iri":
            print("WARN  KnowledgeBase.tbox()    :: Ranges of data properties are not implemented for the 'iri' mode!")
        include_all = False
        results = set()  # Using a set to avoid yielding duplicated results.
        classes = False
        if isinstance(entities, Iterable):
            ens = list(entities)
            if isinstance(ens[0], OWLClass):
                classes = True
        elif not isinstance(entities, Iterable) and entities:
            ens = [entities]
            if isinstance(entities, OWLClass):
                classes = True
        else:
            ens = list(self.get_concepts())
            ens.extend(list(self.get_object_properties()))
            ens.extend(list(self.get_data_properties()))
            include_all = True

        if include_all or classes:
            for concept in ens:
                if not isinstance(concept, OWLClass):
                    continue
                if mode == 'native':
                    [results.add((j, IRI.create("http://www.w3.org/2000/01/rdf-schema#subClassOf"), concept)) for j in
                     self.get_direct_sub_concepts(concept)]
                    [results.add((concept, IRI.create("http://www.w3.org/2002/07/owl#equivalentClass"), j)) for j in
                     self.reasoner.equivalent_classes(concept, only_named=True)]
                    if not include_all:  # This kind of check is just for performance purposes
                        [results.add((concept, IRI.create("http://www.w3.org/2000/01/rdf-schema#subClassOf"), j)) for j
                         in
                         self.get_direct_parents(concept)]
                elif mode == 'iri':
                    [results.add((j.str, "http://www.w3.org/2000/01/rdf-schema#subClassOf",
                                  concept.str)) for j in self.get_direct_sub_concepts(concept)]
                    [results.add((concept.str, "http://www.w3.org/2002/07/owl#equivalentClass",
                                  cast(OWLClass, j).str)) for j in
                     self.reasoner.equivalent_classes(concept, only_named=True)]
                    if not include_all:
                        [results.add((concept.str, "http://www.w3.org/2000/01/rdf-schema#subClassOf",
                                      j.str)) for j in self.get_direct_parents(concept)]
                elif mode == "axiom":
                    [results.add(OWLSubClassOfAxiom(super_class=concept, sub_class=j)) for j in
                     self.get_direct_sub_concepts(concept)]
                    [results.add(OWLEquivalentClassesAxiom([concept, j])) for j in
                     self.reasoner.equivalent_classes(concept, only_named=True)]
                    if not include_all:
                        [results.add(OWLSubClassOfAxiom(super_class=j, sub_class=concept)) for j in
                         self.get_direct_parents(concept)]
        if include_all or not classes:
            for prop in ens:
                if isinstance(prop, OWLObjectProperty):
                    prop_type = "Object"
                elif isinstance(prop, OWLDataProperty):
                    prop_type = "Data"
                else:
                    continue
                if mode == 'native':
                    [results.add((j, IRI.create("http://www.w3.org/2000/01/rdf-schema#subPropertyOf"), prop)) for j in
                     getattr(self.reasoner, "sub_" + prop_type.lower() + "_properties")(prop, direct=True)]
                    [results.add((prop, IRI.create("http://www.w3.org/2002/07/owl#equivalentProperty"), j)) for j in
                     getattr(self.reasoner, "equivalent_" + prop_type.lower() + "_properties")(prop)]
                    [results.add((prop, IRI.create("http://www.w3.org/2000/01/rdf-schema#domain"), j)) for j in
                     getattr(self.reasoner, prop_type.lower() + "_property_domains")(prop, direct=True)]
                    [results.add((prop, IRI.create("http://www.w3.org/2000/01/rdf-schema#range"), j)) for j in
                     getattr(self.reasoner, prop_type.lower() + "_property_ranges")(prop, direct=True)]
                    if not include_all:
                        [results.add((prop, IRI.create("http://www.w3.org/2000/01/rdf-schema#subPropertyOf"), j)) for j
                         in getattr(self.reasoner, "super_" + prop_type.lower() + "_properties")(prop, direct=True)]
                elif mode == 'iri':
                    [results.add((j.str, "http://www.w3.org/2000/01/rdf-schema#subPropertyOf",
                                  prop.str)) for j in
                     getattr(self.reasoner, "sub_" + prop_type.lower() + "_properties")(prop, direct=True)]
                    [results.add((prop.str, "http://www.w3.org/2002/07/owl#equivalentProperty",
                                  j.str)) for j in
                     getattr(self.reasoner, "equivalent_" + prop_type.lower() + "_properties")(prop)]
                    [results.add((prop.str, "http://www.w3.org/2000/01/rdf-schema#domain",
                                  j.str)) for j in
                     getattr(self.reasoner, prop_type.lower() + "_property_domains")(prop, direct=True)]
                    if prop_type == 'Object':
                        [results.add((prop.str, "http://www.w3.org/2000/01/rdf-schema#range",
                                      j.str)) for j in
                         self.reasoner.object_property_ranges(prop, direct=True)]
                    # # ranges of data properties not implemented for this mode
                    # else:
                    #     [results.add((prop.str, "http://www.w3.org/2000/01/rdf-schema#range",
                    #                   str(j))) for j in self.reasoner.data_property_ranges(prop, direct=True)]

                    if not include_all:
                        [results.add((prop.str, "http://www.w3.org/2000/01/rdf-schema#subPropertyOf",
                                      j.str)) for j
                         in getattr(self.reasoner, "super_" + prop_type.lower() + "_properties")(prop, direct=True)]
                elif mode == 'axiom':
                    [results.add(getattr(owlapy.owl_axiom, "OWLSub" + prop_type + "PropertyOfAxiom")(j, prop)) for j in
                     getattr(self.reasoner, "sub_" + prop_type.lower() + "_properties")(prop, direct=True)]
                    [results.add(getattr(owlapy.owl_axiom, "OWLEquivalent" + prop_type + "PropertiesAxiom")([j, prop]))
                     for
                     j in
                     getattr(self.reasoner, "equivalent_" + prop_type.lower() + "_properties")(prop)]
                    [results.add(getattr(owlapy.owl_axiom, "OWL" + prop_type + "PropertyDomainAxiom")(prop, j)) for j in
                     getattr(self.reasoner, prop_type.lower() + "_property_domains")(prop, direct=True)]
                    [results.add(getattr(owlapy.owl_axiom, "OWL" + prop_type + "PropertyRangeAxiom")(prop, j)) for j in
                     getattr(self.reasoner, prop_type.lower() + "_property_ranges")(prop, direct=True)]
                    if not include_all:
                        [results.add(getattr(owlapy.owl_axiom, "OWLSub" + prop_type + "PropertyOfAxiom")(prop, j)) for j
                         in getattr(self.reasoner, "super_" + prop_type.lower() + "_properties")(prop, direct=True)]

        return results

    def triples(self, mode="native"):
        """Get all tbox and abox axioms/triples.

        Args:
            mode (str): The return format.
              1) 'native' -> returns triples as tuples of owlapy objects,
              2) 'iri' -> returns triples as tuples of IRIs as string,
              3) 'axiom' -> triples are represented by owlapy axioms.

        Returns: Iterable of tuples or owlapy axiom, depending on the mode.
        """
        yield from self.abox(mode=mode)
        yield from self.tbox(mode=mode)

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
        new.manager = self.manager
        new.ontology = self.ontology
        new.reasoner = self.reasoner
        new.length_metric = self.length_metric
        new.ind_set = self.ind_set
        new.path = self.path
        new.use_individuals_cache = self.use_individuals_cache
        new.generator = self.generator
        new.op_domains = self.op_domains
        new.op_ranges = self.op_ranges
        new.dp_domains = self.dp_domains
        new.dp_ranges = self.dp_ranges

        if self.use_individuals_cache:
            new.ind_cache = LRUCache(maxsize=self.ind_cache.maxsize)

        if ignored_classes is not None:
            owl_concepts_to_ignore = set()
            for i in ignored_classes:
                if self.contains_class(i):
                    owl_concepts_to_ignore.add(i)
                else:
                    raise ValueError(
                        f'{i} could not found in \n{self} \n'
                        f'{[_ for _ in self.ontology.classes_in_signature()]}.')
            if logger.isEnabledFor(logging.INFO):
                r = DLSyntaxObjectRenderer()
                logger.info('Concepts to ignore: {0}'.format(' '.join(map(r.render, owl_concepts_to_ignore))))
            new.class_hierarchy = self.class_hierarchy.restrict_and_copy(remove=owl_concepts_to_ignore)
        else:
            new.class_hierarchy = self.class_hierarchy

        if ignored_object_properties is not None:
            new.object_property_hierarchy = self.object_property_hierarchy.restrict_and_copy(
                remove=ignored_object_properties)
        else:
            new.object_property_hierarchy = self.object_property_hierarchy

        if ignored_data_properties is not None:
            new.data_property_hierarchy = self.data_property_hierarchy.restrict_and_copy(
                remove=ignored_data_properties)
        else:
            new.data_property_hierarchy = self.data_property_hierarchy

        return new

    def concept_len(self, ce: OWLClassExpression) -> int:
        """Calculates the length of a concept and is used by some concept learning algorithms to
        find the best results considering also the length of the concepts.

        Args:
            ce: The concept to be measured.
        Returns:
            Length of the concept.
        """
        # @TODO: CD: Computing the length of a concept should be disantangled from KB
        # @TODO: CD: Ideally, this should be a static function

        return self.length_metric.length(ce)

    def clean(self):
        """Clean all stored values (states and caches) if there is any.

        Note:
            1. If you have more than one learning problem that you want to fit to the same model (i.e. to learn the
            concept using the same concept learner model) use this method to make sure that you have cleared every
            previous stored value.
            2. If you store another KnowledgeBase instance using the same variable name as before, it is recommended to
            use this method before the initialization to avoid data mismatch.
        """

        self.op_domains.clear()
        if self.use_individuals_cache:
            self.ind_cache.cache_clear()

    def cache_individuals(self, ce: OWLClassExpression) -> None:
        if not self.use_individuals_cache:
            raise TypeError
        if ce in self.ind_cache:
            return
        if isinstance(self.reasoner, OWLReasoner_FastInstanceChecker):
            self.ind_cache[ce] = self.reasoner._find_instances(ce)  # performance hack
        else:
            temp = self.reasoner.instances(ce)
            self.ind_cache[ce] = frozenset(temp)

    def maybe_cache_individuals(self, ce: OWLClassExpression) -> Iterable[OWLNamedIndividual]:
        if self.use_individuals_cache:
            self.cache_individuals(ce)
            yield from self.ind_cache[ce]
        else:
            yield from self.reasoner.instances(ce)

    def maybe_cache_individuals_count(self, ce: OWLClassExpression) -> int:
        if self.use_individuals_cache:
            self.cache_individuals(ce)
            r = self.ind_cache[ce]
            return len(r)
        else:
            return iter_count(self.reasoner.instances(ce))

    def individuals_count(self, concept: Optional[OWLClassExpression] = None) -> int:
        """Returns the number of all individuals belonging to the concept in the ontology.

        Args:
            concept: Class expression of the individuals to count.
        Returns:
            Number of the individuals belonging to the given class.
        """

        if concept is None or concept.is_owl_thing():
            return len(self.ind_set)
        else:
            return self.maybe_cache_individuals_count(concept)

    @overload
    def individuals_set(self, concept: OWLClassExpression):
        ...

    @overload
    def individuals_set(self, individual: OWLNamedIndividual):
        ...

    @overload
    def individuals_set(self, individuals: Iterable[OWLNamedIndividual]):
        ...

    def individuals_set(self,
                        arg: Union[Iterable[OWLNamedIndividual], OWLNamedIndividual, OWLClassExpression]) -> FrozenSet:
        """Retrieve the individuals specified in the arg as a frozenset. If `arg` is an OWLClassExpression then this
        method behaves as the method "individuals" but will return the final result as a frozenset.

        Args:
            arg: more than one individual/ single individual/ class expression of which to list individuals.
        Returns:
            Frozenset of the individuals depending on the arg type.
        """

        if isinstance(arg, OWLClassExpression):
            if self.use_individuals_cache:
                self.cache_individuals(arg)
                r = self.ind_cache[arg]
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

        if self.ind_set is not None:
            return self.ind_set
        else:
            return frozenset(self.ontology.individuals_in_signature())

    def most_general_object_properties(self, *, domain: OWLClassExpression, inverse: bool = False) \
            -> Iterable[OWLObjectProperty]:
        """Find the most general object property.

        Args:
            domain: Domain for which to search properties.
            inverse: Inverse order?
        """
        assert isinstance(domain, OWLClassExpression)
        func: Callable
        func = self.get_object_property_ranges if inverse else self.get_object_property_domains

        inds_domain = self.individuals_set(domain)
        for prop in self.object_property_hierarchy.most_general_roles():
            if domain.is_owl_thing() or inds_domain <= self.individuals_set(func(prop)):
                yield prop

    def data_properties_for_domain(self, domain: OWLClassExpression, data_properties: Iterable[OWLDataProperty]) \
            -> Iterable[OWLDataProperty]:
        assert isinstance(domain, OWLClassExpression)

        inds_domain = self.individuals_set(domain)
        for prop in data_properties:
            if domain.is_owl_thing() or inds_domain <= self.individuals_set(self.get_data_property_domains(prop)):
                yield prop

    # in case more types of AbstractLearningProblem are introduced to the project uncomment the method below and use
    # decorators
    # @singledispatchmethod
    # def encode_learning_problem(self, lp: AbstractLearningProblem):
    #     raise NotImplementedError(lp)

    def encode_learning_problem(self, lp: PosNegLPStandard):
        """
        @TODO: A learning problem (DL concept learning problem) should not be a part of a knowledge base

        Provides the encoded learning problem (lp), i.e. the class containing the set of OWLNamedIndividuals
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

        @ TODO: A knowledge base is a data structure and the context of "evaluating" a concept seems to be unrelated

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

    def get_leaf_concepts(self, concept: OWLClass):
        """Get leaf classes.

        Args:
            concept: Atomic class for which to find leaf classes.

        Returns:
            Leaf classes { x \\| (x subClassOf concept) AND not exist y: y subClassOf x )}. """
        assert isinstance(concept, OWLClass)
        yield from self.class_hierarchy.leaves(of=concept)

    def get_least_general_named_concepts(self) -> Generator[OWLClass, None, None]:
        """Get leaf classes.
        @TODO: Docstring needed
        Returns:
        """
        yield from self.class_hierarchy.leaves()

    def least_general_named_concepts(self) -> Generator[OWLClass, None, None]:
        """Get leaf classes.
        @TODO: Docstring needed
        Returns:
        """
        yield from self.class_hierarchy.leaves()

    def get_most_general_classes(self) -> Generator[OWLClass, None, None]:
        """Get most general named concepts classes.
        @TODO: Docstring needed
        Returns:"""
        yield from self.class_hierarchy.roots()

    def get_direct_sub_concepts(self, concept: OWLClass) -> Iterable[OWLClass]:
        """Direct sub-classes of atomic class.

        Args:
            concept: Atomic concept.

        Returns:
            Direct sub classes of concept { x \\| ( x subClassOf concept )}."""
        assert isinstance(concept, OWLClass)
        yield from self.class_hierarchy.sub_classes(concept, direct=True)

    def get_object_property_domains(self, prop: OWLObjectProperty) -> OWLClassExpression:
        """Get the domains of an object property.

        Args:
            prop: Object property.

        Returns:
            Domains of the property.
        """
        if prop not in self.op_domains:
            domains = list(self.reasoner.object_property_domains(prop, direct=True))
            self.op_domains[prop] = self.generator.intersection(domains) if len(domains) > 1 else domains[0]
        return self.op_domains[prop]

    def get_object_property_ranges(self, prop: OWLObjectProperty) -> OWLClassExpression:
        """Get the ranges of an object property.

        Args:
            prop: Object property.

        Returns:
            Ranges of the property.
        """
        if prop not in self.op_ranges:
            ranges = list(self.reasoner.object_property_ranges(prop, direct=True))
            self.op_ranges[prop] = self.generator.intersection(ranges) if len(ranges) > 1 else ranges[0]
        return self.op_ranges[prop]

    def get_data_property_domains(self, prop: OWLDataProperty) -> OWLClassExpression:
        """Get the domains of a data property.

        Args:
            prop: Data property.

        Returns:
            Domains of the property.
        """
        if prop not in self.dp_domains:
            domains = list(self.reasoner.data_property_domains(prop, direct=True))
            self.dp_domains[prop] = self.generator.intersection(domains) if len(domains) > 1 else domains[0]
        return self.dp_domains[prop]

    def get_data_property_ranges(self, prop: OWLDataProperty) -> FrozenSet[OWLDataRange]:
        """Get the ranges of a data property.

        Args:
            prop: Data property.

        Returns:
            Ranges of the property.
        """
        if prop not in self.dp_ranges:
            self.dp_ranges[prop] = frozenset(self.reasoner.data_property_ranges(prop, direct=True))
        return self.dp_ranges[prop]

    def most_general_data_properties(self, *, domain: OWLClassExpression) -> Iterable[OWLDataProperty]:
        """Find most general data properties that are applicable to a domain.

        Args:
            domain: Domain for which to search properties.

        Returns:
            Most general data properties for the given domain.
        """
        yield from self.data_properties_for_domain(domain, self.get_data_properties())

    def most_general_boolean_data_properties(self, *, domain: OWLClassExpression) -> Iterable[OWLDataProperty]:
        """Find most general boolean data properties that are applicable to a domain.

        Args:
            domain: Domain for which to search properties.

        Returns:
            Most general boolean data properties for the given domain.
        """
        yield from self.data_properties_for_domain(domain, self.get_boolean_data_properties())

    def most_general_numeric_data_properties(self, *, domain: OWLClassExpression) -> Iterable[OWLDataProperty]:
        """Find most general numeric data properties that are applicable to a domain.

        Args:
            domain: Domain for which to search properties.

        Returns:
            Most general numeric data properties for the given domain.
        """
        yield from self.data_properties_for_domain(domain, self.get_numeric_data_properties())

    def most_general_time_data_properties(self, *, domain: OWLClassExpression) -> Iterable[OWLDataProperty]:
        """Find most general time data properties that are applicable to a domain.

        Args:
            domain: Domain for which to search properties.

        Returns:
            Most general time data properties for the given domain.
        """
        yield from self.data_properties_for_domain(domain, self.get_time_data_properties())

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
        yield from self.class_hierarchy.super_classes(concept, direct=True)

    def get_all_direct_sub_concepts(self, concept: OWLClassExpression) -> Iterable[OWLClassExpression]:
        """All direct sub concepts of a concept.

        Args:
            concept: Parent concept for which to get sub concepts.

        Returns:
            Direct sub concepts.
        """
        assert isinstance(concept, OWLClass)
        yield from self.class_hierarchy.sub_classes(concept, direct=True)

    def get_all_sub_concepts(self, concept: OWLClassExpression) -> Iterable[OWLClassExpression]:
        """All sub concepts of a concept.

        Args:
            concept: Parent concept for which to get sub concepts.

        Returns:
            Sub concepts.
        """
        assert isinstance(concept, OWLClass)
        yield from self.class_hierarchy.sub_classes(concept, direct=False)

    def get_concepts(self) -> Iterable[OWLClass]:
        """Get all concepts of this concept generator.

        Returns:
            Concepts.
        """
        yield from self.class_hierarchy.items()

    def get_classes_in_signature(self):
        return self.get_concepts()

    @property
    def concepts(self) -> Iterable[OWLClass]:
        """Get all concepts of this concept generator.

        Returns:
            Concepts.
        """
        yield from self.class_hierarchy.items()

    @property
    def object_properties(self) -> Iterable[OWLObjectProperty]:
        """Get all object properties of this concept generator.

        Returns:
            Object properties.
        """

        yield from self.object_property_hierarchy.items()

    @property
    def data_properties(self) -> Iterable[OWLDataProperty]:
        """Get all data properties of this concept generator.

        Returns:
            Data properties for the given range.
        """
        yield from self.data_property_hierarchy.items()

    def get_object_properties(self) -> Iterable[OWLObjectProperty]:
        """Get all object properties of this concept generator.

        Returns:
            Object properties.
        """

        yield from self.object_property_hierarchy.items()

    def get_data_properties(self, ranges: Set[OWLDatatype] = None) -> Iterable[OWLDataProperty]:
        """Get all data properties of this concept generator for the given ranges.

        Args:
           ranges: Ranges for which to extract the data properties.

        Returns:
            Data properties for the given range.
        """
        if ranges is not None:
            for dp in self.data_property_hierarchy.items():
                if self.get_data_property_ranges(dp) & ranges:
                    yield dp
        else:
            yield from self.data_property_hierarchy.items()

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

    def get_double_data_properties(self) -> Iterable[OWLDataProperty]:
        """Get all numeric data properties of this concept generator.

        Returns:
            Numeric data properties.
        """
        yield from self.get_data_properties(DoubleOWLDatatype)

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
        for type_ in self.reasoner.types(ind, direct):
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
        yield from (pe for pe in self.reasoner.ind_object_properties(ind, direct) if pe in properties)

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
        yield from (pe for pe in self.reasoner.ind_data_properties(ind, direct) if pe in properties)

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
        yield from self.reasoner.object_property_values(ind, property_, direct)

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
        yield from self.reasoner.data_property_values(ind, property_, direct)

    def contains_class(self, concept: OWLClassExpression) -> bool:
        """Check if an atomic class is contained within this concept generator.

        Args:
            concept: Atomic class.

        Returns:
            Whether the class is contained in the concept generator.
        """
        assert isinstance(concept, OWLClass)
        return concept in self.class_hierarchy

    def __repr__(self):
        properties_count = iter_count(self.ontology.object_properties_in_signature()) + iter_count(
            self.ontology.data_properties_in_signature())
        class_count = iter_count(self.ontology.classes_in_signature())
        individuals_count = self.individuals_count()

        return f'KnowledgeBase(path={repr(self.path)} <{class_count} classes, {properties_count} properties, ' \
               f'{individuals_count} individuals)'
