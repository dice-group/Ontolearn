from collections import defaultdict
from .concept_generator import ConceptGenerator
from .concept import Concept
from typing import Dict, Tuple, Set, Generator, Iterable, List

from .core.owl import ClassHierarchy
from .owlapy.model import OWLOntologyManager, OWLOntology, OWLReasoner
from .owlapy.owlready2 import OWLOntologyManager_Owlready2
from .util import parametrized_performance_debugger, get_full_iri
from .abstracts import AbstractKnowledgeBase
import warnings
from .static_funcs import build_concepts_mapping

#warnings.filterwarnings("ignore")


class KnowledgeBase(AbstractKnowledgeBase):
    """ Knowledge Base Class representing Tbox and Abox along with the concept hierarchy """

    manager: OWLOntologyManager
    onto: OWLOntology
    reasoner: OWLReasoner

    def __init__(self, path):
        super().__init__()
        self.path = path
        self.manager = OWLOntologyManager_Owlready2()
        self.onto = self.manager.load_ontology('file://' + self.path)
        self.reasoner = OWLReasoner
        self.class_hierarchy = ClassHierarchy(self.reasoner)
        self.property_hierarchy = PropertyHierarchy(self.onto)
        self.name = self.onto.name
        self.parse()
        self.str_to_instance_obj = dict(zip([get_full_iri(i) for i in self.thing.instances], self.thing.instances))
        self.obj_to_str_iri_instances = dict(
            zip(self.str_to_instance_obj.values(), self.str_to_instance_obj.keys()))
        self.idx_of_instances = dict(zip(self.thing.instances, range(len(self.str_to_instance_obj))))
        self.uri_individuals = list(self.str_to_instance_obj.keys())
        self.concept_generator = ConceptGenerator(concepts=self.uri_to_concepts,
                                                  thing=self.thing,
                                                  nothing=self.nothing,
                                                  onto=self.onto)
        self.describe()

    def clean(self):
        """
        Clearn all stored values if there is any.
        @return:
        """

    def set_min_size_of_concept(self, n):
        self.min_size_of_concept = n

    def max_size_of_concept(self, n):
        self.max_size_of_concept = n

    def get_leaf_concepts(self, concept: Concept) -> Generator:
        """ Return : { x | (x subClassOf concept) AND not exist y: y subClassOf x )} """
        assert isinstance(concept, Concept)
        for leaf in self.concepts_to_leafs[concept]:
            yield leaf

    @parametrized_performance_debugger()
    def negation_from_iterables(self, s: Generator) -> Generator:
        """ Return : { x | ( x \equv not s} """
        assert isinstance(s, Generator)
        for item in s:
            yield self.concept_generator.negation(item)

    @parametrized_performance_debugger()
    def get_direct_sub_concepts(self, concept: Concept) -> Generator:
        """ Return : { x | ( x subClassOf concept )} """
        assert isinstance(concept, Concept)
        for v in self.top_down_direct_concept_hierarchy[concept]:
            yield v

    def most_general_existential_restrictions(self, concept: Concept) -> Generator:
        """ Return : { \exist.r.x | r \in MostGeneral r} """
        assert isinstance(concept, Concept)
        for prob in self.property_hierarchy.get_most_general_property():
            yield self.concept_generator.existential_restriction(concept, prob)

    def most_general_universal_restrictions(self, concept: Concept) -> Generator:
        """ Return : { \forall.r.x | r \in MostGeneral r} """
        assert isinstance(concept, Concept)
        for prob in self.property_hierarchy.get_most_general_property():
            yield self.concept_generator.universal_restriction(concept, prob)

    def num_concepts_generated(self) -> int:
        """ Return: the number of all concepts """
        return len(self.concept_generator.log_of_universal_restriction) + len(
            self.concept_generator.log_of_existential_restriction) + \
               len(self.concept_generator.log_of_intersections) + len(self.concept_generator.log_of_unions) + \
               len(self.concept_generator.log_of_negations) + len(self.concepts)
