from collections import defaultdict
from typing import Dict, Tuple, Set, Generator, Iterable, List, Type, Optional, Callable

from .core.owl.hierarchy import ClassHierarchy, ObjectPropertyHierarchy, DataPropertyHierarchy
from .owlapy.model import OWLOntologyManager, OWLOntology, OWLReasoner, OWLClassExpression, OWLObjectComplementOf
from .owlapy.owlready2 import OWLOntologyManager_Owlready2
from .utils import parametrized_performance_debugger
from .owlready2.utils import get_full_iri
from .abstracts import AbstractKnowledgeBase
import warnings
from .static_funcs import build_concepts_mapping

#warnings.filterwarnings("ignore")


class KnowledgeBase(AbstractKnowledgeBase):
    """ Knowledge Base Class representing Tbox and Abox along with the concept hierarchy """

    manager: OWLOntologyManager
    onto: OWLOntology
    reasoner: OWLReasoner

    class_hierarchy: ClassHierarchy
    object_property_hierarchy: ObjectPropertyHierarchy
    data_property_hierarchy: DataPropertyHierarchy

    def __init__(self, *, path: Optional[str] = None,
                 ontologymanager_factory: Callable[[], OWLOntologyManager],
                 reasoner_factory: Type[OWLReasoner]):
        super().__init__()
        self.path = path
        self.manager = ontologymanager_factory()
        self.onto = self.manager.load_ontology('file://' + self.path)
        self.reasoner = reasoner_factory(self.onto)
        self.class_hierarchy = ClassHierarchy(self.reasoner)
        self.object_property_hierarchy = ObjectPropertyHierarchy(self.reasoner)
        self.data_property_hierarchy = DataPropertyHierarchy(self.reasoner)
        self.describe()

    def ontology(self) -> OWLOntology:
        return self.onto

    def clean(self):
        """
        Clearn all stored values if there is any.
        @return:
        """

    def set_min_size_of_concept(self, n):
        self.min_size_of_concept = n

    def max_size_of_concept(self, n):
        self.max_size_of_concept = n

    def get_leaf_concepts(self, concept: OWLClassExpression) -> Iterable[OWLClassExpression]:
        """ Return : { x | (x subClassOf concept) AND not exist y: y subClassOf x )} """
        assert isinstance(concept, OWLClassExpression)
        self.class_hierarchy.leaves(of=concept)
        for leaf in self.concepts_to_leafs[concept]:
            yield leaf

    @parametrized_performance_debugger()
    def negation_from_iterables(self, s: Iterable[OWLClassExpression]) -> Iterable[OWLClassExpression]:
        """ Return : { x | ( x \equv not s} """
        assert isinstance(s, Generator)
        for item in s:
            yield item.get_object_complement_of()

    @parametrized_performance_debugger()
    def get_direct_sub_concepts(self, concept: OWLClassExpression) -> Iterable[OWLClassExpression]:
        """ Return : { x | ( x subClassOf concept )} """
        assert isinstance(concept, OWLClassExpression)
        for v in self.top_down_direct_concept_hierarchy[concept]:
            yield v

    def most_general_existential_restrictions(self, concept: OWLClassExpression) -> Iterable[OWLClassExpression]:
        """ Return : { \\exist.r.x | r \\in MostGeneral r} """
        assert isinstance(concept, OWLClassExpression)
        for prob in self.property_hierarchy.get_most_general_property():
            yield self.concept_generator.existential_restriction(concept, prob)

    def most_general_universal_restrictions(self, concept: OWLClassExpression) -> Iterable[OWLClassExpression]:
        """ Return : { \\forall.r.x | r \\in MostGeneral r} """
        assert isinstance(concept, OWLClassExpression)
        for prob in self.property_hierarchy.get_most_general_property():
            yield self.concept_generator.universal_restriction(concept, prob)
