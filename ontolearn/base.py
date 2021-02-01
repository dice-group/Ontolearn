from typing import Dict, Tuple, Set, Generator, Iterable, List, Type, Optional, Callable, TypeVar, overload

from .core.owl.hierarchy import ClassHierarchy, ObjectPropertyHierarchy, DatatypePropertyHierarchy
from .owlapy import IRI
from .owlapy.fast_instance_checker import OWLReasoner_FastInstanceChecker
from .owlapy.model import OWLOntologyManager, OWLOntology, OWLReasoner, OWLClassExpression, OWLObjectComplementOf, \
    OWLClass, OWLObjectSomeValuesFrom, OWLObjectAllValuesFrom
from .owlapy.owlready2 import OWLOntologyManager_Owlready2
from .owlapy.owlready2.base import OWLOntology_Owlready2
from .owlapy.owlready2.temp_classes import OWLReasoner_Owlready2_TempClasses
from .utils import parametrized_performance_debugger
from .abstracts import AbstractKnowledgeBase
import warnings

# warnings.filterwarnings("ignore")

Factory = Callable

_Default_OntologyManagerFactory = OWLOntologyManager_Owlready2


def _Default_ReasonerFactory(onto: OWLOntology) -> OWLReasoner_FastInstanceChecker:
    assert isinstance(onto, OWLOntology_Owlready2)
    base_reasoner = OWLReasoner_Owlready2_TempClasses(ontology=onto)
    reasoner = OWLReasoner_FastInstanceChecker(ontology=onto,
                                               base_reasoner=base_reasoner)
    return reasoner


class KnowledgeBase(AbstractKnowledgeBase):
    """ Knowledge Base Class representing Tbox and Abox along with the concept hierarchy """

    manager: OWLOntologyManager
    onto: OWLOntology
    reasoner: OWLReasoner

    class_hierarchy: ClassHierarchy
    object_property_hierarchy: ObjectPropertyHierarchy
    data_property_hierarchy: DatatypePropertyHierarchy

    @overload
    def __init__(self, *,
                 path: str,
                 ontologymanager_factory: Factory[[], OWLOntologyManager] = _Default_OntologyManagerFactory,
                 reasoner_factory: Factory[[OWLOntology], OWLReasoner] = _Default_ReasonerFactory):
        ...

    @overload
    def __init__(self, *, ontology: OWLOntology, reasoner: OWLReasoner):
        ...

    def __init__(self, *,
                 path: Optional[str] = None,
                 ontologymanager_factory: Optional[Factory[[], OWLOntologyManager]] = None,
                 reasoner_factory: Optional[Factory[[OWLOntology], OWLReasoner]] = None,
                 ontology: Optional[OWLOntology] = None,
                 reasoner: Optional[OWLReasoner] = None):
        super().__init__()
        self.path = path
        if ontology is not None:
            self.manager = ontology.get_manager()
            self.onto = ontology
        elif ontologymanager_factory is not None:
            self.manager = ontologymanager_factory()
        else:  # default to Owlready2 implementation
            self.manager = _Default_OntologyManagerFactory()
            # raise TypeError("neither ontology nor manager factory given")

        if ontology is None:
            if path is None:
                raise TypeError("path missing")
            self.onto = self.manager.load_ontology(IRI.create('file://' + self.path))

        if reasoner is not None:
            self.reasoner = reasoner
        elif reasoner_factory is not None:
            self.reasoner = reasoner_factory(self.onto)
        else:  # default to fast instance checker
            self.reasoner = _Default_ReasonerFactory(self.onto)
            # raise TypeError("neither reasoner nor reasoner factory given")

        self.class_hierarchy = ClassHierarchy(self.reasoner)
        self.object_property_hierarchy = ObjectPropertyHierarchy(self.reasoner)
        self.data_property_hierarchy = DatatypePropertyHierarchy(self.reasoner)
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

    def get_leaf_concepts(self, concept: OWLClass) -> Iterable[OWLClass]:
        """ Return : { x | (x subClassOf concept) AND not exist y: y subClassOf x )} """
        assert isinstance(concept, OWLClass)
        yield from self.class_hierarchy.leaves(of=concept)

    @parametrized_performance_debugger()
    def negation_from_iterables(self, s: Iterable[OWLClassExpression]) -> Iterable[OWLClassExpression]:
        """ Return : { x | ( x \equv not s} """
        for item in s:
            yield item.get_object_complement_of()

    @parametrized_performance_debugger()
    def get_direct_sub_concepts(self, concept: OWLClass) -> Iterable[OWLClass]:
        """ Return : { x | ( x subClassOf concept )} """
        assert isinstance(concept, OWLClass)
        yield from self.class_hierarchy.sub_classes(concept, direct=True)

    def most_general_existential_restrictions(self, concept: OWLClassExpression) -> Iterable[OWLObjectSomeValuesFrom]:
        """ Return : { \\exist.r.x | r \\in MostGeneral r} """
        assert isinstance(concept, OWLClassExpression)
        for prob in self.object_property_hierarchy.most_general_roles():
            yield OWLObjectSomeValuesFrom(property=prob, filler=concept)

    def most_general_universal_restrictions(self, concept: OWLClassExpression) -> Iterable[OWLObjectAllValuesFrom]:
        """ Return : { \\forall.r.x | r \\in MostGeneral r} """
        assert isinstance(concept, OWLClassExpression)
        for prob in self.object_property_hierarchy.most_general_roles():
            yield OWLObjectAllValuesFrom(property=prob, filler=concept)
