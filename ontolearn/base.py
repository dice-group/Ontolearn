from collections import defaultdict
from itertools import chain
from owlready2 import Ontology, get_ontology
import owlready2
from .concept_generator import ConceptGenerator
from .concept import Concept
from typing import Dict, Tuple, Set, Generator, Iterable
from .util import parametrized_performance_debugger
import warnings

warnings.filterwarnings("ignore")


class KnowledgeBase:
    """Knowledge Base Class representing Tbox and Abox along with concept hierarchies"""

    def __init__(self, path):
        self.path = path
        self.onto = get_ontology(self.path).load(reload=True)
        self.name = self.onto.name
        self.concepts = dict()
        self.thing = None
        self.nothing = None
        self.top_down_concept_hierarchy = defaultdict(set)  # Next time thing about including this into Concepts.
        self.top_down_direct_concept_hierarchy = defaultdict(set)
        self.down_top_concept_hierarchy = defaultdict(set)
        self.down_top_direct_concept_hierarchy = defaultdict(set)
        self.concepts_to_leafs = defaultdict(set)
        self.property_hierarchy = None
        self.parse()
        # self.concepts = MappingProxyType(self.concepts)
        self.__concept_generator = ConceptGenerator(concepts=self.concepts, T=self.thing, Bottom=self.nothing,
                                                    onto=self.onto)

    @staticmethod
    def apply_type_enrichment_from_iterable(concepts: Iterable[Concept]):
        """
        Extend ABOX by
        (1) Obtaining all instances of selected concepts.
        (2) For all instances in (1) include type information into ABOX.
        @param concepts:
        @return:
        """
        for c in concepts:
            for ind in c.owl.instances():
                ind.is_a.append(c.owl)

    @staticmethod
    def apply_type_enrichment(concept: Concept):
        for ind in concept.owl.instances():
            ind.is_a.append(concept.owl)

    def save(self, path, rdf_format='ntriples'):
        self.onto.save(file=path, format=rdf_format)

    def get_all_individuals(self) -> Set:
        return self.thing.instances

    @staticmethod
    def is_atomic(c: owlready2.entity.ThingClass):
        assert isinstance(c, owlready2.entity.ThingClass)
        if '¬' in c.name and not (' ' in c.name):
            return False
        elif ' ' in c.name or '∃' in c.name or '∀' in c.name:
            return False
        else:
            return True

    def __parse_complex(self, c: owlready2.entity.ThingClass):
        assert isinstance(c, owlready2.entity.ThingClass)
        assert len(c.is_a) == 2
        print(c)

        owl_type = c.is_a[1]
        print(owl_type)
        print(type(owl_type))

        print(owl_type.property)
        print(owl_type.value)

        exit(1)

        return True

    def __build_concepts_mapping(self, onto: Ontology) -> Tuple[Dict, Concept, Concept]:
        """
        Construct a mapping from full_iri to corresponding Concept objects.

        concept.namespace.base_iri + concept.name
        mappings from concepts uri to concept objects
            1) full_iri:= owlready2.ThingClass.namespace.base_iri + owlready2.ThingClass.name
            2) Concept:
        """
        concepts = dict()
        T = Concept(owlready2.Thing, kwargs={'form': 'Class'})
        T.owl.equivalent_to.append(owlready2.Thing)
        bottom = Concept(owlready2.Nothing, kwargs={'form': 'Class'})
        complex_classes = []
        for i in onto.classes():
            if self.is_atomic(i):
                temp_concept = Concept(i, kwargs={'form': 'Class'})  # wrap owl object into AtomicConcept.
                concepts[temp_concept.full_iri] = temp_concept
            else:
                complex_classes.append(i)


        #self.__parse_complex(complex_=complex_classes,concepts_dict)

        concepts[T.full_iri] = T
        concepts[bottom.full_iri] = bottom
        return concepts, T, bottom

    def __build_hierarchy(self, onto: Ontology) -> None:
        """
        Builds concept sub and super classes hierarchies.

        1) self.top_down_concept_hierarchy is a mapping from Concept objects to a set of Concept objects that are
        direct subclasses of given Concept object.

        2) self.down_top_concept_hierarchy is a mapping from Concept objects to set of Concept objects that are
        direct superclasses of given Concept object.

        """

        self.concepts, self.thing, self.nothing = self.__build_concepts_mapping(onto)

        self.down_top_concept_hierarchy[self.thing] = set()
        self.top_down_concept_hierarchy[self.thing] = {_ for _ in self.concepts.values()}

        for str_, concept_A in self.concepts.items():  # second loop over concepts in the execution,

            for desc in concept_A.owl.descendants(include_self=False):

                wrapped_desc = self.concepts[desc.namespace.base_iri + desc.name]

                # Include all sub class that are wrapped with AtomicConcept class into hierarchy.
                self.top_down_concept_hierarchy[concept_A].add(wrapped_desc)
                if len(wrapped_desc.owl.descendants(
                        include_self=False)) == 0:  # if no descendant, then it is a leaf concept.
                    self.concepts_to_leafs.setdefault(concept_A, set()).add(wrapped_desc)

            for ans in concept_A.owl.ancestors(include_self=False):
                wrapped_ans = self.concepts[ans.namespace.base_iri + ans.name]
                # Include all superclasses into down top hierarchy
                self.down_top_concept_hierarchy[concept_A].add(wrapped_ans)

            for subs in concept_A.owl.subclasses():  # returns direct subclasses
                if concept_A.owl == subs:
                    continue
                wrapped_subs = self.concepts[subs.namespace.base_iri + subs.name]

                self.top_down_direct_concept_hierarchy[concept_A].add(wrapped_subs)
                self.down_top_direct_concept_hierarchy[wrapped_subs].add(concept_A)

    def parse(self):
        """
        Top-down and bottom up hierarchies are constructed from from owlready2.Ontology
        """
        self.__build_hierarchy(self.onto)
        self.property_hierarchy = PropertyHierarchy(self.onto)

    def get_leaf_concepts(self, concept: Concept):
        """ Return : { x | (x subClassOf concept) AND not exist y: y subClassOf x )} """
        for leaf in self.concepts_to_leafs[concept]:
            yield leaf

    def negation(self, concept: Concept):
        """ Return a Concept object that is a negation of given concept."""
        yield self.__concept_generator.negation(concept)

    @parametrized_performance_debugger()
    def negation_from_iterables(self, s: Generator):
        """ Return : { x | ( x \equv not s} """
        for item in s:
            yield self.__concept_generator.negation(item)

    @parametrized_performance_debugger()
    def get_direct_sub_concepts(self, concept: Concept):
        """ Return : { x | ( x subClassOf concept )} """
        for v in self.top_down_direct_concept_hierarchy[concept]:
            yield v

    def get_all_sub_concepts(self, concept: Concept):
        """ Return : { x | ( x subClassOf concept ) OR ..."""
        for v in self.top_down_concept_hierarchy[concept]:
            yield v

    def get_direct_parents(self, concept: Concept):
        """ Return : { x | (concept subClassOf x)} """
        for direct_parent in self.down_top_direct_concept_hierarchy[concept]:
            yield direct_parent

    def most_general_existential_restrictions(self, concept: Concept):

        for prob in self.property_hierarchy.get_most_general_property():
            yield self.__concept_generator.existential_restriction(concept, prob)

    def most_general_universal_restriction(self, concept: Concept):
        assert isinstance(concept, Concept)
        for prob in self.property_hierarchy.get_most_general_property():
            yield self.__concept_generator.universal_restriction(concept, prob)

    def union(self, conceptA: Concept, conceptB: Concept):
        """Return a concept c == (conceptA OR conceptA)"""
        assert isinstance(conceptA, Concept)
        assert isinstance(conceptB, Concept)
        return self.__concept_generator.union(conceptA, conceptB)

    def intersection(self, conceptA: Concept, conceptB: Concept) -> Generator:
        """Return a concept c == (conceptA AND conceptA)"""
        assert isinstance(conceptA, Concept)
        assert isinstance(conceptB, Concept)
        return self.__concept_generator.intersection(conceptA, conceptB)

    def existential_restriction(self, concept: Concept, property_) -> Generator:
        """Return a concept c == (Exist R.C)"""
        assert isinstance(concept, Concept)
        return self.__concept_generator.existential_restriction(concept, property_)

    def universal_restriction(self, concept: Concept, property_) -> Generator:
        """Return a concept c == (Forall R.C)"""
        assert isinstance(concept, Concept)
        return self.__concept_generator.universal_restriction(concept, property_)

    def num_concepts_generated(self):

        return len(self.__concept_generator.log_of_universal_restriction) + len(
            self.__concept_generator.log_of_existential_restriction) + \
               len(self.__concept_generator.log_of_intersections) + len(self.__concept_generator.log_of_unions) + \
               len(self.__concept_generator.log_of_negations) + len(self.concepts)

    def get_all_concepts(self):
        return set(chain(self.concepts,
                         self.__concept_generator.log_of_universal_restriction.values(),
                         self.__concept_generator.log_of_negations.values(),
                         self.__concept_generator.log_of_intersections.values(),
                         self.__concept_generator.log_of_universal_restriction.values(),
                         self.__concept_generator.log_of_existential_restriction.values()))


class PropertyHierarchy:

    def __init__(self, onto):
        self.all_properties = [i for i in onto.properties()]

        self.data_properties = [i for i in onto.data_properties()]

        self.object_properties = [i for i in onto.object_properties()]

    def get_most_general_property(self):
        for i in self.all_properties:
            yield i
