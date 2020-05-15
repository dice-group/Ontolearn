from collections import defaultdict
from owlready2 import get_ontology, Ontology, Thing, Nothing
from core.concept_generator import ConceptGenerator
from core.concept import Concept
from typing import Dict, Tuple, Set

class KnowledgeBase:
    """Knowledge Base Class representing Tbox and Abox along with concept hierarchies"""
    def __init__(self, path):

        self._kb_path = path
        self.property_hierarchy = None
        self.onto = get_ontology(self._kb_path).load()
        self.name = self.onto.name
        self.concepts = dict()
        self.T = None
        self.Bottom = None
        self.top_down_concept_hierarchy = defaultdict(set)  # Next time thing about including this into Concepts.
        self.top_down_direct_concept_hierarchy = defaultdict(set)
        self.down_top_concept_hierarchy = defaultdict(set)
        self.down_top_direct_concept_hierarchy = defaultdict(set)
        self.concepts_to_leafs = defaultdict(set)
        self.parse()
        self.__concept_generator = ConceptGenerator(concepts=self.concepts, T=self.T, Bottom=self.Bottom,
                                                    onto=self.onto)

    def get_individuals(self) -> Set:
        return self.T.instances()

    @staticmethod
    def __build_concepts_mapping(onto: Ontology) -> Tuple[Dict, Concept, Concept]:
        """
        Construct a mapping from full_iri to corresponding Concept objects.

        concept.namespace.base_iri + concept.name
        mappings from concepts uri to concept objects
            1) full_iri:= owlready2.ThingClass.namespace.base_iri + owlready2.ThingClass.name
            2) Concept:
        """
        concepts = dict()
        T = Concept(Thing, kwargs={'form': 'Class'})
        bottom = Concept(Nothing, kwargs={'form': 'Class'})

        for i in onto.classes():
            i.is_a.append(T.owl)  # include T as most general class.

            temp_concept = Concept(i, kwargs={'form': 'Class'})  # wrap owl object into AtomicConcept.
            concepts[temp_concept.full_iri] = temp_concept

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

        self.concepts, self.T, self.Bottom = self.__build_concepts_mapping(onto)

        self.down_top_concept_hierarchy[self.T] = set()
        self.top_down_concept_hierarchy[self.T] = {_ for _ in self.concepts.values()}

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
        """
        TODO:
        """
        for leaf in self.concepts_to_leafs[concept]:
            yield leaf

    def negation(self, concept: Concept):
        """ Return a Concept object that is a negation of given concept."""
        return self.__concept_generator.negation(concept)

    def negation_from_iterables(self, s):
        """
        TODO:
        """
        for item in s:
            yield self.__concept_generator.negation(item)

    def get_direct_sub_concepts(self, concept: Concept):
        """
        Return concepts s.t. are explicit subClassOf a given concepts.
        """
        for v in self.top_down_direct_concept_hierarchy[concept]:
            yield v

    def get_direct_parents(self, concept: Concept):
        """
        TODO:
        """
        for direct_parent in self.down_top_direct_concept_hierarchy[concept]:
            yield direct_parent

    def most_general_existential_restrictions(self, concept: Concept):
        """
        TODO:
        # TODO: Obtain the definition of being most general.
        """
        properties = self.property_hierarchy.get_most_general_property()

        for prob in properties:
            existential = self.__concept_generator.existential_restriction(concept, prob)
            yield existential

    def most_general_universal_restriction(self, concept: Concept):
        """
        TODO:
        """
        properties = self.property_hierarchy.get_most_general_property()

        for prob in properties:
            universal = self.__concept_generator.universal_restriction(concept, prob)
            yield universal

    def union(self, conceptA, conceptB):
        """
        Return a Concept object that is equivalent to conceptA OR conceptB
        """
        return self.__concept_generator.union(conceptA, conceptB)

    def intersection(self, conceptA, conceptB):
        """Return a Concept object that is equivalent to conceptA OR conceptB"""
        return self.__concept_generator.intersection(conceptA, conceptB)

    def existential_restriction(self, concept: Concept, property_):
        """
        TODO:
        """
        assert isinstance(concept, Concept)

        direct_sub_concepts = [x for x in self.get_direct_sub_concepts(concept)]
        result = set()
        for sub_c in direct_sub_concepts:
            ref_ = self.__concept_generator.existential_restriction(sub_c, property_)
            result.add(ref_)
        return result

    def universal_restriction(self, concept: Concept, property_):
        """
        TODO:
        """
        assert isinstance(concept, Concept)

        direct_sub_concepts = (x for x in self.get_direct_sub_concepts(concept))
        result = set()
        for sub_c in direct_sub_concepts:
            ref_ = self.__concept_generator.universal_restriction(sub_c, property_)
            result.add(ref_)
        return result

    def num_concepts_generated(self):


        return len(self.__concept_generator.log_of_universal_restriction)+ len(self.__concept_generator.log_of_existential_restriction)+\
          len(self.__concept_generator.log_of_intersections)+ len(self.__concept_generator.log_of_unions)+\
          len(self.__concept_generator.log_of_negations)+len(self.concepts)

class PropertyHierarchy:
    """

    """

    def __init__(self, onto):
        self.all_properties = [i for i in onto.properties()]

        self.data_properties = [i for i in onto.data_properties()]

        self.object_properties = [i for i in onto.object_properties()]

    def get_most_general_property(self):
        """

        """
        for i in self.all_properties:
            yield i
