from collections import defaultdict
from itertools import chain

from owlready2 import get_ontology, Ontology, Thing, Nothing, class_construct
from core.concept_generator import ConceptGenerator
from core.concept import Concept
from typing import Dict, Tuple, Set, Generator
from types import MappingProxyType
from core.util import parametrized_performance_debugger, get_full_iri


class KnowledgeBase:
    """Knowledge Base Class representing Tbox and Abox along with concept hierarchies"""

    def __init__(self, path):

        self._kb_path = path
        self.property_hierarchy = None
        self.onto = get_ontology(self._kb_path).load()
        self.name = self.onto.name
        self.concepts = dict()
        self.thing = None
        self.nothing = None
        self.top_down_concept_hierarchy = defaultdict(set)  # Next time thing about including this into Concepts.
        self.top_down_direct_concept_hierarchy = defaultdict(set)
        self.down_top_concept_hierarchy = defaultdict(set)
        self.down_top_direct_concept_hierarchy = defaultdict(set)
        self.concepts_to_leafs = defaultdict(set)
        self.parse()
        # self.concepts = MappingProxyType(self.concepts)
        self.__concept_generator = ConceptGenerator(concepts=self.concepts, T=self.thing, Bottom=self.nothing,
                                                    onto=self.onto)

    def retrieval_func(self, concept: Concept):

        if len(concept.instances) > 0:
            return

        print('Retrieve instances of {0}'.format(concept))

        assert len(concept.owl.equivalent_to) > 0
        assert isinstance(concept.owl.equivalent_to, list)

        for i in concept.owl.equivalent_to:

            if isinstance(i, class_construct.And):
                concept_A, concept_B = i.__getattribute__('Classes')[0], i.__getattribute__('Classes')[1]

                # Get object from concept dict
                full_iri_A = get_full_iri(concept_A)
                full_iri_B = get_full_iri(concept_B)

                try:
                    self.retrieval_func(self.concepts[full_iri_A])
                except:
                    print(full_iri_A)
                    print('Not found')

                    for k, v in self.concepts.items():
                        print(k)

                    exit(1)

                self.retrieval_func(self.concepts[full_iri_B])

                self.__concept_generator.type_enrichments(
                    self.concepts[full_iri_A].instances & self.concepts[full_iri_B].instances, concept.owl)

            elif isinstance(i, class_construct.Or):
                concept_A, concept_B = i.__getattribute__('Classes')[0], i.__getattribute__('Classes')[1]

                # Get object from concept dict
                full_iri_A = get_full_iri(concept_A)
                full_iri_B = get_full_iri(concept_B)

                try:
                    self.retrieval_func(self.concepts[full_iri_A])
                except:
                    print(full_iri_A)
                    print('Not found')

                    for k, v in self.concepts.items():
                        print(k, '\t', k == full_iri_A)

                    exit(1)
                self.retrieval_func(self.concepts[full_iri_B])

                self.__concept_generator.type_enrichments(
                    self.concepts[full_iri_A].instances | self.concepts[full_iri_B].instances, concept.owl)

            elif isinstance(i, class_construct.Not):
                concept_A, = i.__dict__('Class')
                # Get object from concept dict
                full_iri_A = get_full_iri(concept_A)
                self.retrieval_func(self.concepts[full_iri_A])
                self.__concept_generator.type_enrichments(self.T.instance - self.concepts[full_iri_A].intances,
                                                          concept.owl)


            else:

                # TODO assign instances by extractin info from equivalances.
                # print(new_concept.equivalent_to[0].__getattr__('Classes'))

                # print(type(new_concept.equivalent_to[0]))
                print(concept)
                print('FFFFUK')
                print()
                exit(1)

    def save(self, path):
        self.onto.save(path)

    def get_individuals(self) -> Set:
        return self.thing.instances()

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
        T.owl.equivalent_to.append(Thing)
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
        return self.__concept_generator.negation(concept)

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

    def get_direct_parents(self, concept: Concept):
        """ Return : { x | (concept subClassOf x)} """
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
        """Return {x | x ==(conceptA OR conceptA)}"""
        return self.__concept_generator.union(conceptA, conceptB)

    def intersection(self, conceptA, conceptB):
        """Return {x | x ==(conceptA AND conceptA)}"""
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
