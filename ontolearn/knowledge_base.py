from collections import defaultdict
from owlready2 import Ontology, get_ontology
import owlready2
from .concept_generator import ConceptGenerator
from .concept import Concept
from typing import Dict, Tuple, Set, Generator, Iterable
from .util import parametrized_performance_debugger, get_full_iri
import warnings

warnings.filterwarnings("ignore")


class KnowledgeBase:
    """Knowledge Base (KB) Class representing Tbox and Abox along with concept hierarchies"""

    @property
    def path(self) -> str:
        """
        Path to the file that contains the KB.
        """
        return self._path

    @property
    def onto(self) -> Ontology:
        """
        The ontology of the KB
        """
        return self._onto

    @property
    def name(self) -> str:
        """
        The name of the ontology of the knowledge base.
        TODO: necessary?
        """
        return self._name

    @property
    def concepts(self) -> Dict[str, Concept]:
        """
        A mapping from resource IRI to concepts for all concepts in the KB
        :return:
        """
        return self._concepts

    @property
    def thing(self) -> Concept:
        """
        The top concept in the KB
        TODO: is that right? rename it to top_concept?
        """
        return self._thing

    @property
    def nothing(self) -> Concept:
        """
        The bottom concept in the KB.
        TODO: is that right? rename it to bottom_concept?
        """
        return self._nothing

    @property
    def top_down_concept_hierarchy(self) -> Dict[Concept, Set[Concept]]:
        """
        top down hierarchy of concepts that maps a concept A from the KB to a set of sub concepts B such that ∀b ∊ B: b ⊑ A.
        """
        return self._top_down_concept_hierarchy

    @property
    def top_down_direct_concept_hierarchy(self) -> Dict[Concept, Set[Concept]]:
        """
        top down hierarchy of concepts that maps a concept A from the KB to a set of sub concepts B such that ∀b ∊ B: b ⊑ A if b is a most specific sub concept.
        """
        return self._top_down_direct_concept_hierarchy

    @property
    def down_top_concept_hierarchy(self) -> Dict[Concept, Set[Concept]]:
        """
        bottom up down hierarchy of concepts that maps a concept A from the KB to a set of super concepts B such that ∀b ∊ B: A ⊑ b.
        """
        return self._down_top_concept_hierarchy

    @property
    def down_top_direct_concept_hierarchy(self) -> Dict[Concept, Set[Concept]]:
        """
        bottom up down hierarchy of concepts that maps a concept A from the KB to a set of super concepts B such that ∀b ∊ B: A ⊑ b is a most specific super concept.
        """
        return self._down_top_direct_concept_hierarchy

    @property
    def concepts_to_leafs(self) -> Dict[Concept, Set[Concept]]:
        """
        maps from concept A to a set of concepts B = { x | (x subClassOf A) AND not exist y: y subClassOf x )}
        """
        return self._concepts_to_leafs

    @property
    def property_hierarchy(self) -> 'PropertyHierarchy':
        """
        Hierarchy of concepts
        """
        return self._property_hierarchy

    @property
    def min_size_of_concept(self) -> int:
        """
        minimal size of concepts that are inferred
        """
        return self._min_size_of_concept

    @min_size_of_concept.setter
    def min_size_of_concept(self, n: int):
        self._min_size_of_concept = n

    @property
    def max_size_of_concept(self) -> int:
        """
        minimal size of concepts that are inferred
        """
        return self._max_size_of_concept

    @max_size_of_concept.setter
    def max_size_of_concept(self, n: int):
        self._max_size_of_concept = n

    @property
    def all_individuals(self) -> Set[owlready2.entity.ThingClass]:
        """
        all individuals in the KB
        """
        return self.thing.instances

    def __init__(self, path, min_size_of_concept=0, max_size_of_concept=None):
        """
        :param path: path to the RDF file containing the knowledge base
        :param min_size_of_concept: the minimum size of concepts that are inferred
        :param max_size_of_concept: the maximum size of concepts that are inferred
        """
        self._path: str = path
        self._onto: Ontology = get_ontology(self.path).load(reload=True)
        self._name: str = self.onto.name
        self._concepts: Dict[str, Concept] = dict()

        self._top_down_concept_hierarchy: Dict[Concept, Set[Concept]] = defaultdict(
            set)  # Next time thing about including this into Concepts.
        self._top_down_direct_concept_hierarchy: Dict[Concept, Set[Concept]] = defaultdict(set)
        self._down_top_concept_hierarchy: Dict[Concept, Set[Concept]] = defaultdict(set)
        self._down_top_direct_concept_hierarchy: Dict[Concept, Set[Concept]] = defaultdict(set)
        self._concepts_to_leafs: Dict[Concept, Set[Concept]] = defaultdict(set)
        self._property_hierarchy: PropertyHierarchy
        self._thing: Concept
        self._nothing: Concept
        self.parse()

        self._min_size_of_concept: int = min_size_of_concept

        self._max_size_of_concept = len(self.thing.instances) if max_size_of_concept is None else max_size_of_concept
        self.__concept_generator: ConceptGenerator = ConceptGenerator(concepts=self.concepts,
                                                                      T=self.thing,
                                                                      Bottom=self.nothing,
                                                                      onto=self.onto,
                                                                      min_size_of_concept=self.min_size_of_concept,
                                                                      max_size_of_concept=self.max_size_of_concept)

    @staticmethod
    def apply_type_enrichment_from_iterable(concepts: Iterable[Concept]):
        """
        TODO: method docu and code seem not to match
        Extend ABOX by
        (1) Obtaining all instances of selected concepts.
        (2) For all instances in (1) include type information into ABOX.
        @param concepts:
        """
        for c in concepts:
            for ind in c.owl.instances():
                ind.is_a.append(c.owl)

    @staticmethod
    def apply_type_enrichment(concept: Concept):
        """
        TODO: what exactly does it do?
        """
        for ind in concept.instances:
            ind.is_a.append(concept.owl)

    def save(self, path, rdf_format='ntriples'):
        """
        serializes the knowledge base to RDF
        :param path: file path to store the file
        :param rdf_format: available formats are 'rdfxml' and 'ntriples'
        """
        self.onto.save(file=path, format=rdf_format)

    @staticmethod
    def is_atomic(c: owlready2.entity.ThingClass) -> bool:
        """
        check wheather a concept is atomic
        """
        assert isinstance(c, owlready2.entity.ThingClass)
        if '¬' in c.name and not (' ' in c.name):
            return False
        elif ' ' in c.name or '∃' in c.name or '∀' in c.name:
            return False
        else:
            return True

    # TODO: remove this code?
    """
    @staticmethod
    def __parse_complex(complex_concepts: Iterable[owlready2.entity.ThingClass], concept_mapping: Dict):
        assert isinstance(complex_concepts, Iterable)

        # TODO get all negations
        negations = (c for c in complex_concepts
                     if isinstance(c.is_a[1], owlready2.class_construct.Not))

        restictions = (c for c in complex_concepts
                       if isinstance(c.is_a[1], owlready2.class_construct.Restriction))

        conj_and_disjunctions = (c for c in complex_concepts
                                 if isinstance(c.is_a[1], owlready2.class_construct.And) or isinstance(c.is_a[1],
                                                                                                       owlready2.class_construct.Or))
        import re

        for c in complex_concepts:
            print(c)

        exit(1)
        for c in negations:
            owl_obj = c.is_a[1]
            # print(c)
            # print(c.name)
            # print(owl_obj.Class)
            # print(concept_mapping[get_full_iri(owl_obj.Class)])
            temp_concept = Concept(c, kwargs={'form': 'ObjectComplementOf',
                                              'root': concept_mapping[get_full_iri(owl_obj.Class)]})
            concept_mapping[temp_concept.full_iri] = temp_concept

        for c in complex_concepts:
            if isinstance(c.is_a[1], owlready2.class_construct.Not):
                continue

            # components = re.findall(r"\((.*?)\)", c.name)
            # assert len(components)==1
            # print(components)
            print(c.name)
            # text_in_list=c.name.split()

            #            print(text_in_list)
            continue
            if len(text_in_list) == 3:
                ## then this means that we have ['(A', '⊓', 'B)'] or  ['(A', 'OR', 'B)']
                # where A or B can only be a negation or atomic concept.
                owl_ready_concept = c.is_a[1]
                if isinstance(owl_ready_concept, owlready2.class_construct.And):
                    pass
                elif isinstance(owl_ready_concept, owlready2.class_construct.Or):
                    pass
                else:
                    print(c)
                    print(text_in_list)
                    raise ValueError

                pass

        exit(1)
        for c in complex_concepts:
            assert len(c.is_a) == 2  # [owl.Thing, XXXX]
            owl_ready_concept = c.is_a[1]

            if isinstance(owl_ready_concept, owlready2.class_construct.Restriction):
                print(owl_ready_concept)
                rel = owl_ready_concept.property
                value = owl_ready_concept.value
                type_of_restriction = owl_ready_concept.type  # TODO: For the love of goo, type of restriction is mapped to integer ?!!

                if type_of_restriction == 24:  # Exists
                    pass  # Concept(c,kwargs=)
                elif type_of_restriction == 25:  # FORall
                    pass
                else:
                    raise ValueError



            elif isinstance(owl_ready_concept, owlready2.class_construct.And):
                pass
            elif isinstance(owl_ready_concept, owlready2.class_construct.Or):
                pass
            elif isinstance(owl_ready_concept, owlready2.class_construct.Not):
                pass
            else:
                raise ValueError

        exit(1)
        print(c)

        owl_type = c.is_a[1]
        print(owl_type)
        print(type(owl_type))

        print(owl_type.property)
        print(owl_type.value)

        exit(1)

        return True
    """

    def __build_concepts_mapping(self) -> Tuple[Dict[str, Concept], Concept, Concept]:
        """
        Construct a mapping from full_iri to corresponding Concept objects.

        concept.namespace.base_iri + concept.name
        mappings from concepts uri to concept objects
            1) full_iri:= owlready2.ThingClass.namespace.base_iri + owlready2.ThingClass.name
            2) Concept:
        """
        individuals: Set[Concept] = set()

        self._concepts: Dict[str, Concept] = dict()

        self._thing: Concept = Concept(owlready2.Thing, form='Class')
        self._nothing: Concept = Concept(owlready2.Nothing, form='Class')

        for thing_class in self.onto.classes():
            temp_concept: Concept = Concept(thing_class, form='Class')  # Regardless of concept length
            self._concepts[temp_concept.full_iri] = temp_concept

            individuals.update(temp_concept.instances)
        # check that the _thing concept contains individuals
        try:
            assert self._thing.instances
        except:
            print('owlready2.Thing does not contains any individuals.\t')
            self._thing.instances = individuals
        # add _thing and _nothing concepts
        self._concepts[self._thing.full_iri] = self._thing
        self._concepts[self._nothing.full_iri] = self._nothing
        return self._concepts, self._thing, self._nothing

    def __build_hierarchy(self) -> None:
        """
        Builds concept sub and super classes hierarchies.

        1) self.top_down_concept_hierarchy is a mapping from Concept objects to a set of Concept objects that are
        direct subclasses of given Concept object.

        2) self.down_top_concept_hierarchy is a mapping from Concept objects to set of Concept objects that are
        direct superclasses of given Concept object.

        """

        self.down_top_concept_hierarchy[self.thing] = set()
        self.top_down_concept_hierarchy[self.thing] = set(self.concepts.values())

        for concept_A in self.concepts.values():  # second loop over concepts in the execution,

            for desc in concept_A.owl.descendants(include_self=False):

                wrapped_desc: Concept = self.concepts[desc.namespace.base_iri + desc.name]

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
        self.__build_concepts_mapping()
        self.__build_hierarchy()
        self._property_hierarchy = PropertyHierarchy(self.onto)

    def get_leaf_concepts(self, concept: Concept):
        """ Return : { x | (x subClassOf concept) AND not exist y: y subClassOf x )} """
        for leaf in self.concepts_to_leafs[concept]:
            yield leaf

    def negation(self, concept: Concept):
        """ Return a Concept object that is a negation of given concept."""
        yield self.__concept_generator.negation(concept)

    @parametrized_performance_debugger()
    def negation_from_iterables(self, s: Generator[Concept, None, None]):
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
            print(direct_parent)
            exit(1)
            yield direct_parent

    def most_general_existential_restrictions(self, concept: Concept):
        """
        R is the most general property
        ∃R.C = {x ∊ Δ | ∃y ∊ Δ: (x, y) ∊ R^I AND y ∊ C^I }
        """
        for prop in self.property_hierarchy.get_most_general_property():
            yield self.__concept_generator.existential_restriction(concept, prop)

    def most_general_universal_restriction(self, concept: Concept) -> Concept:
        """
        TODO: might be wrong. see PropertyHierarchy TODO
        R is the most general property
        ∀R.C => extension => {x ∊ Δ | ∃y ∊ Δ : (x, y) ∊ R^I ⇒ y ∊ C^I }
        """
        assert isinstance(concept, Concept)
        for prop in self.property_hierarchy.get_most_general_property():
            yield self.__concept_generator.universal_restriction(concept, prop)

    def union(self, conceptA: Concept, conceptB: Concept) -> Concept:
        """Return a concept c == (conceptA OR conceptA)"""
        assert isinstance(conceptA, Concept)
        assert isinstance(conceptB, Concept)
        return self.__concept_generator.union(conceptA, conceptB)

    def intersection(self, conceptA: Concept, conceptB: Concept) -> Concept:
        """Return a concept c == (conceptA AND conceptA)"""
        assert isinstance(conceptA, Concept)
        assert isinstance(conceptB, Concept)
        return self.__concept_generator.intersection(conceptA, conceptB)

    def existential_restriction(self, concept: Concept, property_) -> Concept:
        """Return a concept c == (Exist R.C)"""
        assert isinstance(concept, Concept)
        return self.__concept_generator.existential_restriction(concept, property_)

    def universal_restriction(self, concept: Concept, property_) -> Concept:
        """Return a concept c == (Forall R.C)"""
        assert isinstance(concept, Concept)
        return self.__concept_generator.universal_restriction(concept, property_)

    def num_concepts_generated(self):
        """
        total number of concepts generated
        """
        # TODO: review this after reviewing concept generator
        return len(self.__concept_generator.log_of_universal_restriction) + \
               len(self.__concept_generator.log_of_existential_restriction) + \
               len(self.__concept_generator.log_of_intersections) + \
               len(self.__concept_generator.log_of_unions) + \
               len(self.__concept_generator.log_of_negations) + \
               len(self.concepts)

    def get_all_concepts(self):
        """
        get all concepts of the KB
        """
        return self.concepts.values()
        # TODO: remove commented code
        # set(chain(self.concepts.values()))
        # self.__concept_generator.log_of_universal_restriction.values(),
        # self.__concept_generator.log_of_negations.values(),
        # self.__concept_generator.log_of_intersections.values(),
        # self.__concept_generator.log_of_universal_restriction.values(),
        # self.__concept_generator.log_of_existential_restriction.values()))


class PropertyHierarchy:
    # TODO: is this class a stub? it doesn't seem to be used anywhere productively?

    def __init__(self, onto):
        self.all_properties = [i for i in onto.properties()]

        self.data_properties = [i for i in onto.data_properties()]

        self.object_properties = [i for i in onto.object_properties()]

    def get_most_general_property(self):
        # TODO: what is this good for? is it actually implemented correctly?
        # I would assume there should only be one most general property, right?
        for i in self.all_properties:
            yield i
