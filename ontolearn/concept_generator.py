import types
from owlready2 import Not, AllDisjoint
from .concept import Concept
import concurrent.futures
import owlready2
from typing import Dict
from .static_funcs import concepts_sorter
from .util import get_full_iri


class ConceptGenerator:
    def __init__(self, concepts: Dict, thing: Concept, nothing: Concept, onto):

        self.concepts = concepts
        self.thing = thing
        self.nothing = nothing
        self.onto = onto
        # TODO Have a data structure that can only be written once.
        self.log_of_intersections = dict()
        self.log_of_unions = dict()
        self.log_of_negations = dict()
        self.log_of_universal_restriction = dict()
        self.log_of_existential_restriction = dict()

    def clean(self):
        """

        @return:
        """

        # @todo temporray, next embedings will be stored in nodes
        for k, v in self.log_of_intersections.items():
            a, b = k
            a.embeddings = None
            b.embeddings = None
            v.embeddings = None

        for k, v in self.log_of_unions.items():
            a, b = k
            a.embeddings = None
            b.embeddings = None
            v.embeddings = None

        for k, v in self.log_of_negations.items():
            a = k
            a.embeddings = None
            v.embeddings = None

        for k, v in self.log_of_universal_restriction.items():
            a, b = k
            a.embeddings = None
            v.embeddings = None

        for k, v in self.log_of_existential_restriction.items():
            a, b = k
            a.embeddings = None
            v.embeddings = None

    def get_instances_for_restrictions(self, exist, role, filler):
        if exist:
            temp = set()
            # {(x,y) | (x,r,y) \in G}.
            for x, y in role.get_relations():
                if y in filler.instances:
                    temp.add(x)
            return temp
        else:
            temp = set()
            # {(s,o) | (s,r,o) \in G}.
            for s, o in role.get_relations():
                if not (o in filler.instances):
                    temp.add(o)
            return self.thing.instances - temp

    def negation(self, concept: Concept) -> Concept:
        """
        ¬C = \Delta^I \ C.
        @param concept: an instance of Concept class
        @return: ¬C: an instance of Concept class
        """
        if concept in self.log_of_negations:
            return self.log_of_negations[concept]
        if not (concept.owl.name == 'Thing'):
            possible_instances_ = self.thing.instances - concept.instances

            with self.onto:
                not_concept = types.new_class(name="¬{0}".format(concept.owl.name), bases=(self.thing.owl,))
                AllDisjoint([not_concept, concept.owl])
                not_concept.equivalent_to.append(Not(concept.owl))

                c = Concept(concept=not_concept, kwargs={'form': 'ObjectComplementOf', 'root': concept})
                c.instances = possible_instances_  # self.T.instances - concept.instances

                self.log_of_negations[concept] = c
                self.concepts[c.full_iri] = c

            return self.log_of_negations[concept]
        elif concept.form == 'ObjectComplementOf':
            assert concept.str[0] == '¬'
            full_iri = concept.owl.namespace.base_iri + concept.owl.name[1:]
            return self.concepts[full_iri]
        elif concept.owl.name == 'Thing':
            self.log_of_negations[concept.full_iri] = self.nothing
            self.log_of_negations[self.nothing.full_iri] = concept
            return self.log_of_negations[concept.full_iri]
        else:
            raise ValueError

    def existential_restriction(self, concept: Concept, relation, base=None) -> Concept:
        """
        ∃R.C =>  {x \in \Delta | ∃y \in \Delta : (x, y) \in R^I AND y ∈ C^I }

        @param concept: an instance of Concept
        @param relation: an isntance of owlready2.prop.ObjectPropertyClass'
        @param base:
        @return:
        """

        if (concept, relation) in self.log_of_existential_restriction:
            return self.log_of_existential_restriction[(concept, relation)]

        if not base:
            base = self.thing.owl

        possible_instances_ = self.get_instances_for_restrictions(True, relation, concept)
        with self.onto:
            new_concept = types.new_class(name="(∃{0}.{1})".format(relation.name, concept.str), bases=(base,))
            new_concept.equivalent_to.append(relation.some(concept.owl))

            c = Concept(concept=new_concept,
                        kwargs={'form': 'ObjectSomeValuesFrom', 'Role': relation, 'Filler': concept})

            for i in possible_instances_:
                assert type(i) is not str
            c.instances = possible_instances_  # self.get_instances_for_restrictions(True, relation, concept)

            self.log_of_existential_restriction[(concept, relation)] = c
            self.concepts[c.full_iri] = c

        return self.log_of_existential_restriction[(concept, relation)]

    def universal_restriction(self, concept: Concept, relation, base=None) -> Concept:
        """

        ∀R.C => extension => {x \in \Delta | ∃y \in \Delta : (x, y) ∈ \in R^I \implies y ∈ C^I }

        Brief explanation:
                    The universal quantifier defines a class as
                    *   The set of all instances for which the given role "only" attains values from the given class.

        :param concept:
        :param relation:
        :param base:
        :return:
        """

        if (concept, relation) in self.log_of_universal_restriction:
            return self.log_of_universal_restriction[(concept, relation)]

        if not base:
            base = self.thing.owl

        possible_instances_ = self.get_instances_for_restrictions(False, relation, concept)
        with self.onto:
            new_concept = types.new_class(name="(∀{0}.{1})".format(relation.name, concept.str), bases=(base,))
            new_concept.equivalent_to.append(relation.only(base))
            c = Concept(concept=new_concept,
                        kwargs={'form': 'ObjectAllValuesFrom', 'Role': relation, 'Filler': concept})

            for i in possible_instances_:
                assert type(i) is not str
            c.instances = possible_instances_  # self.get_instances_for_restrictions(False, relation, concept)

            self.log_of_universal_restriction[(concept, relation)] = c

            self.concepts[c.full_iri] = c

        return self.log_of_universal_restriction[(concept, relation)]

    def union(self, A: Concept, B: Concept, base=None):

        A, B = concepts_sorter(A, B)
        if A.str == B.str:
            return A
        # Crude workaround
        if A.str == 'Nothing':
            return B

        if B.str == 'Nothing':
            return A

        if (A, B) in self.log_of_unions:
            return self.log_of_unions[(A, B)]

        if not base:
            base = self.thing.owl

        possible_instances_ = A.instances | B.instances
        with self.onto:
            new_concept = types.new_class(name="({0} ⊔ {1})".format(A.str, B.str), bases=(base,))
            new_concept.equivalent_to.append(A.owl | B.owl)
            c = Concept(concept=new_concept, kwargs={'form': 'ObjectUnionOf', 'ConceptA': A, 'ConceptB': B})
            for i in possible_instances_:
                assert type(i) is not str

            c.instances = possible_instances_  # A.instances | B.instances
            self.log_of_unions[(A, B)] = c

            self.concepts[c.full_iri] = c
        return self.log_of_unions[(A, B)]

    def intersection(self, A: Concept, B: Concept, base=None) -> Concept:
        A, B = concepts_sorter(A, B)
        if A.str == B.str:
            return A

        if (A, B) in self.log_of_intersections:
            return self.log_of_intersections[(A, B)]

        # Crude workaround
        if A.str == 'Nothing' or B.str == 'Nothing':
            self.log_of_intersections[(A, B)] = self.nothing
            return self.log_of_intersections[(A, B)]

        if not base:
            base = self.thing.owl

        possible_instances_ = A.instances & B.instances

        with self.onto:
            new_concept = types.new_class(name="({0}  ⊓  {1})".format(A.str, B.str), bases=(base,))
            new_concept.equivalent_to.append(A.owl & B.owl)
            c = Concept(concept=new_concept, kwargs={'form': 'ObjectIntersectionOf', 'ConceptA': A, 'ConceptB': B})
            for i in possible_instances_:
                assert type(i) is not str
            c.instances = possible_instances_  # A.instances & B.instances
            self.log_of_intersections[(A, B)] = c
            self.concepts[c.full_iri] = c

        return self.log_of_intersections[(A, B)]
