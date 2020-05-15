import types
from time import sleep

from owlready2 import Not, AllDisjoint
from core.concept import Concept
import concurrent.futures


class ConceptGenerator:

    def __init__(self, concepts, T, Bottom, onto):
        self.concepts = concepts
        self.T = T
        self.Bottom = Bottom
        self.onto = onto

        self.log_of_intersections = dict()
        self.log_of_unions = dict()
        self.log_of_negations = dict()
        self.log_of_universal_restriction = dict()
        self.log_of_existential_restriction = dict()

        self.executor = concurrent.futures.ProcessPoolExecutor()

    @staticmethod
    def __concepts_sorter(A, B):
        if len(A) < len(B):
            return A, B
        if len(A) > len(B):
            return B, A

        args = [A, B]
        args.sort(key=lambda ce: ce.str)
        return args[0], args[1]

    @staticmethod
    def type_enrichments(instances, new_concept):
        for i in instances:
            i.is_a.append(new_concept)

    @staticmethod
    def type__restrictions_enrichments(exist, role, filler, c):
        if exist:
            # {(x,y) | (x,r,y) \in G}.
            # TODO use itertools.
            for x, y in role.get_relations():
                if y in filler.instances:
                    x.is_a.append(c)

        else:
            raise ValueError

    def negation(self, concept: Concept):

        if concept in self.log_of_negations:
            return self.log_of_negations[concept]

        if concept.is_atomic and not (concept.owl.name == 'Thing'):
            with self.onto:
                not_concept = types.new_class(name="¬{0}".format(concept.owl.name), bases=(self.T.owl,))
                not_concept.namespace = concept.owl.namespace
                AllDisjoint([not_concept, concept.owl])
                not_concept.is_a.append(self.T.owl)  # superclass
                not_concept.equivalent_to.append(Not(concept.owl))
                self.executor.submit(self.type_enrichments, (self.T.instances - concept.instances, not_concept))

            self.log_of_negations[concept] = Concept(concept=not_concept, kwargs={'form': 'ObjectComplementOf'})

            return self.log_of_negations[concept]
        elif concept.form == 'ObjectComplementOf':
            assert concept.str[0] == '¬'
            full_iri = concept.owl.namespace.base_iri + concept.owl.name[1:]
            return self.concepts[full_iri]
        elif concept.owl.name == 'Thing':
            self.log_of_negations[concept.full_iri] = self.Bottom
            self.log_of_negations[self.Bottom.full_iri] = concept
            return self.log_of_negations[concept.full_iri]
        else:
            raise ValueError

    def existential_restriction(self, concept: Concept, relation, base=None):

        if (concept, relation) in self.log_of_existential_restriction:
            return self.log_of_existential_restriction[(concept, relation)]

        if not base:
            base = self.T.owl

        with self.onto:
            new_concept = types.new_class(name="(∃ {0}.{1})".format(relation.name, concept.str), bases=(base,))
            new_concept.namespace = relation.namespace
            new_concept.is_a.append(relation.some(concept.owl))
            new_concept.equivalent_to.append(relation.some(concept.owl))

            relation.range.append(concept.owl)
            relation.domain.append(base)

            self.executor.submit(self.type__restrictions_enrichments, (True, relation, concept, new_concept))

            c = Concept(concept=new_concept,
                        kwargs={'form': 'ObjectSomeValuesFrom', 'Role': relation, 'Filler': concept})
            self.log_of_existential_restriction[(concept, relation)] = c

        return self.log_of_existential_restriction[(concept, relation)]

    def universal_restriction(self, concept: Concept, relation, base=None):
        """
        The universal quantifier defines a class as
        the set of all objects/individuals/instances
        for which the given role "only" attains values from the given class.

        which states that examiners must always be professors
        :param concept:
        :param relation:
        :param base:
        :return:
        """

        if (concept, relation) in self.log_of_universal_restriction:
            return self.log_of_universal_restriction[(concept, relation)]

        if not base:
            base = self.T.owl

        with self.onto:
            new_concept = types.new_class(name="(∀ {0}.{1})".format(relation.name, concept.str), bases=(base,))
            new_concept.namespace = relation.namespace
            new_concept.is_a.append(relation.only(base))
            new_concept.equivalent_to.append(relation.only(base))
            relation.range.append(concept.owl)
            # relation.domain.append(base)

            temp = set()
            # {(s,o) | (s,r,o) \in G}.
            for s, o in relation.get_relations():
                if not (o in concept.instances):
                    temp.add(o)
                    # s.is_a.append(new_concept)
            temp = self.T.instances - temp
            for i in temp:
                i.is_a.append(new_concept)

            self.log_of_universal_restriction[(concept, relation)] = Concept(concept=new_concept,
                                                                             kwargs={'form': 'ObjectAllValuesFrom',
                                                                                     'Role': relation,
                                                                                     'Filler': concept})
        return self.log_of_universal_restriction[(concept, relation)]

    def union(self, A: Concept, B: Concept, base=None):

        A, B = self.__concepts_sorter(A, B)

        # Crude workaround
        if A.str == 'Nothing':
            return B
        if B.str == 'Nothing':
            return A

        if (A, B) in self.log_of_unions:
            return self.log_of_unions[(A, B)]

        if not base:
            base = self.T.owl

        with self.onto:
            new_concept = types.new_class(name="({0} ⊔ {1})".format(A.str, B.str), bases=(base,))
            new_concept.namespace = A.owl.namespace
            new_concept.equivalent_to.append(A.owl | B.owl)

            self.executor.submit(self.type_enrichments, (A.instances | B.instances, new_concept))

            self.log_of_unions[(A, B)] = Concept(concept=new_concept,
                                                 kwargs={'form': 'ObjectUnionOf', 'ConceptA': A, 'ConceptB': B})
        return self.log_of_unions[(A, B)]

    def intersection(self, A: Concept, B: Concept, base=None):
        A, B = self.__concepts_sorter(A, B)

        # Crude workaround
        if A.str == 'Nothing':
            return B
        if B.str == 'Nothing':
            return A

        if (A, B) in self.log_of_intersections:
            return self.log_of_intersections[(A, B)]

        if not base:
            base = self.T.owl

        with self.onto:
            new_concept = types.new_class(name="({0}  ⊓  {1})".format(A.str, B.str), bases=(base,))
            new_concept.namespace = A.owl.namespace
            new_concept.equivalent_to.append(A.owl & B.owl)

            self.executor.submit(self.type_enrichments, (A.instances & B.instances, new_concept))

            self.log_of_intersections[(A, B)] = Concept(concept=new_concept,
                                                        kwargs={'form': 'ObjectIntersectionOf', 'ConceptA': A,
                                                                'ConceptB': B})

        return self.log_of_intersections[(A, B)]
