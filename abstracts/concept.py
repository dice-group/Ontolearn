from abc import ABC, abstractmethod

from owlready2 import ThingClass


class AbstractScorer(ABC):
    @abstractmethod
    def __init__(self, pos, neg):
        self.pos = pos
        self.neg = neg

    def apply(self, n):
        pass


class AbstractConcept(ABC):
    __slots__ = ['owl', 'full_iri', 'str', 'is_atomic',
                 'length', 'individuals', 'form', 'role', 'filler', 'concept_a', 'concept_b']

    @abstractmethod
    def __init__(self, concept: ThingClass, kwargs):
        assert isinstance(concept, ThingClass)
        assert kwargs['form'] in ['Class', 'ObjectIntersectionOf', 'ObjectUnionOf', 'ObjectComplementOf',
                                  'ObjectSomeValuesFrom', 'ObjectAllValuesFrom']

        self.owl = concept
        self.full_iri = concept.namespace.base_iri + concept.name
        self.str = concept.name
        self.form = kwargs['form']

        self.is_atomic = self.__is_atomic()  # TODO consider the necessity.
        self.length = self.__calculate_length()

        self.individuals = {jjj for jjj in concept.instances()} # TODO: maybe we do not need to store in memory?

    def __str__(self):
        return '{self.__repr__}\t{self.full_iri}'.format(self=self)

    def __len__(self):
        return self.length

    def __calculate_length(self):
        """
        The length of a concept is defined as
        the sum of the numbers of
            concept names, role names, quantifiers,and connective symbols occurring in the concept

        The length |A| of a concept CAis defined inductively:
        |A| = |\top| = |\bot| = 1
        |¬D| = |D| + 1
        |D \sqcap E| = |D \sqcup E| = 1 + |D| + |E|
        |∃r.D| = |∀r.D| = 2 + |D|
        :return:
        """
        num_of_exists = self.str.count("∃")
        num_of_for_all = self.str.count("∀")
        num_of_negation = self.str.count("¬")
        is_dot_here = self.str.count('.')

        num_of_operand_and_operator = len(self.str.split())
        count = num_of_negation + num_of_operand_and_operator + num_of_exists + is_dot_here + num_of_for_all
        return count

    def __is_atomic(self):
        """

        :return:
        """
        if '∃' in self.str or '∀' in self.str:
            return False
        elif '⊔' in self.str or '⊓' in self.str or '¬' in self.str:
            return False
        return True

    def instances(self):
        return self.individuals
