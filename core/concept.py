from abstracts.concept import AbstractConcept

class Concept(AbstractConcept):

    def __init__(self, concept, kwargs):
        super().__init__(concept, kwargs)
        self.__parse(kwargs)

    def __parse(self, kwargs):
        """

        :param kwargs:
        :return:
        """
        if not self.is_atomic:
            if self.form in ['ObjectSomeValuesFrom', 'ObjectAllValuesFrom']:
                self.role = kwargs['Role']  # property
                self.filler = kwargs['Filler']  # Concept
            elif self.form in ['ObjectUnionOf', 'ObjectIntersectionOf']:
                self.concept_a = kwargs['ConceptA']
                self.concept_b = kwargs['ConceptB']
            elif self.form == 'ObjectComplementOf':
                ''''''
            else:
                raise ValueError

    def __str__(self):
        return '{self.__repr__}\t{self.full_iri}'.format(self=self)

    def __len__(self):
        return self.length

    def __is_atomic(self):
        """

        :return:
        """
        if '∃' in self.str or '∀' in self.str:
            return False
        elif '⊔' in self.str or '⊓' in self.str or '¬' in self.str:
            return False
        return True

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

    def instances(self):
        return self.individuals
