from collections import defaultdict
import types
from owlready2 import get_ontology, ThingClass,Ontology, Thing, Nothing, Not, AllDisjoint


class Concept:
    __slots__ = ['owl', 'full_iri', 'str', 'is_atomic',
                 'length', 'individuals', 'form', 'role', 'filler', 'concept_a', 'concept_b']

    def __init__(self, concept: ThingClass, kwargs):
        assert isinstance(concept, ThingClass)
        assert kwargs['form'] in ['Class', 'ObjectIntersectionOf', 'ObjectUnionOf', 'ObjectComplementOf',
                                  'ObjectSomeValuesFrom', 'ObjectAllValuesFrom']

        self.owl = concept
        self.full_iri = concept.namespace.base_iri + concept.name
        self.str = concept.name
        self.form = kwargs['form']
        self.is_atomic = self.__is_atomic()
        self.length = self.__calculate_length()
        self.individuals = {jjj for jjj in concept.instances()}

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


class ConceptGenerator:

    def __init__(self, concepts, T, Bottom, onto: Ontology):
        self.concepts = concepts
        self.T = T
        self.Bottom = Bottom
        self.onto = onto

        self.log_of_intersections = dict()
        self.log_of_unions = dict()
        self.log_of_negations = dict()
        self.log_of_universal_restriction = dict()
        self.log_of_existential_restriction = dict()

    @staticmethod
    def __concepts_sorter(A, B):
        if len(A) < len(B):
            return A, B
        if len(A) > len(B):
            return B, A

        args = [A, B]
        args.sort(key=lambda ce: ce.str)
        return args[0], args[1]

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

                for i in set(self.T.owl.instances()) - set(concept.owl.instances()):
                    i.is_a.append(not_concept)

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
            # {(x,y) | (x,r,y) \in G}.
            for x, y in relation.get_relations():  # WHY?
                if y in concept.instances():
                    x.is_a.append(new_concept)
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
                if not (o in concept.instances()):
                    temp.add(o)
                    # s.is_a.append(new_concept)

            temp = self.T.instances() - temp
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
        if A.str=='Nothing':
            return B
        if B.str=='Nothing':
            return A

        if (A, B) in self.log_of_unions:
            return self.log_of_unions[(A, B)]

        if not base:
            base = self.T.owl

        with self.onto:
            new_concept = types.new_class(name="({0} ⊔ {1})".format(A.str, B.str), bases=(base,))
            new_concept.namespace = A.owl.namespace
            try:
                new_concept.equivalent_to.append(A.owl | B.owl)
            except:
                print(A)
                print(B)
                exit(1)


            instances = set(A.instances()).union(set(B.instances()))
            for i in instances:
                i.is_a.append(new_concept)

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
            try:
                new_concept.equivalent_to.append(A.owl & B.owl)
            except:
                print(A)
                print(B)
                exit(1)
            for i in (set(A.instances()) & set(B.instances())):
                i.is_a.append(new_concept)

            self.log_of_intersections[(A, B)] = Concept(concept=new_concept,
                                                        kwargs={'form': 'ObjectIntersectionOf', 'ConceptA': A,
                                                                'ConceptB': B})
        return self.log_of_intersections[(A, B)]


class KnowledgeBase:
    def __init__(self, *, path):

        self._kb_path = path
        self.__properties = None
        self.onto = get_ontology(self._kb_path).load()
        self.name=self.onto.name
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

    def get_individuals(self):

        return self.T.instances()

    @staticmethod
    def __build_concepts_mapping(onto: Ontology):
        """

        :param onto:
        :return:
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

    def __build_hierarchy(self, onto):

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
        self.__build_hierarchy(self.onto)
        self.__properties = PropertyHierarchy(self.onto)

    def concept_from_string(self, str_representation: str):
        """
        Given str, we should create complex concept. Say Forall r.Concept
        (1) Check whether () exist.
        (2) Detect the position of atomic concepts.
        (3) Do it by learning it :D

        (1)
        :param str_representation:
        :return:
        """

    def get_leaf_concepts(self, concept: Concept):
        for leaf in self.concepts_to_leafs[concept]:
            yield leaf

    def negation(self, concept: Concept):
        return self.__concept_generator.negation(concept)

    def negation_from_iterables(self, s):
        for item in s:
            yield self.__concept_generator.negation(item)

    def get_direct_sub_concepts(self, concept: Concept):
        return self.top_down_direct_concept_hierarchy[concept]

    def get_direct_parents(self, concept: Concept):
        for direct_parent in self.down_top_direct_concept_hierarchy[concept]:
            yield direct_parent

    def most_general_existential_restrictions(self, concept: Concept):
        properties = self.__properties.get_most_general_property()  # TODO: Obtain the definition of being most general.

        for prob in properties:
            existential = self.__concept_generator.existential_restriction(concept, prob)
            yield existential

    def most_general_universal_restriction(self, concept: Concept):
        properties = self.__properties.get_most_general_property()

        for prob in properties:
            universal = self.__concept_generator.universal_restriction(concept, prob)
            yield universal

    def union(self, conceptA, conceptB):
        return self.__concept_generator.union(conceptA, conceptB)

    def intersection(self, conceptA, conceptB):
        return self.__concept_generator.intersection(conceptA, conceptB)

    def existential_restriction(self, concept: Concept, property_):
        assert isinstance(concept, Concept)

        direct_sub_concepts = [x for x in self.get_direct_sub_concepts(concept)]
        result = set()
        for sub_c in direct_sub_concepts:
            ref_ = self.__concept_generator.existential_restriction(sub_c, property_)
            result.add(ref_)
        return result

    def universal_restriction(self, concept: Concept, property_):
        assert isinstance(concept, Concept)

        direct_sub_concepts = [x for x in self.get_direct_sub_concepts(concept)]
        result = set()
        for sub_c in direct_sub_concepts:
            ref_ = self.__concept_generator.universal_restriction(sub_c, property_)
            result.add(ref_)
        return result


class PropertyHierarchy:
    def __init__(self, onto):
        self.properties = [i for i in onto.properties()]

        self.data_properties = [i for i in onto.data_properties()]

        self.object_properties = [i for i in onto.object_properties()]

    def get_most_general_property(self):
        for i in self.properties:
            yield i
