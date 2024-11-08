""" Test the default pipeline for structured machine learning"""
import json
from owlapy.class_expression import OWLClass
from owlapy.iri import IRI
from owlapy.owl_individual import OWLNamedIndividual
from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.concept_learner import CELOE
from ontolearn.learning_problem import PosNegLPStandard
from ontolearn.utils import compute_f1_score
from owlapy.render import DLSyntaxObjectRenderer
import json
import os
from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.learners import OCEL
from ontolearn.learning_problem import PosNegLPStandard
from ontolearn.utils import setup_logging
from owlapy.owl_individual import OWLNamedIndividual, IRI
from owlapy.class_expression import OWLClass

PATH_FAMILY = 'KGs/Family/family-benchmark_rich_background.owl'
PATH_MUTAGENESIS = 'KGs/Mutagenesis/mutagenesis.owl'
PATH_DATA_FATHER = 'KGs/Family/father.owl'

with open('examples/synthetic_problems.json') as json_file:
    settings = json.load(json_file)


class TestCeloe:

    def test_ocel_example(self):

        kb = KnowledgeBase(path=PATH_FAMILY)

        for str_target_concept, examples in settings['problems'].items():
            p = set(examples['positive_examples'])
            n = set(examples['negative_examples'])
            print('Target concept: ', str_target_concept)
            concepts_to_ignore = set()
            # lets inject more background info
            if str_target_concept in ['Granddaughter', 'Aunt', 'Sister']:
                NS = 'http://www.benchmark.org/family#'
                concepts_to_ignore = {
                    OWLClass(IRI(NS, 'Brother')),
                    OWLClass(IRI(NS, 'Sister')),
                    OWLClass(IRI(NS, 'Daughter')),
                    OWLClass(IRI(NS, 'Mother')),
                    OWLClass(IRI(NS, 'Grandmother')),
                    OWLClass(IRI(NS, 'Father')),
                    OWLClass(IRI(NS, 'Grandparent')),
                    OWLClass(IRI(NS, 'PersonWithASibling')),
                    OWLClass(IRI(NS, 'Granddaughter')),
                    OWLClass(IRI(NS, 'Son')),
                    OWLClass(IRI(NS, 'Child')),
                    OWLClass(IRI(NS, 'Grandson')),
                    OWLClass(IRI(NS, 'Grandfather')),
                    OWLClass(IRI(NS, 'Grandchild')),
                    OWLClass(IRI(NS, 'Parent')),
                }
                target_kb = kb.ignore_and_copy(ignored_classes=concepts_to_ignore)
            else:
                target_kb = kb

            typed_pos = set(map(OWLNamedIndividual, map(IRI.create, p)))
            typed_neg = set(map(OWLNamedIndividual, map(IRI.create, n)))
            lp = PosNegLPStandard(pos=typed_pos, neg=typed_neg)

            model = OCEL(knowledge_base=target_kb,
                         max_runtime=3,
                         max_num_of_concepts_tested=10_000_000_000,
                         iter_bound=10_000_000_000)
            model.fit(lp)

            model.save_best_hypothesis(n=3, path='Predictions_{0}'.format(str_target_concept))
            hypotheses = model.best_hypotheses(n=3)
            [print(_) for _ in hypotheses]

    def test_celoe(self):
        kb = KnowledgeBase(path=PATH_FAMILY)

        exp_qualities = {'Aunt': .80392, 'Brother': 1.0,
                         'Cousin': .68063, 'Granddaughter': 1.0,
                         'Uncle': .88372, 'Grandgrandfather': 0.94444}
        tested = dict()
        found_qualities = dict()
        for str_target_concept, examples in settings['problems'].items():
            typed_pos = set(map(OWLNamedIndividual, map(IRI.create, set(examples['positive_examples']))))
            typed_neg = set(map(OWLNamedIndividual, map(IRI.create, set(examples['negative_examples']))))
            lp = PosNegLPStandard(pos=typed_pos, neg=typed_neg)
            print('Target concept: ', str_target_concept)
            concepts_to_ignore = set()
            # lets inject more background info
            if str_target_concept in ['Granddaughter', 'Aunt', 'Sister']:
                # Use URI
                concepts_to_ignore.update(
                    map(OWLClass, map(IRI.create, {
                        'http://www.benchmark.org/family#Brother',
                        'http://www.benchmark.org/family#Father',
                        'http://www.benchmark.org/family#Grandparent'})))

            target_kb = kb.ignore_and_copy(ignored_classes=concepts_to_ignore)
            model = CELOE(knowledge_base=target_kb, max_runtime=60, max_num_of_concepts_tested=3000)

            returned_val = model.fit(learning_problem=lp)
            assert returned_val==model, "fit should return its self"
            hypotheses = model.best_hypotheses(n=3)
            f1_qualities=[compute_f1_score(individuals=frozenset({i for i in kb.individuals(owl)}),pos=lp.pos,neg=lp.neg)  for owl in hypotheses]
            tested[str_target_concept] = model.number_of_tested_concepts
            found_qualities[str_target_concept] = f1_qualities[0]
            assert f1_qualities[0]>=exp_qualities[str_target_concept]
            assert f1_qualities[0]>= f1_qualities[1]
            assert f1_qualities[1]>= f1_qualities[2]


    def test_celoe_mutagenesis(self):
        kb = KnowledgeBase(path=PATH_MUTAGENESIS)

        namespace_ = 'http://dl-learner.org/mutagenesis#'
        pos_inds = ['d190', 'd191', 'd194', 'd197', 'e1', 'e2', 'e27', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6']
        pos = {OWLNamedIndividual(IRI.create(namespace_, ind)) for ind in pos_inds}
        neg_inds = ['d189', 'd192', 'd193', 'd195', 'd196', 'e10', 'e11', 'e12', 'e13', 'e14', 'e15', 'e16',
                    'e17', 'e18', 'e19', 'e20', 'e21', 'e22', 'e23', 'e24', 'e25', 'e26', 'e3', 'e4', 'e5',
                    'e6', 'e7', 'e8', 'e9']
        neg = {OWLNamedIndividual(IRI.create(namespace_, ind)) for ind in neg_inds}

        lp = PosNegLPStandard(pos=pos, neg=neg)
        model = CELOE(knowledge_base=kb, max_runtime=60, max_num_of_concepts_tested=3000)
        returned_model = model.fit(learning_problem=lp)
        best_pred = returned_model.best_hypotheses(n=1)

        assert compute_f1_score(individuals=frozenset({i for i in kb.individuals(best_pred)}), pos=lp.pos, neg=lp.neg)>=0.96

        r = DLSyntaxObjectRenderer()
        assert r.render(best_pred)== '∃ act.xsd:double[≥ 0.325]'

    def test_celoe_father(self):
        kb = KnowledgeBase(path=PATH_DATA_FATHER)

        examples = {
            'positive_examples': [
                OWLNamedIndividual(IRI.create("http://example.com/father#stefan")),
                OWLNamedIndividual(IRI.create("http://example.com/father#markus")),
                OWLNamedIndividual(IRI.create("http://example.com/father#martin"))],
            'negative_examples': [
                OWLNamedIndividual(IRI.create("http://example.com/father#heinz")),
                OWLNamedIndividual(IRI.create("http://example.com/father#anna")),
                OWLNamedIndividual(IRI.create("http://example.com/father#michelle"))]
        }

        p = set(examples['positive_examples'])
        n = set(examples['negative_examples'])

        lp = PosNegLPStandard(pos=p, neg=n)
        model = CELOE(knowledge_base=kb)

        model.fit(learning_problem=lp)
        best_pred = model.best_hypotheses(n=1)

        assert compute_f1_score(individuals=frozenset({i for i in kb.individuals(best_pred)}), pos=lp.pos, neg=lp.neg)==1.0
        r = DLSyntaxObjectRenderer()
        assert r.render(best_pred)=='(¬female) ⊓ (∃ hasChild.⊤)'

    def test_multiple_fits(self):
        kb = KnowledgeBase(path=PATH_FAMILY)

        pos_aunt = set(map(OWLNamedIndividual,
                           map(IRI.create,
                               settings['problems']['Aunt']['positive_examples'])))
        neg_aunt = set(map(OWLNamedIndividual,
                           map(IRI.create,
                               settings['problems']['Aunt']['negative_examples'])))

        pos_uncle = set(map(OWLNamedIndividual,
                            map(IRI.create,
                                settings['problems']['Uncle']['positive_examples'])))
        neg_uncle = set(map(OWLNamedIndividual,
                            map(IRI.create,
                                settings['problems']['Uncle']['negative_examples'])))
        model = CELOE(knowledge_base=kb, max_runtime=1000, max_num_of_concepts_tested=100)
        model.fit(pos=pos_aunt, neg=neg_aunt)
        kb.clean()
        model.fit(pos=pos_uncle, neg=neg_uncle)

        print("First fitted on Aunt then on Uncle:")
        hypotheses = list(model.best_hypotheses(n=2))

        q, str_concept = compute_f1_score(individuals={i for i in kb.individuals(hypotheses[0])}, pos=pos_uncle, neg=neg_uncle), hypotheses[0]
        kb.clean()
        kb = KnowledgeBase(path=PATH_FAMILY)
        model = CELOE(knowledge_base=kb, max_runtime=1000, max_num_of_concepts_tested=100)
        model.fit(pos=pos_uncle, neg=neg_uncle)

        print("Only fitted on Uncle:")
        hypotheses = list(model.best_hypotheses(n=2))

        q2, str_concept2 = compute_f1_score(individuals={i for i in kb.individuals(hypotheses[0])}, pos=pos_uncle, neg=neg_uncle), hypotheses[0]

        assert q == q2
        assert str_concept == str_concept2
