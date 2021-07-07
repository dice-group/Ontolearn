""" Test the default pipeline for structured machine learning"""
import json
import unittest

from ontolearn import KnowledgeBase
from ontolearn.concept_learner import CELOE
from ontolearn.learning_problem import PosNegLPStandard
from ontolearn.model_adapter import ModelAdapter
from ontolearn.utils import setup_logging
from owlapy.model import OWLNamedIndividual, OWLClass, IRI
from owlapy.render import DLSyntaxObjectRenderer

setup_logging("logging_test.conf")

PATH_FAMILY = 'KGs/Family/family-benchmark_rich_background.owl'
PATH_DATA_FATHER = 'KGs/father.owl'

with open('examples/synthetic_problems.json') as json_file:
    settings = json.load(json_file)


class Celoe_Test(unittest.TestCase):
    def test_celoe(self):
        kb = KnowledgeBase(path=PATH_FAMILY)

        exp_qualities = {'Aunt': .80392, 'Brother': 1.0,
                         'Cousin': .68063, 'Granddaughter': 1.0,
                         'Uncle': .88372, 'Grandgrandfather': 0.94444}
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
            model = CELOE(knowledge_base=target_kb)

            returned_val = model.fit(learning_problem=lp)
            self.assertEqual(returned_val, model, "fit should return its self")
            hypotheses = list(model.best_hypotheses(n=3))
            self.assertGreaterEqual(hypotheses[0].quality, exp_qualities[str_target_concept],
                                    "we only ever improve the quality")
            self.assertGreaterEqual(hypotheses[0].quality, hypotheses[1].quality, "the hypotheses are quality ordered")
            self.assertGreaterEqual(hypotheses[1].quality, hypotheses[2].quality)

    def test_celoe_father(self):
        kb = KnowledgeBase(path=PATH_DATA_FATHER)
        # with (kb.onto):
        #    sync_reasoner()
        # sync_reasoner()

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
        best_pred = model.best_hypotheses(n=1).__iter__().__next__()
        print(best_pred)
        self.assertEqual(best_pred.quality, 1.0)
        r = DLSyntaxObjectRenderer()
        self.assertEqual(r.render(best_pred.concept), 'male ⊓ (∃ hasChild.⊤)')

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

        model = ModelAdapter(learner_type=CELOE, knowledge_base=kb, max_runtime=1000, max_num_of_concepts_tested=100)
        model.fit(pos=pos_aunt, neg=neg_aunt)
        kb.clean()
        model.fit(pos=pos_uncle, neg=neg_uncle)

        print("First fitted on Aunt then on Uncle:")
        hypotheses = list(model.best_hypotheses(n=2))
        q, str_concept = hypotheses[0].quality, hypotheses[0].concept
        kb.clean()
        kb = KnowledgeBase(path=PATH_FAMILY)
        model = ModelAdapter(learner_type=CELOE, knowledge_base=kb, max_runtime=1000, max_num_of_concepts_tested=100)
        model.fit(pos=pos_uncle, neg=neg_uncle)

        print("Only fitted on Uncle:")
        hypotheses = list(model.best_hypotheses(n=2))
        q2, str_concept2 = hypotheses[0].quality, hypotheses[0].concept

        self.assertEqual(q, q2)
        self.assertEqual(str_concept, str_concept2)


if __name__ == '__main__':
    unittest.main()
