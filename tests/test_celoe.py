""" Test the default pipeline for structured machine learning"""
import json
from ontolearn import KnowledgeBase
from ontolearn.concept_learner import CELOE

with open('examples/synthetic_problems.json') as json_file:
    settings = json.load(json_file)
# because '../data/family-benchmark_rich_background.owl'
kb = KnowledgeBase(path=settings['data_path'][3:])

PATH_DATA_FATHER = 'data/father.owl'


class TestCeloe:
    def test_celoe(self):
        exp_qualities = {'Aunt': .80392, 'Brother': 1.0,
                         'Cousin': .68063, 'Granddaughter': 1.0,
                         'Uncle': .88372, 'Grandgrandfather': 0.94444}
        for str_target_concept, examples in settings['problems'].items():
            p = set(examples['positive_examples'])
            n = set(examples['negative_examples'])
            print('Target concept: ', str_target_concept)
            concepts_to_ignore = set()
            # lets inject more background info
            if str_target_concept in ['Granddaughter', 'Aunt', 'Sister']:
                # Use URI, or concept with length 1
                concepts_to_ignore.update(
                    {'http://www.benchmark.org/family#Brother',
                     'Father', 'http://www.benchmark.org/family#Grandparent'})

            model = CELOE(knowledge_base=kb,
                          ignored_concepts=concepts_to_ignore)

            returned_val = model.fit(pos=p, neg=n)
            assert returned_val == model
            hypotheses = model.best_hypotheses(n=3)
            assert hypotheses[0].quality >= exp_qualities[str_target_concept]
            assert hypotheses[0].quality >= hypotheses[1].quality
            assert hypotheses[1].quality >= hypotheses[2].quality

    def test_celoe_predictions(self):
        with open('examples/synthetic_problems.json') as json_file:
            settings = json.load(json_file)

        for str_target_concept, examples in settings['problems'].items():
            if str_target_concept == 'Aunt':
                p = set(examples['positive_examples'])
                n = set(examples['negative_examples'])
                print('Target concept: ', str_target_concept)
                model = CELOE(knowledge_base=kb)

                model.fit(pos=p, neg=n)

                best_preds = model.best_hypotheses(n=1)
                assert best_preds[0].quality > 0.8

    def test_celoe_father(self):
        kb = KnowledgeBase(path=PATH_DATA_FATHER)
        # with (kb.onto):
        #    sync_reasoner()
        # sync_reasoner()

        examples = {
            'positive_examples': [
                "http://example.com/father#stefan",
                "http://example.com/father#markus",
                "http://example.com/father#martin"],
            'negative_examples': [
                "http://example.com/father#heinz",
                "http://example.com/father#anna",
                "http://example.com/father#michelle"]
        }

        p = set(examples['positive_examples'])
        n = set(examples['negative_examples'])

        model = CELOE(knowledge_base=kb)

        model.fit(pos=p, neg=n)
        best_pred = model.best_hypotheses(n=1)[0]
        print(best_pred)
        assert (best_pred.quality == 1.0)
        assert (best_pred.concept.str == '(male  ⊓  (∃hasChild.person))')
