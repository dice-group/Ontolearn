import json
import random
from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.learners import CELOE
from ontolearn.heuristics import CELOEHeuristic
from ontolearn.learning_problem import PosNegLPStandard
from ontolearn.metrics import Accuracy
from owlapy.owl_individual import OWLNamedIndividual, IRI
from owlapy.class_expression import OWLClass
from ontolearn.refinement_operators import ModifiedCELOERefinement


class TestCELOE:
    def test_example(self):
        with open('examples/synthetic_problems.json') as json_file:
            settings = json.load(json_file)

        kb = KnowledgeBase(path="KGs/Family/family-benchmark_rich_background.owl")

        random.seed(0)

        # noinspection DuplicatedCode
        for str_target_concept, examples in settings['problems'].items():
            p = set(examples['positive_examples'])
            n = set(examples['negative_examples'])
            print('Target concept: ', str_target_concept)

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

            qual = Accuracy()
            heur = CELOEHeuristic(expansionPenaltyFactor=0.05, startNodeBonus=1.0, nodeRefinementPenalty=0.01)
            op = ModifiedCELOERefinement(knowledge_base=target_kb, use_negation=False, use_all_constructor=False)

            model = CELOE(knowledge_base=target_kb,
                          max_runtime=600,
                          refinement_operator=op,
                          quality_func=qual,
                          heuristic_func=heur,
                          max_num_of_concepts_tested=100,
                          iter_bound=100)
            model.fit(lp)
            model.save_best_hypothesis(n=1, path=f'Predictions_{str_target_concept}')
            # Get Top n hypotheses
            hypotheses = list(model.best_hypotheses(n=3))
            # Use hypotheses as binary function to label individuals.
            predictions = model.predict(individuals=list(typed_pos | typed_neg),
                                        hypotheses=hypotheses)

    def text_second_ma_example(self):

        with open('examples/synthetic_problems.json') as json_file:
            settings = json.load(json_file)

        kb = KnowledgeBase(path="KGs/Family/family-benchmark_rich_background.owl")

        random.seed(0)

        # noinspection DuplicatedCode
        for str_target_concept, examples in settings['problems'].items():
            p = set(examples['positive_examples'])
            n = set(examples['negative_examples'])
            print('Target concept: ', str_target_concept)

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

            qual = Accuracy()
            heur = CELOEHeuristic(expansionPenaltyFactor=0.05, startNodeBonus=1.0, nodeRefinementPenalty=0.01)
            op = ModifiedCELOERefinement(knowledge_base=target_kb, use_negation=False, use_all_constructor=False)

            model = CELOE(knowledge_base=target_kb,
                          max_runtime=600,
                          refinement_operator=op,
                          quality_func=qual,
                          heuristic_func=heur,
                          max_num_of_concepts_tested=100,
                          iter_bound=100)
            model.fit(lp)
            model.save_best_hypothesis(n=1, path=f'Predictions_{str_target_concept}')
            # Get Top n hypotheses
            hypotheses = list(model.best_hypotheses(n=3))
            # Use hypotheses as binary function to label individuals.
            predictions = model.predict(individuals=list(typed_pos | typed_neg),
                                        hypotheses=hypotheses)
