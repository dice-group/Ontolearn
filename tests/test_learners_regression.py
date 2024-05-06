import json
from ontolearn.learning_problem import PosNegLPStandard
from owlapy.owl_individual import OWLNamedIndividual
from owlapy.iri import IRI

from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.concept_learner import EvoLearner, CELOE, OCEL
from ontolearn.learners import Drill
from ontolearn.metrics import F1
from ontolearn.utils.static_funcs import compute_f1_score


class TestConceptLearnerReg:

    def test_regression_family(self):
        with open('examples/synthetic_problems.json') as json_file:
            settings = json.load(json_file)
        kb = KnowledgeBase(path=settings['data_path'][3:])
        max_runtime = 10

        ocel = OCEL(knowledge_base=kb, quality_func=F1(), max_runtime=max_runtime)
        celoe = CELOE(knowledge_base=kb, quality_func=F1(), max_runtime=max_runtime)
        evo = EvoLearner(knowledge_base=kb, quality_func=F1(), max_runtime=max_runtime)
        # drill = Drill(knowledge_base=kb, quality_func=F1(), max_runtime=max_runtime)

        drill_quality = []
        celoe_quality = []
        ocel_quality = []
        evo_quality = []

        for str_target_concept, examples in settings['problems'].items():
            pos = set(map(OWLNamedIndividual, map(IRI.create, set(examples['positive_examples']))))
            neg = set(map(OWLNamedIndividual, map(IRI.create, set(examples['negative_examples']))))
            print('Target concept: ', str_target_concept)

            lp = PosNegLPStandard(pos=pos, neg=neg)
            # Untrained & max runtime is not fully integrated.
            # Compute qualities explicitly
            ocel_quality.append(compute_f1_score(individuals=
                                                 frozenset({i for i in kb.individuals(
                                                     ocel.fit(lp).best_hypotheses(n=1, return_node=False))}),
                                                 pos=lp.pos,
                                                 neg=lp.neg))
            celoe_quality.append(compute_f1_score(individuals=
                                                  frozenset({i for i in kb.individuals(
                                                      celoe.fit(lp).best_hypotheses(n=1, return_node=False))}),
                                                  pos=lp.pos,
                                                  neg=lp.neg))
            evo_quality.append(compute_f1_score(individuals=
                                                frozenset({i for i in kb.individuals(
                                                    evo.fit(lp).best_hypotheses(n=1, return_node=False))}),
                                                pos=lp.pos,
                                                neg=lp.neg))
            # @TODO:CD:Will be added after least_generate and most_general_owl get methods are implemented in KB class.
            #drill_quality.append(compute_f1_score(individuals=
            #                                      frozenset({i for i in kb.individuals(
            #                                          drill.fit(lp).best_hypotheses(n=1, return_node=False))}),
            #                                      pos=lp.pos,
            #                                      neg=lp.neg))

        # assert sum(evo_quality) >= sum(drill_quality)
        assert sum(celoe_quality) >= sum(ocel_quality)
