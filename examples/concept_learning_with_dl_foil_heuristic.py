import json
from ontolearn import KnowledgeBase
from ontolearn.concept_learner import CustomConceptLearner
from ontolearn.refinement_operators import CustomRefinementOperator
from ontolearn.heuristics import DLFOILHeuristic
from ontolearn.metrics import Precision,Accuracy,Recall,F1
from ontolearn.search import SearchTree

with open('synthetic_problems.json') as json_file:
    settings = json.load(json_file)

kb = KnowledgeBase(path=settings['data_path'])

for str_target_concept, examples in settings['problems'].items():
    p = set(examples['positive_examples'])
    n = set(examples['negative_examples'])
    print('Target concept: ', str_target_concept)
    model = CustomConceptLearner(knowledge_base=kb,
                                 refinement_operator=CustomRefinementOperator(kb=kb),
                                 quality_func=F1(),
                                 heuristic_func=DLFOILHeuristic(),
                                 search_tree=SearchTree(),
                                 terminate_on_goal=True,
                                 max_num_of_concepts_tested=300,
                                 iter_bound=1_000,
                                 verbose=True)

    best_pred=model.predict(pos=p, neg=n)
    model.show_best_predictions(top_n=10, key='quality',
                                serialize_name=str_target_concept + '_quality_structured_prediction.owl')
    # model.show_best_predictions(top_n=10, key='heuristic',serialize_name=str_target_concept +
    # '_heuristic_structured_prediction.owl') model.show_best_predictions(top_n=10, key='length',
    # serialize_name=str_target_concept + '_length_structured_prediction.owl')
    # model.extend_ontology(top_n_concepts=20)

