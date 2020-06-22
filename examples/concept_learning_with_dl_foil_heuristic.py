from ontolearn import KnowledgeBase,CustomConceptLearner,CustomRefinementOperator,F1,DLFOILHeuristic,SearchTree
import json

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
                                 iter_bound=1_000,
                                 verbose=True)

    model.predict(pos=p, neg=n)
    model.show_best_predictions(top_n=10)
