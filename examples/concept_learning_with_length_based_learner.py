import json

from ontolearn import LengthBaseLearner
from ontolearn import LengthBasedRefinement
from ontolearn import KnowledgeBase
from ontolearn import SearchTreePriorityQueue

from ontolearn.heuristics import CELOEHeuristic
from ontolearn.metrics import F1


with open('synthetic_problems.json') as json_file:
    settings = json.load(json_file)

kb = KnowledgeBase(path=settings['data_path'])

for str_target_concept, examples in settings['problems'].items():
    p = set(examples['positive_examples'])
    n = set(examples['negative_examples'])
    print('Target concept: ', str_target_concept)
    concepts_to_ignore = set()
    # lets inject more background info
    if str_target_concept in ['Granddaughter', 'Aunt', 'Sister']:
        concepts_to_ignore.update(
            {'Brother', 'Father', 'Uncle', 'Grandparent'})

    model = LengthBaseLearner(knowledge_base=kb,
                              refinement_operator=LengthBasedRefinement(kb=kb),
                              quality_func=F1(),
                              min_length=1,  # think better variable name
                              heuristic_func=CELOEHeuristic(),
                              search_tree=SearchTreePriorityQueue(),
                              terminate_on_goal=True,
                              iter_bound=1_000,
                              max_num_of_concepts_tested=5_000,
                              ignored_concepts={},
                              verbose=True)

    predictions = model.predict(pos=p, neg=n, n=10)

    model.save_predictions(
        predictions,
        key='quality',
        serialize_name=(str_target_concept +
                        '_quality_structured_prediction.owl'))
