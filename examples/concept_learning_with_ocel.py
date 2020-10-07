import json

from ontolearn import KnowledgeBase
from ontolearn.concept_learner import OCEL
from ontolearn.heuristics import OCELHeuristic
from ontolearn.metrics import F1
from ontolearn.refinement_operators import ModifiedCELOERefinement
from ontolearn.search import CELOESearchTree


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

    model = OCEL(knowledge_base=kb,
                 refinement_operator=ModifiedCELOERefinement(kb=kb),
                 quality_func=F1(),
                 min_horizontal_expansion=0,
                 heuristic_func=OCELHeuristic(),
                 search_tree=CELOESearchTree(),
                 terminate_on_goal=True,
                 iter_bound=1000,
                 max_num_of_concepts_tested=300,
                 ignored_concepts=concepts_to_ignore,
                 verbose=False)

    model.fit(pos=p, neg=n)
    hypotheses=model.best_hypotheses(n=10)
    predictions=model.predict(individuals=list(p),hypotheses=hypotheses)
    print(predictions)