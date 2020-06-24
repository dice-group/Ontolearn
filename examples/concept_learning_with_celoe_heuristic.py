from ontolearn import *
import json

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
        concepts_to_ignore.update({'Brother', 'Father', 'Uncle','Grandparent'})

    model = CELOE(knowledge_base=kb,
                  refinement_operator=ModifiedCELOERefinement(kb=kb),
                  quality_func=F1(),
                  min_horizontal_expansion=3,
                  heuristic_func=CELOEHeuristic(),
                  search_tree=CELOESearchTree(),
                  terminate_on_goal=True,
                  iter_bound=1_000,
                  ignored_concepts=concepts_to_ignore,
                  verbose=False)

    model.predict(pos=p, neg=n)
    model.show_best_predictions(top_n=10)
    model.extend_ontology(top_n_concepts=10)
