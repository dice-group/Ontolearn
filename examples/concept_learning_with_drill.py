from ontolearn import KnowledgeBase, LengthBasedRefinement, SearchTreePriorityQueue, LearningProblemGenerator, \
    LengthBaseLearner
from ontolearn.rl import DrillTrainer, DrillHeuristic, DrillConceptLearner
from ontolearn.metrics import F1
from ontolearn.heuristics import Reward
import json
import numpy as np
import random
import pandas as pd
from ontolearn.util import apply_TSNE_on_df

with open('synthetic_problems.json') as json_file:
    settings = json.load(json_file)

kb = KnowledgeBase(path=settings['data_path'])
rho = LengthBasedRefinement(kb=kb)
lp_gen = LearningProblemGenerator(knowledge_base=kb,
                                  refinement_operator=rho,
                                  num_problems=3, depth=1, min_length=1)

instance_emb = pd.read_csv('../embeddings/instance_emb.csv', index_col=0)
# apply_TSNE_on_df(instance_emb)
trainer = DrillTrainer(
    knowledge_base=kb,
    refinement_operator=rho,
    quality_func=F1(),
    reward_func=Reward(),
    search_tree=SearchTreePriorityQueue(),
    pre_trained_drill=None,
    learning_problem_generator=lp_gen,
    instance_embeddings=instance_emb,
    verbose=False)
trainer.start()

for str_target_concept, examples in settings['problems'].items():
    p = set(examples['positive_examples'])
    n = set(examples['negative_examples'])
    print('Target concept: ', str_target_concept)
    concepts_to_ignore = set()
    # lets inject more background info
    if str_target_concept in ['Granddaughter', 'Aunt', 'Sister']:
        concepts_to_ignore.update({'Brother', 'Father', 'Uncle', 'Grandparent'})

    model = DrillConceptLearner(knowledge_base=kb,
                                refinement_operator=rho,
                                quality_func=F1(),
                                heuristic_func=DrillHeuristic(model=trainer.model),
                                instance_emb=instance_emb,
                                search_tree=SearchTreePriorityQueue(),
                                min_length=1,  # think better variable name
                                terminate_on_goal=True,
                                iter_bound=1_000,
                                max_num_of_concepts_tested=5_000,
                                ignored_concepts={},
                                verbose=True)
    model.predict(pos=p, neg=n)
    model.best_hypotheses(top_n=10)
    model.save_best_hypotheses(file_path=str_target_concept + '_best_hypothesis.owl', top_n=10)