from ontolearn import KnowledgeBase, LengthBasedRefinement, SearchTreePriorityQueue, LearningProblemGenerator
from ontolearn.rl import DrillTrainer
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
                                  num_problems=200, depth=1, min_length=1)

instance_emb = pd.read_csv('../embeddings/instance_emb.csv', index_col=0)
# apply_TSNE_on_df(instance_emb)

trainer = DrillTrainer(
    knowledge_base=kb,
    refinement_operator=rho,
    quality_func=F1(),
    reward_func=Reward(),
    search_tree=SearchTreePriorityQueue(),
    train_data=settings['problems'],
    pre_trained_drill=None,
    relearn_rate_per_problem=3.0,
    learning_problem_generator=lp_gen,
    instance_embeddings=instance_emb)
trainer.start()
