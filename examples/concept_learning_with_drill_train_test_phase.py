"""
====================================================================
Drill -- Deep Reinforcement Learning for Refinement Operators in ALC
====================================================================
Drill is a convolutional deep Q-learning approach that prunes the search space of refinement operators for class expression learning.
In this example, we illustrates a default workflow of Drill.
"""
# Authors: Caglar Demir

from ontolearn import *
import json
import random
import pandas as pd

PATH_FAMILY = '../data/family-benchmark_rich_background.owl'
drill_pretrained_model_path = '../agent_pre_trained/model.pth'
family_embeddings_path = '../embeddings/Dismult_family_benchmark/instance_emb.csv'

with open('synthetic_problems.json') as json_file:
    settings = json.load(json_file)
data_path = settings['data_path']

kb = KnowledgeBase(PATH_FAMILY)
rho = LengthBasedRefinement(kb=kb)
lp_gen = LearningProblemGenerator(knowledge_base=kb, refinement_operator=rho,
                                  num_problems=5, min_length=3, max_length=6)

instance_emb = pd.read_csv(family_embeddings_path, index_col=0)
# util.apply_TSNE_on_df(instance_emb) # if needed.
model = DrillAverage(knowledge_base=kb, refinement_operator=rho,
                     heuristic_func=DrillHeuristic(),
                     # heuristic_func=DrillHeuristic(model_path=drill_pretrained_model_path),
                     max_num_of_concepts_tested=50_000,
                     instance_embeddings=instance_emb, verbose=1)
# model.train(lp_gen)
for str_target_concept, examples in settings['problems'].items():
    p = set(examples['positive_examples'])
    n = set(examples['negative_examples'])
    print('Target concept: ', str_target_concept)
    concepts_to_ignore = set()
    # lets inject more background info
    if str_target_concept in ['Granddaughter', 'Aunt', 'Sister']:
        concepts_to_ignore.update(
            {'http://www.benchmark.org/family#Brother', 'Father', 'Grandparent'})  # Use URI, or concept with length 1.

    model.fit(pos=p, neg=n)
    hypotheses = model.best_hypotheses(n=1)
    print(hypotheses[0])
