"""
====================================================================
Drill -- Deep Reinforcement Learning for Refinement Operators in ALC
====================================================================
Drill is a convolutional deep Q-learning approach that prunes the search space of refinement operators for class expression learning.
In this example, we illustrates a default workflow of Drill.
"""
# Authors: Caglar Demir

from ontolearn import KnowledgeBase, LengthBasedRefinement, LearningProblemGenerator, DrillAverage, DrillSample
import json
import pandas as pd

PATH_FAMILY = '../data/family-benchmark_rich_background.owl'
family_embeddings_path = '../embeddings/dismult_family_benchmark/instance_emb.csv'

with open('synthetic_problems.json') as json_file:
    settings = json.load(json_file)
data_path = settings['data_path']

kb = KnowledgeBase(PATH_FAMILY)
rho = LengthBasedRefinement(kb=kb)

balanced_examples = LearningProblemGenerator(knowledge_base=kb,  num_problems=1,min_num_ind=15).balanced_examples
instance_emb = pd.read_csv(family_embeddings_path, index_col=0)
# apply_TSNE_on_df(instance_emb)  # if needed.


model_sub = DrillSample(knowledge_base=kb, refinement_operator=rho,
                        num_episode=1,
                        instance_embeddings=instance_emb).train(balanced_examples)

model_avg = DrillAverage(knowledge_base=kb, refinement_operator=rho,
                         num_episode=1,
                         instance_embeddings=instance_emb).train(balanced_examples)

for str_target_concept, examples in settings['problems'].items():
    p = set(examples['positive_examples'])
    n = set(examples['negative_examples'])
    print('Target concept: ', str_target_concept)
    concepts_to_ignore = set()
    # lets inject more background info
    if str_target_concept in ['Granddaughter', 'Aunt', 'Sister']:
        concepts_to_ignore.update(
            {'http://www.benchmark.org/family#Brother', 'Father', 'Grandparent'})

    model_avg.fit(pos=p, neg=n, ignore=concepts_to_ignore)
    print(model_avg.best_hypotheses(n=1)[0])

    model_sub.fit(pos=p, neg=n, ignore=concepts_to_ignore)
    print(model_sub.best_hypotheses(n=1)[0])
