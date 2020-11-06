"""
====================================================================
Drill -- Deep Reinforcement Learning for Refinement Operators in ALC
====================================================================
Drill is a convolutional deep Q-learning approach that prunes the search space of refinement operators for class expression learning.

In this example, we illustrates a default workflow of Drill.

1) KnowledgeBase class constructs TBOX and ABOX.

2) LengthBasedRefinement class represents length based refinement operator.

3) LearningProblemGenerator class requires  **num_problems**, **depth**, **min_length** to generate training data.

4) DrillTrainer class trains Drill on concepts generated in (3).

5) DrillConceptLearner class accepts Drill and applies it as heuristic function.

###################################################################
"""
# Authors: Caglar Demir

from ontolearn import *
import json
import random
import pandas as pd
PATH_FAMILY = '../data/family-benchmark_rich_background.owl'
drill_pretrained_model_path = '../agent_pre_trained/model.pth'
family_embeddings_path = '../embeddings/Dismult_family_benchmark/instance_emb.csv'
synthetic_problems_path = '../examples/synthetic_problems.json'

with open(synthetic_problems_path) as json_file:
    settings = json.load(json_file)

kb = KnowledgeBase(PATH_FAMILY)
rho = LengthBasedRefinement(kb=kb)

instance_emb = pd.read_csv(family_embeddings_path, index_col=0)

for str_target_concept, examples in settings['problems'].items():
    p = set(examples['positive_examples'])
    n = set(examples['negative_examples'])
    print('Target concept: ', str_target_concept)
    concepts_to_ignore = set()
    # lets inject more background info
    if str_target_concept in ['Granddaughter', 'Aunt', 'Sister']:
        concepts_to_ignore.update(
            {'http://www.benchmark.org/family#Brother', 'Father', 'Grandparent'})  # Use URI, or concept with length 1.

    model = DrillConceptLearner(knowledge_base=kb, refinement_operator=rho,
                                heuristic_func=DrillHeuristic(model_path=drill_pretrained_model_path),
                                terminate_on_goal=True, instance_emb=instance_emb, ignored_concepts=concepts_to_ignore)
    model.fit(pos=p, neg=n)
    hypotheses = model.best_hypotheses(n=1)
    for h in hypotheses:
        print(h)
