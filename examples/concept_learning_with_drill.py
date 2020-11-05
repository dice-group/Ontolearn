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

with open('synthetic_problems.json') as json_file:
    settings = json.load(json_file)
data_path = settings['data_path']

kb = KnowledgeBase(path=data_path)
rho = LengthBasedRefinement(kb=kb)
instance_emb = pd.read_csv('../embeddings/Dismult_family_benchmark/instance_emb.csv', index_col=0)
trainer = DrillTrainer(
    knowledge_base=kb,
    refinement_operator=rho,
    path_pretrained_agent='../agent_pre_trained',  # for Incremental/Continual learning.
    instance_embeddings=instance_emb)

for str_target_concept, examples in settings['problems'].items():
    p = set(examples['positive_examples'])
    n = set(examples['negative_examples'])
    print('Target concept: ', str_target_concept)
    concepts_to_ignore = set()
    # lets inject more background info
    if str_target_concept in ['Granddaughter', 'Aunt', 'Sister']:
        concepts_to_ignore.update(
            {'http://www.benchmark.org/family#Brother', 'Father', 'Grandparent'}) # Use URI, or concept with length 1.

    model = DrillConceptLearner(knowledge_base=kb, refinement_operator=rho,
                                quality_func=F1(), heuristic_func=DrillHeuristic(model=trainer.model),
                                terminate_on_goal=True, instance_emb=instance_emb, ignored_concepts=concepts_to_ignore)
    model.fit(pos=p, neg=n)
    hypotheses = model.best_hypotheses(n=1)
    for h in hypotheses:
        print(h)

