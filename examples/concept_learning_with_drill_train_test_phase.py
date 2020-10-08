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
lp_gen = LearningProblemGenerator(knowledge_base=kb,
                                  refinement_operator=rho,
                                  num_problems=3, depth=2, min_length=2)

instance_emb = pd.read_csv('../embeddings/instance_emb.csv', index_col=0)
# util.apply_TSNE_on_df(instance_emb) # if needed.
trainer = DrillTrainer(
    knowledge_base=kb,
    refinement_operator=rho,
    quality_func=F1(),
    reward_func=Reward(),  # Reward func.
    search_tree=SearchTreePriorityQueue(),
    path_pretrained_agent='../agent_pre_trained',  # for Incremental/Continual learning.
    learning_problem_generator=lp_gen,
    instance_embeddings=instance_emb,
    verbose=False)
#trainer.start() # comment for only testing.

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
                                terminate_on_goal=True,
                                iter_bound=1_000,
                                max_num_of_concepts_tested=5_000,
                                ignored_concepts={},
                                verbose=True)
    model.fit(pos=p, neg=n)
