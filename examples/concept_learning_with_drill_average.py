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

from ontolearn import KnowledgeBase, LengthBasedRefinement, DrillAverage
import json
import pandas as pd

PATH_FAMILY = '../data/family-benchmark_rich_background.owl'
# drill_pretrained_model_path = '../agent_pre_trained/model.pth'
family_embeddings_path = '../embeddings/Dismult_family_benchmark/instance_emb.csv'
synthetic_problems_path = '../examples/synthetic_problems.json'

with open(synthetic_problems_path) as json_file:
    settings = json.load(json_file)

kb = KnowledgeBase(PATH_FAMILY)
rho = LengthBasedRefinement(kb=kb)

instance_emb = pd.read_csv(family_embeddings_path, index_col=0)
model = DrillAverage(knowledge_base=kb, refinement_operator=rho, instance_embeddings=instance_emb)
for str_target_concept, examples in settings['problems'].items():
    p = set(examples['positive_examples'])
    n = set(examples['negative_examples'])
    print('Target concept: ', str_target_concept)
    concepts_to_ignore = set()
    # lets inject more background info
    if str_target_concept in ['Granddaughter', 'Aunt', 'Sister']:
        concepts_to_ignore.update(
            {'http://www.benchmark.org/family#Brother', 'Father', 'Grandparent'})  # Use URI, or concept with length 1.
    model.fit(pos=p, neg=n, ignore=concepts_to_ignore)
    # Get Top n hypotheses
    hypotheses = model.best_hypotheses(n=2)
    # Use hypotheses as binary function to label individuals.
    predictions = model.predict(individuals=list(p) + list(n), hypotheses=hypotheses)
    print(predictions)
