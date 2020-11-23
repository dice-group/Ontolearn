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

"""
====================================================================
Drill -- Deep Reinforcement Learning for Refinement Operators in ALC
====================================================================
Drill is a convolutional deep Q-learning approach that prunes the search space of refinement operators for class expression learning.
In this example, we illustrates a default workflow of Drill.
"""
# Authors: Caglar Demir

from ontolearn import KnowledgeBase, LearningProblemGenerator
from ontolearn.rl import DrillSample, DrillAverage
from ontolearn.refinement_operators import LengthBasedRefinement
from ontolearn.util import apply_TSNE_on_df
import json
import pandas as pd

PATH_FAMILY = '../data/family-benchmark_rich_background.owl'
family_embeddings_path = '../embeddings/Dismult_family_benchmark/instance_emb.csv'

with open('synthetic_problems.json') as json_file:
    settings = json.load(json_file)
data_path = settings['data_path']

kb = KnowledgeBase(PATH_FAMILY)
rho = LengthBasedRefinement(kb=kb)

balanced_examples = LearningProblemGenerator(knowledge_base=kb, num_problems=100, min_num_ind=15).balanced_examples

instance_emb = pd.read_csv(family_embeddings_path, index_col=0)
# apply_TSNE_on_df(instance_emb)  # if needed.

model_sub = DrillSample(knowledge_base=kb, refinement_operator=rho,
                        num_episode=20, instance_embeddings=instance_emb).train(balanced_examples, relearn_ratio=2)

model_avg = DrillAverage(knowledge_base=kb, refinement_operator=rho,
                         num_episode=20, instance_embeddings=instance_emb).train(balanced_examples, relearn_ratio=2)
