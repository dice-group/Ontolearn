from ontolearn import KnowledgeBase, LearningProblemGenerator, DrillSample, DrillAverage
from ontolearn import CELOE, OCEL, DLFOILHeuristic, CustomConceptLearner
from ontolearn import Experiments
from ontolearn.binders import DLLearnerBinder
import pandas as pd

PATH_FAMILY = '/home/demir/Desktop/Onto-learn_dev/data/family-benchmark_rich_background.owl'
family_embeddings_path = '../embeddings/dismult_family_benchmark/instance_emb.csv'
dl_learner_path = '/home/demir/Desktop/DL/dllearner-1.4.0/'

kb = KnowledgeBase(PATH_FAMILY)
emb = pd.read_csv(family_embeddings_path, index_col=0)

balanced_examples = LearningProblemGenerator(knowledge_base=kb, num_problems=10, min_num_ind=15).balanced_examples

# Initialize models
celoe = DLLearnerBinder(binary_path=dl_learner_path, kb_path=PATH_FAMILY, model='celoe')
ocel = DLLearnerBinder(binary_path=dl_learner_path, kb_path=PATH_FAMILY, model='ocel')
eltl = DLLearnerBinder(binary_path=dl_learner_path, kb_path=PATH_FAMILY, model='eltl')
drill_average = DrillAverage(knowledge_base=kb, instance_embeddings=emb, num_episode=10)  # => almost no training
drill_sample = DrillSample(knowledge_base=kb, instance_embeddings=emb, num_episode=10)
celoe_python = CELOE(knowledge_base=kb)
ocel_python = OCEL(knowledge_base=kb)
dl_foil = CustomConceptLearner(knowledge_base=kb, heuristic_func=DLFOILHeuristic())

exp = Experiments()
k_fold_cross_validation = exp.start_KFold(k=10, dataset=balanced_examples,
                                          models=[celoe_python, ocel_python, drill_average, drill_sample, dl_foil,
                                                  celoe, ocel, eltl],
                                          # celoe, ocel, eltl
                                          max_runtime=1)
print('\n##### K-FOLD CROSS VALUATION RESULTS #####')
for k, v in k_fold_cross_validation.items():
    f1 = v['F-measure']
    acc = v['Accuracy']
    runtime = v['Runtime']
    m = '{}\t F-measure:(avg.{:.2f} | std.{:.2f})\tAccuracy:(avg.{:.2f} | std.{:.2f})\t' \
        'Runtime:(avg.{:.2f} | std.{:.2f})'.format(k,
                                                   f1.mean(), f1.std(),
                                                   acc.mean(),
                                                   acc.std(),
                                                   runtime.mean(), runtime.std())
    print(m)
