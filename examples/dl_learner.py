import json
import os

from ontolearn.binders import DLLearnerBinder

try:
    os.chdir("examples")
except FileNotFoundError:
    pass

with open('synthetic_problems.json') as json_file:
    settings = json.load(json_file)

kb_path = '/home/demir/Desktop/Onto-learn_dev/KGs/Family/family-benchmark_rich_background.owl'
kb_path='/home/demir/Desktop/Softwares/OntoPy/KGs/Family/family-benchmark_rich_background.owl'
# To download DL-learner,  https://github.com/SmartDataAnalytics/DL-Learner/releases.
dl_learner_binary_path = '/home/demir/Desktop/DL/dllearner-1.4.0/'

celoe = DLLearnerBinder(binary_path=dl_learner_binary_path, kb_path=kb_path, model='celoe')
ocel = DLLearnerBinder(binary_path=dl_learner_binary_path, kb_path=kb_path, model='ocel')
eltl = DLLearnerBinder(binary_path=dl_learner_binary_path, kb_path=kb_path, model='eltl')

for str_target_concept, examples in settings['problems'].items():
    print('TARGET CONCEPT:', str_target_concept)
    p = examples['positive_examples']
    n = examples['negative_examples']

    best_pred_celoe = celoe.fit(pos=p, neg=n, max_runtime=3).best_hypotheses()
    print(best_pred_celoe)
    best_pred_ocel = ocel.fit(pos=p, neg=n, max_runtime=3).best_hypotheses()
    print(best_pred_ocel)
    best_pred_eltl = eltl.fit(pos=p, neg=n, max_runtime=3).best_hypotheses()
    print(best_pred_eltl)
