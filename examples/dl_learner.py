"""
====================================================================
Using DL-Learner

In this example, we should how to use DL-Learner within our framework.

Via, DLLearnerBinder, a new class expression learning algorithm can be easily compared against CELOE, OCEL, and ELTL.
Moreover, the user does not need to write a config file for each learning problem.

A log file is on the fly generated and detailed results along with the hyperparameters are stored.
====================================================================
Author: Caglar Demir
"""
from ontolearn.binders import DLLearnerBinder
import json
# (1) Load learning problems
with open('synthetic_problems.json') as json_file:
    settings = json.load(json_file)
# (1) Enter the absolute path of the input knowledge base
kb_path = '/home/demir/Desktop/Softwares/Ontolearn/KGs/Family/family-benchmark_rich_background.owl'
# (2) To download DL-learner,  https://github.com/SmartDataAnalytics/DL-Learner/releases.
dl_learner_binary_path = '/home/demir/Desktop/DL/dllearner-1.4.0/'
# (3) Initialize CELOE, OCEL, and ELTL
celoe = DLLearnerBinder(binary_path=dl_learner_binary_path, kb_path=kb_path, model='celoe')
ocel = DLLearnerBinder(binary_path=dl_learner_binary_path, kb_path=kb_path, model='ocel')
eltl = DLLearnerBinder(binary_path=dl_learner_binary_path, kb_path=kb_path, model='eltl')
# (4) Fit (4) on the learning problems and show the best concept.
for str_target_concept, examples in settings['problems'].items():
    print('TARGET CONCEPT:', str_target_concept)
    p = examples['positive_examples']
    n = examples['negative_examples']

    best_pred_celoe = celoe.fit(pos=p, neg=n, max_runtime=1).best_hypothesis()
    print(best_pred_celoe)
    best_pred_ocel = ocel.fit(pos=p, neg=n, max_runtime=1).best_hypothesis()
    print(best_pred_ocel)
    best_pred_eltl = eltl.fit(pos=p, neg=n, max_runtime=1).best_hypothesis()
    print(best_pred_eltl)
