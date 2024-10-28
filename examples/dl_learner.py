"""
====================================================================
Using DL-Learner

In this example, we show how to use DL-Learner within our framework.

Via, DLLearnerBinder, a new class expression learning algorithm can be easily compared against CELOE, OCEL, and ELTL.
Moreover, the user does not need to write a config file for each learning problem.

# Download DL-Learner
wget --no-check-certificate --content-disposition https://github.com/SmartDataAnalytics/DL-Learner/releases/download/1.5.0/dllearner-1.5.0.zip && unzip dllearner-1.5.0.zip
# Test the DL-learner framework
dllearner-1.5.0/bin/cli dllearner-1.5.0/examples/father.conf


A log file is on the fly generated and detailed results along with the hyperparameters are stored.
====================================================================
Author: Caglar Demir
"""
from ontolearn.binders import DLLearnerBinder
from owlapy.iri import IRI
from owlapy.owl_individual import OWLNamedIndividual
import json

from ontolearn.learning_problem import PosNegLPStandard

# (1) Load learning problems
with open('synthetic_problems.json') as json_file:
    settings = json.load(json_file)
# (1) Enter the absolute path of the input knowledge base
kb_path = '/home/demir/Desktop/Softwares/Ontolearn/KGs/Family/family-benchmark_rich_background.owl'
# (2) To download DL-learner,  https://github.com/SmartDataAnalytics/DL-Learner/releases.
dl_learner_binary_path = 'dllearner-1.5.0/bin/cli'
# (3) Initialize CELOE, OCEL, and ELTL
celoe = DLLearnerBinder(binary_path=dl_learner_binary_path, kb_path=kb_path, model='celoe')
ocel = DLLearnerBinder(binary_path=dl_learner_binary_path, kb_path=kb_path, model='ocel')
eltl = DLLearnerBinder(binary_path=dl_learner_binary_path, kb_path=kb_path, model='eltl')
# (4) Fit (4) on the learning problems and show the best concept.
for str_target_concept, examples in settings['problems'].items():
    print('TARGET CONCEPT:', str_target_concept)
    p = examples['positive_examples']
    n = examples['negative_examples']

    positives = {OWLNamedIndividual(IRI.create(i)) for i in p}
    negatives = {OWLNamedIndividual(IRI.create(i)) for i in n}

    lp = PosNegLPStandard(pos=positives, neg=positives)

    best_pred_celoe = celoe.fit(lp, max_runtime=1).best_hypothesis()
    print(best_pred_celoe)
    best_pred_ocel = ocel.fit(lp, max_runtime=1).best_hypothesis()
    print(best_pred_ocel)
    best_pred_eltl = eltl.fit(lp, max_runtime=1).best_hypothesis()
    print(best_pred_eltl)
