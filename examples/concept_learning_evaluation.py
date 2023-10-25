import json
import os
import time
import pandas as pd

from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.concept_learner import CELOE, EvoLearner, OCEL
from ontolearn.learning_problem import PosNegLPStandard
from ontolearn.metrics import Accuracy, F1
from ontolearn.owlapy.model import OWLClass, OWLNamedIndividual, IRI
from ontolearn.utils import setup_logging

setup_logging()

try:
    os.chdir("examples")
except FileNotFoundError:
    pass

with open('synthetic_problems.json') as json_file:
    settings = json.load(json_file)

kb = KnowledgeBase(path=settings['data_path'])

data = {
    "LP": ["", "Aunt", "Brother", "Cousin", "Granddaughter", "Uncle", "Grandgrandfather"],
    "EvoLearner": ["F1, Acc, RunTime"],
    "CELOE": ["F1, Acc, RunTime"],
    "OCEL": ["F1, Acc, RunTime"]
}
cls = list(data.keys())
quality_metrics = [F1(), Accuracy()]

for str_target_concept, examples in settings['problems'].items():
    p = set(examples['positive_examples'])
    n = set(examples['negative_examples'])
    print('Target concept: ', str_target_concept)

    # let's inject more background info
    if str_target_concept in ['Granddaughter', 'Aunt', 'Sister']:
        NS = 'http://www.benchmark.org/family#'
        concepts_to_ignore = {
            OWLClass(IRI(NS, 'Brother')),
            OWLClass(IRI(NS, 'Sister')),
            OWLClass(IRI(NS, 'Daughter')),
            OWLClass(IRI(NS, 'Mother')),
            OWLClass(IRI(NS, 'Grandmother')),
            OWLClass(IRI(NS, 'Father')),
            OWLClass(IRI(NS, 'Grandparent')),
            OWLClass(IRI(NS, 'PersonWithASibling')),
            OWLClass(IRI(NS, 'Granddaughter')),
            OWLClass(IRI(NS, 'Son')),
            OWLClass(IRI(NS, 'Child')),
            OWLClass(IRI(NS, 'Grandson')),
            OWLClass(IRI(NS, 'Grandfather')),
            OWLClass(IRI(NS, 'Grandchild')),
            OWLClass(IRI(NS, 'Parent')),
        }
        target_kb = kb.ignore_and_copy(ignored_classes=concepts_to_ignore)
    else:
        target_kb = kb

    typed_pos = set(map(OWLNamedIndividual, map(IRI.create, p)))
    typed_neg = set(map(OWLNamedIndividual, map(IRI.create, n)))
    lp = PosNegLPStandard(pos=typed_pos, neg=typed_neg)
    storage = {
        "EvoLearner": [0, 0, 0],
        "CELOE": [0, 0, 0],
        "OCEL": [0, 0, 0]
    }
    for qm in quality_metrics:
        for i in range(1, 4):
            if cls[i] == "EvoLearner":
                model = EvoLearner(knowledge_base=target_kb, quality_func=qm)
            elif cls[i] == "CELOE":
                model = CELOE(knowledge_base=target_kb, quality_func=qm)
            elif cls[i] == "OCEL":
                model = OCEL(knowledge_base=target_kb, quality_func=qm)

            start_time = time.time()
            model.fit(lp)
            runtime = time.time() - start_time
            runtime = round(runtime, 4)
            hypothesis = list(model.best_hypotheses(n=1))

            if qm == quality_metrics[0]:
                storage[cls[i]][0] = round(hypothesis[0].quality, 2)
                storage[cls[i]][2] = runtime
            else:
                storage[cls[i]][1] = round(hypothesis[0].quality, 2)
                storage[cls[i]][2] = round((storage[cls[i]][2] + runtime)/2, 4)

    for i in range(1, 4):
        data[cls[i]].append(f"{storage[cls[i]][0]}, {storage[cls[i]][1]}, {storage[cls[i]][2]}")


df = pd.DataFrame(data)
print(df.to_string())





