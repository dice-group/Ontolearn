import json
from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.concept_learner import EvoLearner
from ontolearn.learning_problem import PosNegLPStandard
from owlapy.owl_individual import OWLNamedIndividual, IRI
from ontolearn.utils import setup_logging
setup_logging()

"""
Before using verbalizer you have to: 
1. Install deeponto. `pip install deeponto` + further requirements like JDK, etc. 
   Check https://krr-oxford.github.io/DeepOnto/ for full instructions.
2. Install graphviz at https://graphviz.org/download/
"""

with open("uncle_lp.json") as json_file:
    settings = json.load(json_file)

kb = KnowledgeBase(path=settings['data_path'])

p = set(settings['problems']['Uncle']['positive_examples'])
n = set(settings['problems']['Uncle']['negative_examples'])
typed_pos = set(map(OWLNamedIndividual, map(IRI.create, p)))
typed_neg = set(map(OWLNamedIndividual, map(IRI.create, n)))
lp = PosNegLPStandard(pos=typed_pos, neg=typed_neg)

model = EvoLearner(knowledge_base=kb)
model.fit(lp)
model.save_best_hypothesis(n=3, path='Predictions_Uncle')

model.verbalize("Predictions_Uncle.owl")  # at most n=3 .png files will be generated
