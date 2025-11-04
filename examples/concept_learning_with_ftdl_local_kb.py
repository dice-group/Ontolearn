import json

from owlapy.owl_individual import OWLNamedIndividual, IRI

from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.learners import FTDL
from ontolearn.learning_problem import PosNegLPStandard
from owlapy.render import DLSyntaxObjectRenderer

kb = KnowledgeBase(path="../KGs/Family/family-benchmark_rich_background.owl")
render = DLSyntaxObjectRenderer()
model = FTDL(kb)


with open('../LPs/Family/lps.json') as json_file:
    settings = json.load(json_file)
p = set(settings['problems']['Aunt']['positive_examples'])
n = set(settings['problems']['Aunt']['negative_examples'])
typed_pos = set(map(OWLNamedIndividual, map(IRI.create, p)))
typed_neg = set(map(OWLNamedIndividual, map(IRI.create, n)))

lp = PosNegLPStandard(pos=typed_pos, neg=typed_neg)

h = model.fit(learning_problem=lp).best_hypotheses(n=1)
#pred_ftdl = ftdl.fit(lp).best_hypotheses(n=1)
str_concept = render.render(h)
print("Concept:", str_concept)

