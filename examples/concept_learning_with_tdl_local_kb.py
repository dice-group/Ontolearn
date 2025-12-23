import json

from owlapy.owl_individual import OWLNamedIndividual, IRI

from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.learners import TDL
from ontolearn.learning_problem import PosNegLPStandard
from owlapy.render import DLSyntaxObjectRenderer

kb = KnowledgeBase(path="../KGs/Mutagenesis/mutagenesis.owl")
render = DLSyntaxObjectRenderer()
model = TDL(kb)


with open('../LPs/Mutagenesis/lps.json') as json_file:
    settings = json.load(json_file)
p = set(settings['problems']['NotKnown']['positive_examples'])
n = set(settings['problems']['NotKnown']['negative_examples'])
typed_pos = set(map(OWLNamedIndividual, map(IRI.create, p)))
typed_neg = set(map(OWLNamedIndividual, map(IRI.create, n)))
lp = PosNegLPStandard(pos=typed_pos, neg=typed_neg)

h = model.fit(learning_problem=lp).best_hypotheses()
str_concept = render.render(h)
print("Concept:", str_concept)

