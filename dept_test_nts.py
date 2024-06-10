""" CD: I guess we can delete this script """

### test file for neural tripel store
from ontolearn.learners import TDL
from ontolearn.learning_problem import PosNegLPStandard
from owlapy.owl_individual import OWLNamedIndividual
from owlapy.iri import IRI
from owlapy.render import DLSyntaxObjectRenderer
from ontolearn.neural_triple_store import NeuralTripleStore
from ontolearn.triple_store import TripleStore
from ontolearn.knowledge_base import KnowledgeBase

# (1) Initialize Triplestore
kb = NeuralTripleStore(path="KGs/father.owl")
# (2) Initialize a DL renderer.
render = DLSyntaxObjectRenderer()
# (3) Initialize a learner.
model = TDL(knowledge_base=kb)
# (4) Define a description logic concept learning problem.
lp = PosNegLPStandard(
    pos={OWLNamedIndividual(IRI.create("http://example.com/father#stefan"))},
    neg={
        OWLNamedIndividual(IRI.create("http://example.com/father#heinz")),
        OWLNamedIndividual(IRI.create("http://example.com/father#anna")),
        OWLNamedIndividual(IRI.create("http://example.com/father#michelle")),
    },
)
# (5) Learn description logic concepts best fitting (4).
h = model.fit(learning_problem=lp).best_hypotheses()

str_concept = render.render(h)
print("Concept:", str_concept)  # Concept: ∃ hasChild.{markus}
