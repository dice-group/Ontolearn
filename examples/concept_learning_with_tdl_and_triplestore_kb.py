
from owlapy.model import IRI, OWLNamedIndividual
from ontolearn.learners import TDL
from ontolearn.learning_problem import PosNegLPStandard
from ontolearn.utils.static_funcs import compute_f1_score
from ontolearn.triple_store import TripleStore
from ontolearn.utils.static_funcs import save_owl_class_expressions
from owlapy.render import DLSyntaxObjectRenderer
# (1) Initialize Triplestore- Make sure that UPB VPN is on
kb = TripleStore(url="http://dice-dbpedia.cs.upb.de:9080/sparql")
# (2) Initialize a DL renderer.
render = DLSyntaxObjectRenderer()
# (3) Initialize a learner.
model = TDL(knowledge_base=kb)
# (4) Define a description logic concept learning problem.
lp = PosNegLPStandard(pos={OWLNamedIndividual(IRI.create("http://dbpedia.org/resource/Angela_Merkel"))},
                      neg={OWLNamedIndividual(IRI.create("http://dbpedia.org/resource/Barack_Obama"))})
# (5) Learn description logic concepts best fitting (4).
h = model.fit(learning_problem=lp).best_hypotheses()
str_concept = render.render(h)
print("Concept:", str_concept)  # e.g.  ∃ predecessor.WikicatPeopleFromBerlin
# (6) Save ∃ predecessor.WikicatPeopleFromBerlin into disk
save_owl_class_expressions(expressions=h, path="owl_prediction")
