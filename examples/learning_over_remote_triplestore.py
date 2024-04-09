from ontolearn.triple_store import TripleStore
from ontolearn.learners import TDL
from ontolearn.learners import Drill
from owlapy.model import OWLNamedIndividual, IRI
from ontolearn.learning_problem import PosNegLPStandard
url = "http://dice-dbpedia.cs.upb.de:9080/sparql"
examples = {"positive_examples": ["http://dbpedia.org/resource/Angela_Merkel"], "negative_examples": ["http://dbpedia.org/resource/Barack_Obama"]}
kb = TripleStore(url=url)
model = TDL(knowledge_base=kb, report_classification=True, kwargs_classifier={"random_state": 1})
# or model = Drill(knowledge_base=kb)
typed_pos = set(map(OWLNamedIndividual, map(IRI.create, examples["positive_examples"])))
typed_neg = set(map(OWLNamedIndividual, map(IRI.create, examples["negative_examples"])))
lp = PosNegLPStandard(pos=typed_pos, neg=typed_neg)
predicted_expression = model.fit(learning_problem=lp).best_hypotheses()
print(predicted_expression)
