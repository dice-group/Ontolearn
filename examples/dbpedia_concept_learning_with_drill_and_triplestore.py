import json, os
from owlapy.owl_individual import OWLNamedIndividual, IRI
from ontolearn.learners import Drill
from ontolearn.learning_problem import PosNegLPStandard
from ontolearn.triple_store import TripleStore
from ontolearn.utils.static_funcs import save_owl_class_expressions
from owlapy.render import DLSyntaxObjectRenderer
# (1) Initialize Triplestore
kb = TripleStore(url="https://dbpedia.data.dice-research.org/sparql")
# (2) Initialize a DL renderer.
renderer = DLSyntaxObjectRenderer()
# (3) Initialize a learner.
model = Drill(knowledge_base=kb, max_runtime=240)
# (4) Solve learning problems
with open("./LPs/DBpedia2022-12/lps.json") as f:
    lps = json.load(f)
for i, item in enumerate(lps):
    print("\nTarget expression: ", item["target expression"], "\n")
    lp = PosNegLPStandard(pos=set(list(map(OWLNamedIndividual,map(IRI.create, item["examples"]["positive examples"])))),
                          neg=set(list(map(OWLNamedIndividual,map(IRI.create, item["examples"]["negative examples"])))))
    # (5) Learn description logic concepts best fitting (4).
    h = model.fit(learning_problem=lp).best_hypotheses()
    str_concept = renderer.render(h)
    print("Concept:", str_concept)  # e.g.  ∃ predecessor.WikicatPeopleFromBerlin
    # (6) Save ∃ predecessor.WikicatPeopleFromBerlin into disk
    if not os.path.exists("./learned_owl_expressions_drill"):
        os.mkdir("./learned_owl_expressions_drill")
    save_owl_class_expressions(expressions=h, path=f"./learned_owl_expressions_drill/owl_prediction_{i}")
