import json, os, sys
from owlapy.owl_individual import OWLNamedIndividual, IRI
from ontolearn.learners import Drill, TDL
from ontolearn.learning_problem import PosNegLPStandard
from ontolearn.triple_store import TripleStore
from ontolearn.utils.static_funcs import save_owl_class_expressions
from owlapy.render import DLSyntaxObjectRenderer

if len(sys.argv) < 2:
    print("You need to provide the model name; either tdl or drill")
    sys.exit(1)

model_name = sys.argv[1]
assert model_name.lower() in ["drill", "tdl"], "Currently, only Drill and TDL are supported"

# (1) Initialize knowledge source with TripleStore
kb = TripleStore(url="https://dbpedia.data.dice-research.org/sparql")
# (2) Initialize a DL renderer.
renderer = DLSyntaxObjectRenderer()
# (3) Initialize a learner.
model = Drill(knowledge_base=kb, max_runtime=240) if model_name.lower() == "drill" else TDL(knowledge_base=kb)
# (4) Solve learning problems
with open("./LPs/DBpedia2022-12/lps.json") as f:
    lps = json.load(f)
for i, item in enumerate(lps):
    print("\nTarget expression: ", item["target expression"], "\n")
    lp = PosNegLPStandard(pos=set(list(map(OWLNamedIndividual,map(IRI.create, item["examples"]["positive examples"])))),
                          neg=set(list(map(OWLNamedIndividual,map(IRI.create, item["examples"]["negative examples"])))))
    # (5) Learn description logic concepts best fitting
    h = model.fit(learning_problem=lp).best_hypotheses()
    str_concept = renderer.render(h)
    print("Concept:", str_concept)  # e.g.  ∃ predecessor.WikicatPeopleFromBerlin
    # (6) Save e.g., ∃ predecessor.WikicatPeopleFromBerlin into disk
    if not os.path.exists(f"./learned_owl_expressions_{model_name}"):
        os.mkdir(f"./learned_owl_expressions_{model_name}")
    save_owl_class_expressions(expressions=h, path=f"./learned_owl_expressions_{model_name}/owl_prediction_{i}")
