from ontolearn.binders import DLLearnerBinder
from owlapy.iri import IRI
from owlapy.owl_individual import OWLNamedIndividual
import json

from ontolearn.learning_problem import PosNegLPStandard

# (1) SPARQL endpoint as knowledge source: supported only by DL-Learner-1.4.0
kb_path = "https://dbpedia.data.dice-research.org/sparql"

# To download DL-learner,  https://github.com/SmartDataAnalytics/DL-Learner/releases.
dl_learner_binary_path = "./dllearner-1.4.0/bin/cli"

# (2) Read learning problem file
with open("./LPs/DBpedia2022-12/lps.json") as f:
    lps = json.load(f)
# (3) Start class expression learning
for i, item in enumerate(lps):
    print(f"\nLP {i+1}/{len(lps)} ==> Target expression: ", item["target expression"], "\n")
    lp = PosNegLPStandard(pos=set(list(map(OWLNamedIndividual,map(IRI.create, item["examples"]["positive examples"])))),
                          neg=set(list(map(OWLNamedIndividual,map(IRI.create, item["examples"]["negative examples"])))))
    
    celoe = DLLearnerBinder(binary_path=dl_learner_binary_path, kb_path=kb_path, model='celoe')
    print("\nStarting class expression learning with DL-Learner")
    best_pred_celoe = celoe.fit(lp, use_sparql=True).best_hypothesis()
    print("\nLearned expression: ", best_pred_celoe)
