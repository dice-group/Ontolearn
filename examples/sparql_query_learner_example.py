import json
import os
import random

from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.concept_learner import CELOE
from ontolearn.heuristics import CELOEHeuristic
from ontolearn.learners.sparql_query_learner import SPARQLQueryLearner
from ontolearn.learning_problem import PosNegLPStandard
from ontolearn.lp_generator import LPGen
from ontolearn.metrics import Accuracy, F1
from owlapy.owl_individual import OWLNamedIndividual, IRI
from ontolearn.refinement_operators import ModifiedCELOERefinement

try:
    os.chdir("examples")
except FileNotFoundError:
    pass

if not os.path.exists("PremierLeague"):
    print("Please prepare the data required by this example")
    print("1. Download and extract the data (in the examples directory): `wget https://files.dice-research.org/projects/Ontolearn/PremierLeague.zip && unzip PremierLeague.zip`")
    print("2. Load the knowledge graph found at `PremierLeague/premierleague-hermit-inf.nt` to the triplestore")
    exit(1)

# premier league
kb_file="./PremierLeague/premierleague-hermit-inf.owl"
endpoint_url="http://127.0.0.1:9080/sparql" # the url of the endpoint (change accordingly, the data need to be already loaded)
lps_file="./PremierLeague/LPs/LPs.json"
namespace="http://dl-learner.org/res/"


kb = KnowledgeBase(path=kb_file)

# is this needed for CELOE?
random.seed(0)


# noinspection DuplicatedCode
with open(lps_file, "r") as lp:
    # read LPs file
    lps_json = json.loads(lp.read())
    lp_count = 0
    # iterate over the learning problems
    for target_concept, lp_json in lps_json.items():
        lp_count += 1
        print("######################################")
        print("Learning Problem: {} (Target Concept: {})".format(lp_count, target_concept))
        print("######################################")
        # prepare the positive and negative examples
        p = set(lp_json['positive examples'])
        n = set(lp_json['negative examples'])
        typed_pos = set()
        typed_neg = set()
        for i in p:
            typed_pos.add(OWLNamedIndividual(IRI.create(namespace, i)))
        for i in n:
            typed_neg.add(OWLNamedIndividual(IRI.create(namespace, i)))
        lp = PosNegLPStandard(pos=typed_pos, neg=typed_neg)

        # prepare CELOE
        qual = F1()
        heur = CELOEHeuristic(expansionPenaltyFactor=0.05, startNodeBonus=1.0, nodeRefinementPenalty=0.01)
        op = ModifiedCELOERefinement(knowledge_base=kb,
                                     use_negation=False,
                                     use_card_restrictions=False,
                                     use_numeric_datatypes = False,
                                     use_time_datatypes = False,
                                     use_boolean_datatype = False)
        model = CELOE(knowledge_base=kb,
                      max_runtime=10,
                      refinement_operator=op,
                      quality_func=qual,
                      heuristic_func=heur,
                      max_num_of_concepts_tested=10_000_000_000,
                      iter_bound=10_000_000_000)
        try:
            model.fit(lp)
        except Exception as e:
            print("Skipping learning problem (Exception was thrown)")
            continue

        # Get the top 3 hypotheses found by CELOE
        hypotheses = list(model.best_hypotheses(n=3))
        print("--------------------------------------")
        print("CELOE Results (Top 3)")
        [print(str(hyp)) for hyp in hypotheses]
        print("--------------------------------------")
        # CELOE has finished, proceed with SPARQL learner
        # create the SPARQL learner by providing the endpoint's URL and the learning problem
        sparql_learner = SPARQLQueryLearner(endpoint_url=endpoint_url, learning_problem=lp)
        # the SPARQL learner will try to improve all concepts returned by CELOE (top 3)
        for hyp in hypotheses:
            sparql_learner.learn_sparql_query(ce=hyp)
