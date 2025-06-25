import argparse
import json

# from ontolearn.concept_learner import CELOE
from ontolearn.heuristics import CELOEHeuristic
from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.learners import TDL, Drill
from ontolearn.concept_learner import EvoLearner
from ontolearn.learning_problem import PosNegLPStandard
from owlapy.owl_individual import IRI, OWLNamedIndividual
from ontolearn.refinement_operators import ModifiedCELOERefinement
from ontolearn.triple_store import TripleStore

import re
from tqdm import tqdm
import random
import itertools
import ast
import json
import time
import requests
from owlapy.owl_individual import OWLNamedIndividual, IRI
from ontolearn.learners import TDL, Drill, CELOE


"""

This is an example to show how simply you can execute a learning algorithm using the triplestore knowledge base.

Prerequisite:
- Triplestore server

For this example you can fulfill the prerequisites as follows:
- Load and launch the triplestore server following our guide.
  See https://ontolearn-docs-dice-group.netlify.app/usage/06_concept_learners#loading-and-launching-a-triplestore
- Note: The example in this script is for 'family' dataset, make the changes accordingly for the dataset you will be 
        using (for example, in this script we use 'mutagenesis'.

If you don't have the KGs or the LPs folders already, you can make use of the commands below to get them:
- wget https://files.dice-research.org/projects/Ontolearn/KGs.zip
- wget https://files.dice-research.org/projects/Ontolearn/LPs.zip

"""




def run(args):

    def run_query(sparql_query: str):
        return requests.Session().post(args.url, data={"query": sparql_query})


    def individuals_imdb(min_rate: float, max_rate: float):
            imdb_prefix = "PREFIX imdb: <http://example.org/imdb/>" #<https://www.imdb.com/>" (this for the 26GB dataset)
            xsd_prefix = "PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>" 
            
            query = f"""
            {imdb_prefix}
            {xsd_prefix}
            SELECT DISTINCT ?x WHERE {{
                ?x imdb:averageRating ?rating .
                FILTER(?rating >= {min_rate} && ?rating <= {max_rate})
            }} LIMIT 100
            """

            for binding in run_query(query).json()["results"]["bindings"]:
                yield OWLNamedIndividual(IRI.create(binding["x"]["value"]))


    # () Create a TripleStore object for the Mutagenesis dataset using the triplestore endpoint
    kb = TripleStore(url=args.url)
    # kb = KnowledgeBase(path="../KGs/Mutagenesis/mutagenesis.owl")

    assert args.learning_model in ["tdl", "celoe", "drill", "evolearner"], ("Invalid learning model, chose from "
                                                                       "[tdl, celoe, drill, evolearner]")

    # () Define the model
    if args.learning_model == "celoe":
        heuristic = CELOEHeuristic(expansionPenaltyFactor=0.05, startNodeBonus=1.0, nodeRefinementPenalty=0.01)
        op = ModifiedCELOERefinement(knowledge_base=kb, use_negation=False, use_all_constructor=False)
        model = CELOE(knowledge_base=kb, refinement_operator=op, heuristic_func=heuristic, max_runtime=30)
    elif args.learning_model == "tdl":
        model = TDL(knowledge_base=kb)
    elif args.learning_model == "drill":
        model = Drill(knowledge_base=kb)
    elif args.learning_model == "evolearner":
        model = EvoLearner(knowledge_base=kb)

    with open('LPs/IMDB/learning_problems.json', 'r') as f:
        problems_data = json.load(f)

    # Extract and iterate over the learning problems
    for idx, problem in enumerate(problems_data["problems"]):
        print(f"\nSolving LP {idx + 1} with threshold {problem['threshold']}")
        
        positives_uris = problem["positives"]
        negatives_uris = problem["negatives"]

        positives = {OWLNamedIndividual(IRI.create(uri)) for uri in positives_uris}
        negatives = {OWLNamedIndividual(IRI.create(uri)) for uri in negatives_uris}

        # Create learning problem
        lp = PosNegLPStandard(pos=positives, neg=negatives, all_instances=individuals_imdb(1.1, 2.0))

        # () Fit the learning problem to the model
        model.fit(lp)

        # () Retrieve and print top hypotheses
        hypotheses = list(model.best_hypotheses(n=3))
        [print(_) for _ in hypotheses]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_model', default="drill", type=str, help='Specify the learning model you want to use.',
                        choices=["tdl", "celoe", "drill", "evolearner"])
    parser.add_argument('--url', default="http://localhost:3040/imdb_small/sparql",
                        type=str, help='The triplestore endpoint.') #http://localhost:3030/imdb/sparql

    run(parser.parse_args())
