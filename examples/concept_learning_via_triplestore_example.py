import argparse
import json

from ontolearn.concept_learner import CELOE
from ontolearn.heuristics import CELOEHeuristic
from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.learners import TDL, Drill
from ontolearn.concept_learner import EvoLearner
from ontolearn.learning_problem import PosNegLPStandard
from owlapy.owl_individual import IRI, OWLNamedIndividual
from ontolearn.refinement_operators import ModifiedCELOERefinement
from ontolearn.triple_store import TripleStore

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

    # () Define the learning problem
    with open('../LPs/Mutagenesis/lps.json') as json_file:
        settings = json.load(json_file)
    p = set(settings['problems']['NotKnown']['positive_examples'])
    n = set(settings['problems']['NotKnown']['negative_examples'])
    typed_pos = set(map(OWLNamedIndividual, map(IRI.create, p)))
    typed_neg = set(map(OWLNamedIndividual, map(IRI.create, n)))
    lp = PosNegLPStandard(pos=typed_pos, neg=typed_neg)

    # () Fit the learning problem to the model
    model.fit(lp)

    # () Retrieve and print top hypotheses
    hypotheses = list(model.best_hypotheses(n=3))
    [print(_) for _ in hypotheses]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_model', default="tdl", type=str, help='Specify the learning model you want to use.',
                        choices=["tdl", "celoe", "drill", "evolearner"])
    parser.add_argument('--url', default="http://localhost:3030/mutagenesis/sparql",
                        type=str, help='The triplestore endpoint.')

    run(parser.parse_args())
