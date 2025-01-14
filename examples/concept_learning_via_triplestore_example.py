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
- Server hosting the dataset as a triplestore

For this example you can fulfill the prerequisites as follows:
- Load and launch the triplestore server following our guide.
  See https://ontolearn-docs-dice-group.netlify.app/usage/06_concept_learners#loading-and-launching-a-triplestore
- Note: The example in this script is for 'family' dataset, make the changes accordingly when setting up the triplestore 
  server.
  
"""

# Create a knowledge base object for the Family dataset using the URL address of the triplestore host only
kb = TripleStore(url="http://localhost:3030/mutagenesis/sparql")
# kb = KnowledgeBase(path="../KGs/Mutagenesis/mutagenesis.owl")

# Define the model
heur = CELOEHeuristic(expansionPenaltyFactor=0.05, startNodeBonus=1.0, nodeRefinementPenalty=0.01)
op = ModifiedCELOERefinement(knowledge_base=kb, use_negation=False, use_all_constructor=False)
model = CELOE(knowledge_base=kb, refinement_operator=op, heuristic_func=heur, max_runtime=30)
# model = TDL(knowledge_base=kb)
# model = EvoLearner(knowledge_base=kb)
# Define a learning problem
with open('../LPs/Mutagenesis/lps.json') as json_file:
    settings = json.load(json_file)
p = set(settings['problems']['NotKnown']['positive_examples'])
n = set(settings['problems']['NotKnown']['negative_examples'])
typed_pos = set(map(OWLNamedIndividual, map(IRI.create, p)))
typed_neg = set(map(OWLNamedIndividual, map(IRI.create, n)))
lp = PosNegLPStandard(pos=typed_pos, neg=typed_neg)

# Fit the learning problem to the model
model.fit(lp)

# Retrieve and print top hypotheses
hypotheses = list(model.best_hypotheses(n=3))
[print(_) for _ in hypotheses]
