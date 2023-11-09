import json

from ontolearn.concept_learner import CELOE
from ontolearn.heuristics import CELOEHeuristic
from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.learning_problem import PosNegLPStandard
from ontolearn.owlapy.model import IRI, OWLNamedIndividual
from ontolearn.refinement_operators import ModifiedCELOERefinement

"""

This is an example to show how simply you can execute a learning algorithm having a knowledge base that uses a 
triplestore.

Prerequisite:
- Server hosting the dataset as a triplestore

For this example you can fulfill the prerequisites as follows:
- Load and launch the triplestore server following our guide.
  See https://ontolearn-docs-dice-group.netlify.app/usage/06_concept_learners#loading-and-launching-a-triplestore
- Note: The example in this script is for 'family' dataset, make the changes accordingly when setting up the triplestore 
  server.
  
"""

# Create a knowledge base object for the Family dataset using the URL address of the triplestore host only
kb = KnowledgeBase(triplestore_address="http://localhost:3030/family/sparql")

# Define the model
heur = CELOEHeuristic(expansionPenaltyFactor=0.05, startNodeBonus=1.0, nodeRefinementPenalty=0.01)
op = ModifiedCELOERefinement(knowledge_base=kb, use_negation=False, use_all_constructor=False)
model = CELOE(knowledge_base=kb, refinement_operator=op, heuristic_func=heur)

# Define a learning problem
with open('synthetic_problems.json') as json_file:
    settings = json.load(json_file)
p = set(settings['problems']['Uncle']['positive_examples'])
n = set(settings['problems']['Uncle']['negative_examples'])
typed_pos = set(map(OWLNamedIndividual, map(IRI.create, p)))
typed_neg = set(map(OWLNamedIndividual, map(IRI.create, n)))
lp = PosNegLPStandard(pos=typed_pos, neg=typed_neg)

# Fit the learning problem to the model
model.fit(lp)

# Retrieve and print top hypotheses
hypotheses = list(model.best_hypotheses(n=3))
[print(_) for _ in hypotheses]
