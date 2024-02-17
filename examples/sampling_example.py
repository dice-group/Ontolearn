import json
from ontolearn.concept_learner import EvoLearner
from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.learning_problem import PosNegLPStandard
from ontolearn.metrics import F1, Accuracy
from owlapy.model import OWLNamedIndividual, IRI
from ontolearn.utils import setup_logging
from ontosample.lpc_samplers import RandomWalkerJumpsSamplerLPCentralized
setup_logging()

with open('carcinogenesis_lp.json') as json_file:
    settings = json.load(json_file)

# Initialize kb using the file path stored at carcinogenesis_lp.json
kb = KnowledgeBase(path=settings['data_path'])

# Define learning problem
p = set(settings['positive_examples'])
n = set(settings['negative_examples'])
typed_pos = set(map(OWLNamedIndividual, map(IRI.create, p)))
typed_neg = set(map(OWLNamedIndividual, map(IRI.create, n)))
# Creating an encoded learning problem so that we can later evaluate the generated concepts from the sampled kb
encoded_lp = kb.encode_learning_problem(PosNegLPStandard(pos=typed_pos, neg=typed_neg))

sampler = RandomWalkerJumpsSamplerLPCentralized(kb, typed_pos.union(typed_neg))
sampled_kb = sampler.sample(3000)  # create a sample with 3000 individuals

# Removing removed individuals from the learning problem. May not be necessary for an LPC sampler but in case the sample
# size is less than the number of lp individuals then it is important to remove the excluded individuals from the lp set
removed_individuals = set(kb.individuals()) - set(sampled_kb.individuals())
for individual in removed_individuals:
    individual_as_str = individual.get_iri().as_str()
    if individual_as_str in p:
        p.remove(individual_as_str)
    if individual_as_str in n:
        n.remove(individual_as_str)
typed_pos = set(map(OWLNamedIndividual, map(IRI.create, p)))
typed_neg = set(map(OWLNamedIndividual, map(IRI.create, n)))
lp = PosNegLPStandard(pos=typed_pos, neg=typed_neg)

# Create our model, in this case of EvoLearner using the sampled kb
model = EvoLearner(knowledge_base=sampled_kb)
# Fitting the learning problem and finding the best hypotheses
model.fit(lp, verbose=False)
hypotheses = list(model.best_hypotheses(n=3))

# Measuring F1-score and Accuracy in the original graph using the hypotheses generated in the sampled graph.
for hypothesis in hypotheses:
    f1 = kb.evaluate_concept(hypothesis.concept, F1(), encoded_lp)
    accuracy = kb.evaluate_concept(hypothesis.concept, Accuracy(), encoded_lp)
    print(hypothesis)
    print(f'F1: {f1.q} Accuracy: {accuracy.q} \n')

print("Done!")

