import os

from experiments_standard import ClosedWorld_ReasonerFactory
from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.learning_problem_generator import LearningProblemGenerator
from ontolearn.utils import setup_logging

setup_logging("logging_test.conf")

try:
    os.chdir("examples")
except FileNotFoundError:
    pass

path = '../KGs/Biopax/biopax.owl'

# kb = KnowledgeBase(path=path, reasoner_factory=OWLReasoner_Owlready2_TempClasses)
kb = KnowledgeBase(path=path, reasoner_factory=ClosedWorld_ReasonerFactory)
lp = LearningProblemGenerator(knowledge_base=kb)
num_inds = kb.individuals_count()
concepts = list(lp.get_concepts(num_problems=5000,
                                num_diff_runs=10,
                                min_num_instances=int(2),
                                max_num_instances=int(num_inds * .95),
                                min_length=4, max_length=40))
# Each generated concept defines the type information of min 10% and max 80% of instances.
# for c in concepts:
#     print('*', c)

lp.export_concepts(concepts, path='example_concepts')
