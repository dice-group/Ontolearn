#from ontolearn import KnowledgeBase, LearningProblemGenerator
from ontolearn.base import KnowledgeBase
from ontolearn.learning_problem_generator import LearningProblemGenerator
from ontolearn.refinement_operators import ModifiedCELOERefinement
from ontolearn.static_funcs import export_concepts

path = '../KGs/Biopax/biopax.owl'

kb = KnowledgeBase(path=path)
lp = LearningProblemGenerator(knowledge_base=kb)
num_inds = kb.individuals_count()
concepts = lp.get_concepts(num_problems=1000,
                           num_diff_runs=10,
                           min_num_instances=int(num_inds * .1),
                           max_num_instances=int(num_inds * .8),
                           min_length=3, max_length=10)
# Each generated concept defines the type information of min 10% and max 80% of instances.
for c in concepts:
    print('*', c)
#export_concepts(kb, list(concepts), path='example_concepts.owl')
