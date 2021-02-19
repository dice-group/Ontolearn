from ontolearn import KnowledgeBase, LearningProblemGenerator
from ontolearn.static_funcs import export_concepts

path = '../KGs/Biopax/biopax.owl'

kb = KnowledgeBase(path)
lp = LearningProblemGenerator(knowledge_base=kb)
concepts = lp.get_concepts(num_problems=1000,
                           num_diff_runs=10,
                           min_num_instances=int(len(kb.individuals) * .1),
                           max_num_instances=int(len(kb.individuals) * .8),
                           min_length=3, max_length=10)
# Each generated concept defines the type information of min 10% and max 80% of instances.
export_concepts(kb, list(concepts), path='example_concepts.owl')
