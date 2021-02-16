#from ontolearn import KnowledgeBase, LearningProblemGenerator
from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.learning_problem_generator import LearningProblemGenerator
from ontolearn.owlapy.owlready2.temp_classes import OWLReasoner_Owlready2_TempClasses
from ontolearn.refinement_operators import ModifiedCELOERefinement
from ontolearn.static_funcs import export_concepts

path = '../KGs/Biopax/biopax.owl'

#kb = KnowledgeBase(path=path, reasoner_factory=OWLReasoner_Owlready2_TempClasses)
kb = KnowledgeBase(path=path)
lp = LearningProblemGenerator(knowledge_base=kb)
num_inds = kb.individuals_count()
concepts = lp.get_concepts(num_problems=1000,
                           num_diff_runs=10,
                           min_num_instances=int(1),
                           max_num_instances=int(num_inds * .9),
                           min_length=3, max_length=40)
# Each generated concept defines the type information of min 10% and max 80% of instances.
for c in concepts:
    print('*', c)
#export_concepts(kb, list(concepts), path='example_concepts.owl')
