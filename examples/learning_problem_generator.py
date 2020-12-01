from ontolearn import KnowledgeBase, LearningProblemGenerator
from ontolearn.refinement_operators import LengthBasedRefinement
from ontolearn.static_funcs import decompose_to_atomic, export_concepts

path_dbpedia = '../data/dbpedia_2016-10.owl'
kb = KnowledgeBase(path_dbpedia)
kb.describe()
lp = LearningProblemGenerator(knowledge_base=kb, num_problems=100, min_length=3, max_length=5)
export_concepts(kb, lp.concepts)
