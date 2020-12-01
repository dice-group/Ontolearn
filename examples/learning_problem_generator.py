from ontolearn import KnowledgeBase, LearningProblemGenerator
from ontolearn.refinement_operators import LengthBasedRefinement
from ontolearn.static_funcs import export_concepts

path_dbpedia = '../data/dbpedia_2016-10.owl'
kb = KnowledgeBase(path_dbpedia)
kb.describe()
lp = LearningProblemGenerator(knowledge_base=kb, num_problems=50,
                              min_length=3, max_length=5,
                              num_of_concurrent_search=2)
export_concepts(kb, lp.concepts)
