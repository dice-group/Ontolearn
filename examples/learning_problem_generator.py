from ontolearn import KnowledgeBase, LearningProblemGenerator
from ontolearn.static_funcs import export_concepts

path_dbpedia = '../data/dbpedia_2016-10.owl'
kb = KnowledgeBase(path_dbpedia)
kb.describe()
lp = LearningProblemGenerator(knowledge_base=kb)
concepts = lp.get_concepts(num_problems=10_000, min_length=3, max_length=5)
export_concepts(kb, concepts, path='example_concepts.owl')
