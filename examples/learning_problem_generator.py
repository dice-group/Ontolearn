from ontolearn import KnowledgeBase, LearningProblemGenerator
from ontolearn.static_funcs import export_concepts

path_dbpedia = '../data/dbpedia_2016-10.owl'
kb = KnowledgeBase(path_dbpedia)
kb.describe()
lp = LearningProblemGenerator(knowledge_base=kb, num_problems=10_000, min_length=3, max_length=5)
export_concepts(kb, lp.concepts)

"""
Concept generation runtimes.
DBpedia onto having 762 concepts.
num_problems=100:
1. min_length = 3, max_length=5 => 33.8563  seconds
2. min_length = 3, max_length=5 => 33.4223  seconds
3. min_length = 3, max_length=5 => 33.4745  seconds

num_problems=1000:
1. min_length = 3, max_length = 5 => 33.5844  seconds
2. min_length = 3, max_length = 5 => 37.4677  second
3. min_length = 3, max_length = 5 => 37.2564  second

num_problems=10_000:
1. min_length = 3, max_length = 5 => 34.6944  seconds
2. min_length = 3, max_length = 5 => 36.4607  seconds
3. min_length = 3, max_length = 5 => 35.0695  seconds
"""