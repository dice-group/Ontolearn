""" concept_learning.py provides an example of using OntoPy library."""
from OntoPy import KnowledgeBase,SampleConceptLearner

kb = KnowledgeBase(path='OntoPy/data/family-benchmark_rich_background.owl')
model = SampleConceptLearner(knowledge_base=kb,iter_bound=100,verbose=False)
p = {'http://www.benchmark.org/family#F10M173', 'http://www.benchmark.org/family#F10M183'}
n = {'http://www.benchmark.org/family#F1F5', 'http://www.benchmark.org/family#F1F7'}
model.predict(pos=p, neg=n)
model.show_best_predictions(top_n=10)