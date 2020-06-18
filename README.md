# Ontolearn

Ontolearn is an open-source software library for structured machine learning in Python. Ontolearn includes modules for processing knowledge bases, representation learning, inductive logic programming and ontology engineering.

- [Framework](#Framework)
    - [KnowledeBase](#Knowledgebase)
    - [Concept](#Concept)        
    - [Refinement Operator](#Refinements)
            
- [Installation](#installation)

## Current Installation
```
1) git clone https://github.com/dice-group/OntoPy.git
2) conda create -n temp python=3.6.2 # Or be sure that your have Python => 3.6.
3) conda activate temp
4) python ontolearn/setup.py install
# After you receive this Finished processing dependencies for OntoPy==0.0.1
5) python -c "import ontolearn"
```
## Later Installation

```python
pip install ontolearn
```

## Usage

```python
from ontolearn import KnowledgeBase, Refinement

kb = KnowledgeBase(path='data/family-benchmark_rich_background.owl')
rho = Refinement(kb)
for refs in enumerate(rho.refine(kb.thing)):
    print(refs)
```

```python
from ontolearn import KnowledgeBase,SampleConceptLearner
kb = KnowledgeBase(path='data/family-benchmark_rich_background.owl')
model = SampleConceptLearner(knowledge_base=kb,iter_bound=100,verbose=False)
p = {'http://www.benchmark.org/family#F10M173', 'http://www.benchmark.org/family#F10M183'}
n = {'http://www.benchmark.org/family#F1F5', 'http://www.benchmark.org/family#F1F7'}
model.predict(pos=p, neg=n)
model.show_best_predictions(top_n=10)
```



```python
from ontolearn import KnowledgeBase,SampleConceptLearner
from ontolearn.metrics import F1, PredictiveAccuracy, CELOEHeuristic,DLFOILHeuristic
kb = KnowledgeBase(path='data/family-benchmark_rich_background.owl')

p = {'http://www.benchmark.org/family#F10M173', 'http://www.benchmark.org/family#F10M183'}
n = {'http://www.benchmark.org/family#F1F5', 'http://www.benchmark.org/family#F1F7'}

model = SampleConceptLearner(knowledge_base=kb,
                             quality_func=F1(),
                             terminate_on_goal=True,
                             heuristic_func=DLFOILHeuristic(),
                             iter_bound=100,
                             verbose=False)

model.predict(pos=p, neg=n)
model.show_best_predictions(top_n=10)

######################################################################
model = SampleConceptLearner(knowledge_base=kb,
                             quality_func=PredictiveAccuracy(),
                             terminate_on_goal=True,
                             heuristic_func=CELOEHeuristic(),
                             iter_bound=100,
                             verbose=False)

model.predict(pos=p, neg=n)
model.show_best_predictions(top_n=10)
```

