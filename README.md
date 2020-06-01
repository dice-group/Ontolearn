# OntoPy

OntoPy is an open-source software library for structured machine learning in Python. OntoPy includes modules for processing knowledge bases, representation learning, inductive logic programming and ontology engineering.

- [Framework](#Framework)
    - [Core](#Knowledgebase)
        - [KnowledeBase](#Knowledgebase)
        - [Concept](#Concept)        
        - [Refinement Operator](#Refinements)
        
    - [Learners](#Learners)
        - [Differentiable Concept Learner](#dcl) (We are currently working on the idea and research paper.)
        - [Learning length of a concept](#length) (We are currently working on the idea and research paper.)
        - TODO: [Reinforcement Algorithms](#rl)
            - TODO: [Deep Q Learning](#dql)
        - TODO: [Search Algorithms](#search_algo)
            - TODO: [Breadth-First Search](#bfs)
        
- [Installation](#installation)

## Installation

```
pyton setup.py install
```

## Usage

```python
from OntoPy import KnowledgeBase, Refinement

kb = KnowledgeBase(path='../../family-benchmark_rich_background.owl')
rho = Refinement(kb)
for refs in enumerate(rho.refine(kb.thing)):
    print(refs)
```

```python
from OntoPy import KnowledgeBase,SampleConceptLearner

kb = KnowledgeBase(path='../../family-benchmark_rich_background.owl')
model = SampleConceptLearner(knowledge_base=kb,iter_bound=100,verbose=False)

p = {'http://www.benchmark.org/family#F10M173', 'http://www.benchmark.org/family#F10M183'}
n = {'http://www.benchmark.org/family#F1F5', 'http://www.benchmark.org/family#F1F7'}
model.predict(pos=p, neg=n)

model.show_best_predictions(top_n=10)
```

