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
from OntoPy import KnowledgeBase, Refinement, Data

kb = KnowledgeBase(path='../../family-benchmark_rich_background.owl')
rho = Refinement(kb)
for refs in enumerate(rho.refine(kb.thing)):
    print(refs)
```

