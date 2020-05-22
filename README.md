# OntoPy

OntoPy is an open-source software library for structured machine learning in Python. OntoPy includes modules for processing knowledge bases, representation learning, inductive logic programming and ontology engineering.

- [Framework](#Framework)
    - [Knowlede Base](#Knowledgebase)
    - [Refinement Operator](#Refinements)
    - [Learners](#Learners)
        - [Reinforcement Algorithms](#rl)
        - [Search Algorithms](#search_algo)
        
- [Installation](#installation)

## Main Dependencies
- [Owlready2](https://owlready2.readthedocs.io)
- [scikit-learn](https://scikit-learn.org)

## Installation

Currently
```
conda env create -n conda-env -f environment.yml
```
Later.
```
pip install ontopy
```

## Usage

```python
from ontopy import KnowledgeBase, Refinement

kb = KnowledgeBase(path='data/family-benchmark_rich_background.owl')
rho = Refinement(kb)
for refs in enumerate(rho.refine(kb.thing)):
    print(refs)
```

