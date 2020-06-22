# Ontolearn

Ontolearn is an open-source software library for structured machine learning in Python. Ontolearn includes modules for processing knowledge bases, representation learning, inductive logic programming and ontology engineering.

- [Framework](#Framework)
    - [KnowledeBase](#Knowledgebase)
    - [Concept](#Concept)        
    - [Refinement Operator](#Refinements)
            
- [Installation](#installation)

# Installation
### Installation from source
```
1) git clone https://github.com/dice-group/OntoPy.git
2) conda create -n temp python=3.6.2 # Or be sure that your have Python => 3.6.
3) conda activate temp
4) python ontolearn/setup.py install
# After you receive this Finished processing dependencies for OntoPy==0.0.1
5) python -c "import ontolearn"
```
### Installation via pip

```python
pip install ontolearn
```

## Usage

```python
from ontolearn import KnowledgeBase, CustomRefinementOperator
kb = KnowledgeBase(path='../data/family-benchmark_rich_background.owl')
rho = CustomRefinementOperator(kb)
for refs in enumerate(rho.refine(kb.thing)):
    print(refs)
```

```python
from ontolearn import KnowledgeBase,CustomConceptLearner,CustomRefinementOperator,F1,DLFOILHeuristic,SearchTree
import json

with open('synthetic_problems.json') as json_file:
    settings = json.load(json_file)

kb = KnowledgeBase(path=settings['data_path'])

for str_target_concept, examples in settings['problems'].items():
    p = set(examples['positive_examples'])
    n = set(examples['negative_examples'])
    print('Target concept: ', str_target_concept)
    model = CustomConceptLearner(knowledge_base=kb,
                                 refinement_operator=CustomRefinementOperator(kb=kb),
                                 quality_func=F1(),
                                 heuristic_func=DLFOILHeuristic(),
                                 search_tree=SearchTree(),
                                 terminate_on_goal=True,
                                 iter_bound=1_000,
                                 verbose=True)

    model.predict(pos=p, neg=n)
    model.show_best_predictions(top_n=10)

```



```python
from ontolearn import *
import json

with open('synthetic_problems.json') as json_file:
    settings = json.load(json_file)

kb = KnowledgeBase(path=settings['data_path'])

for str_target_concept, examples in settings['problems'].items():
    p = set(examples['positive_examples'])
    n = set(examples['negative_examples'])
    print('Target concept: ',str_target_concept)
    model = CELOE(knowledge_base=kb,
                  refinement_operator=ModifiedCELOERefinement(kb=kb),
                  quality_func=F1(),
                  min_horiziontal_expansion=0,
                  heuristic_func=CELOEHeuristic(),
                  search_tree=CELOESearchTree(),
                  terminate_on_goal=True,
                  iter_bound=1_000,
                  verbose=False)

    model.predict(pos=p, neg=n)
    model.show_best_predictions(top_n=10)
```

