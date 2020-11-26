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
4) pip install -e .
# After you receive this Finished processing dependencies for OntoPy==0.X.X
5) python -c "import ontolearn"
```
### Installation via pip

```python
pip install ontolearn # https://pypi.org/project/ontolearn/ only a place holder.
```

## Usage
### Drill - A convolutional deep Q-learning approach for concept learning.
```python
from ontolearn import *
import json
import random
import pandas as pd

path_kb= '....' # owl
embedding_path = '...'# csv

kb = KnowledgeBase(path_kb)
rho = LengthBasedRefinement(kb=kb)
# Generate learning problems.
lp_gen = LearningProblemGenerator(knowledge_base=kb, refinement_operator=rho,
                                  num_problems=2, max_length=3)
instance_emb = pd.read_csv(embedding_path, index_col=0)

model_avg = DrillAverage(knowledge_base=kb, refinement_operator=rho,
                         instance_embeddings=instance_emb).train(lp_gen)

model_sub = DrillSample(knowledge_base=kb, refinement_operator=rho,
                        instance_embeddings=instance_emb).train(lp_gen)

# Such concepts will be ignored during the search.
concepts_to_ignore={'http://www.benchmark.org/family#Brother', 'Father', 'Grandparent'}
model_avg.fit(pos=p, neg=n, ignore=concepts_to_ignore)
model_sub.fit(pos=p, neg=n, ignore=concepts_to_ignore)

print(model_sub.best_hypotheses(n=1)[0])
print(model_avg.best_hypotheses(n=1)[0])
```

### CELOE

```python
from ontolearn import KnowledgeBase
from ontolearn.concept_learner import CELOE

p = {} # set of positive instances
n = {} # set of negative instances
kb = KnowledgeBase(path='...')
model = CELOE(knowledge_base=kb)
model.fit(pos=p, neg=n, ignore=concepts_to_ignore)
# Get Top n hypotheses
hypotheses = model.best_hypotheses(n=3)
```
## Testing

### Simple Linting and Testing

Run
```shell script
flake8
pylint
```

### Integration Testing with Docker

For testing we use [docker](https://docs.docker.com/engine/install/). 

We have a docker image [`CI/Dockerfile`](./CI/Dockerfile) which builds the package and runs all tests. 

Build it with:
```shell script
docker build -f CI/Dockerfile . --tag ontopy-test
```

Run it with:
```shell script
docker run ontopy-test
```