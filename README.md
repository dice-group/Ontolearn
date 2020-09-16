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
pip install ontolearn
```

## Usage

### Drill- A convolutional deep Q-learning approach that prunes the search space of refinement operators for class expression learning.

```python
from ontolearn import *
import json
import random
import pandas as pd
with open('synthetic_problems.json') as json_file:
    settings = json.load(json_file)
kb = KnowledgeBase(path=settings['data_path'])
rho = LengthBasedRefinement(kb=kb)
lp_gen = LearningProblemGenerator(knowledge_base=kb,
                                  refinement_operator=rho,
                                  num_problems=3, depth=1, min_length=1)

instance_emb = pd.read_csv('../embeddings/instance_emb.csv', index_col=0)
# apply_TSNE_on_df(instance_emb)
trainer = DrillTrainer(
    knowledge_base=kb,
    refinement_operator=rho,
    quality_func=F1(),
    reward_func=Reward(),
    search_tree=SearchTreePriorityQueue(),
    path_pretrained_agent='../agent_pre_trained',  # for Incremental/Continual learning.
    learning_problem_generator=lp_gen,
    instance_embeddings=instance_emb,
    verbose=False)
trainer.start()

for str_target_concept, examples in settings['problems'].items():
    p = set(examples['positive_examples'])
    n = set(examples['negative_examples'])
    print('Target concept: ', str_target_concept)
    concepts_to_ignore = set()
    # lets inject more background info
    if str_target_concept in ['Granddaughter', 'Aunt', 'Sister']:
        concepts_to_ignore.update({'Brother', 'Father', 'Uncle', 'Grandparent'})

    model = DrillConceptLearner(knowledge_base=kb,
                                refinement_operator=rho,
                                quality_func=F1(),
                                heuristic_func=DrillHeuristic(model=trainer.model),
                                instance_emb=instance_emb,
                                search_tree=SearchTreePriorityQueue(),
                                min_length=1,  # think better variable name
                                terminate_on_goal=True,
                                iter_bound=1_000,
                                max_num_of_concepts_tested=5_000,
                                ignored_concepts={},
                                verbose=True)
    model.predict(pos=p, neg=n)
    model.best_hypotheses(top_n=10)
    model.save_best_hypotheses(file_path=str_target_concept + '_best_hypothesis.owl', rdf_format='xml', top_n=10)
```

### CELOE in Python
```python
from ontolearn import *
p = {} # set of positive instances
n = {} # set of negative instances
kb = KnowledgeBase(path='.../family-benchmark_rich_background.owl') 
model = CELOE(knowledge_base=kb,
              refinement_operator=ModifiedCELOERefinement(kb=kb),
              quality_func=F1(),
              min_horizontal_expansion=3,
              heuristic_func=CELOEHeuristic(),
              search_tree=CELOESearchTree(),
              terminate_on_goal=True,
              iter_bound=1_000,
              verbose=False)

model.predict(pos=p, neg=n)
model.show_best_predictions(top_n=10)

```

## Custom Concept Learner
```python
from ontolearn import *
p = {} # set of positive instances
n = {} # set of negative instances
kb = KnowledgeBase(path='.../family-benchmark_rich_background.owl') 
model = CustomConceptLearner(knowledge_base=kb,
                             refinement_operator=CustomRefinementOperator(kb=kb),
                             quality_func=F1(), # Precision, Recall, Accuracy
                             heuristic_func=DLFOILHeuristic(),
                             search_tree=SearchTree(),
                             terminate_on_goal=True,
                             iter_bound=1_000,
                             verbose=True)
model.predict(pos=p, neg=n)
model.show_best_predictions(top_n=10)
```

## Poor people's CI

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
