# Ontolearn

*Ontolearn* is an open-source software library for description logic learning problem.
Find more in the [Documentation](https://ontolearn-docs-dice-group.netlify.app/usage/01_introduction).

Learning algorithms: 
- **Drill** &rarr; [Neuro-Symbolic Class Expression Learning](https://www.ijcai.org/proceedings/2023/0403.pdf)
- **EvoLearner** &rarr; [EvoLearner: Learning Description Logics with Evolutionary Algorithms](https://dl.acm.org/doi/abs/10.1145/3485447.3511925)
- **NCES2** &rarr; (soon) [Neural Class Expression Synthesis in ALCHIQ(D)](https://papers.dice-research.org/2023/ECML_NCES2/NCES2_public.pdf)
- **NCES** &rarr; [Neural Class Expression Synthesis](https://link.springer.com/chapter/10.1007/978-3-031-33455-9_13) 
- **NERO** &rarr; (soon) [Learning Permutation-Invariant Embeddings for Description Logic Concepts](https://link.springer.com/chapter/10.1007/978-3-031-30047-9_9)
- **CLIP** &rarr; (soon) [Learning Concept Lengths Accelerates Concept Learning in ALC](https://link.springer.com/chapter/10.1007/978-3-031-06981-9_14)
- **CELOE** &rarr; [Class Expression Learning for Ontology Engineering](https://www.sciencedirect.com/science/article/abs/pii/S1570826811000023)
- **OCEL** &rarr; A limited version of CELOE

## Installation

```shell
pip install ontolearn 
```
or
```shell
git clone https://github.com/dice-group/Ontolearn.git && conda create --name onto python=3.8 && conda activate onto 
pip3 install -r requirements.txt && python -c "import ontolearn"
wget https://files.dice-research.org/projects/Ontolearn/KGs.zip -O ./KGs.zip && unzip KGs.zip
python -m pytest tests
```
```shell
pytest -p no:warnings -x # Runs > tests leading to > 15 mins
```

## Description Logic Concept Learning 
```python
from ontolearn.concept_learner import CELOE
from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.learning_problem import PosNegLPStandard
from ontolearn.search import EvoLearnerNode
from owlapy.model import OWLClass, OWLClassAssertionAxiom, OWLNamedIndividual, IRI, OWLObjectProperty, OWLObjectPropertyAssertionAxiom
from owlapy.render import DLSyntaxObjectRenderer
# (1) Load a knowledge graph.
kb = KnowledgeBase(path='KGs/father.owl')
# (2) Initialize a learner.
model = CELOE(knowledge_base=kb)
# (3) Define a description logic concept learning problem.
lp = PosNegLPStandard(pos={OWLNamedIndividual(IRI.create("http://example.com/father#stefan")),
                           OWLNamedIndividual(IRI.create("http://example.com/father#markus")),
                           OWLNamedIndividual(IRI.create("http://example.com/father#martin"))},
                      neg={OWLNamedIndividual(IRI.create("http://example.com/father#heinz")),
                           OWLNamedIndividual(IRI.create("http://example.com/father#anna")),
                           OWLNamedIndividual(IRI.create("http://example.com/father#michelle"))})
# (4) Learn description logic concepts best fitting (3).
dl_classifiers=model.fit(learning_problem=lp).best_hypotheses(2)

# (5) Inference over unseen individuals
namespace = 'http://example.com/father#'
# (6) New Individuals
julia = OWLNamedIndividual(IRI.create(namespace, 'julia'))
julian = OWLNamedIndividual(IRI.create(namespace, 'julian'))
thomas = OWLNamedIndividual(IRI.create(namespace, 'thomas'))
# (7) OWLClassAssertionAxiom  about (6)
male = OWLClass(IRI.create(namespace, 'male'))
female = OWLClass(IRI.create(namespace, 'female'))
axiom1 = OWLClassAssertionAxiom(individual=julia, class_expression=female)
axiom2 = OWLClassAssertionAxiom(individual=julian, class_expression=male)
axiom3 = OWLClassAssertionAxiom(individual=thomas, class_expression=male)
# (8) OWLObjectPropertyAssertionAxiom about (6)
has_child = OWLObjectProperty(IRI.create(namespace, 'hasChild'))
# Existing Individuals
anna = OWLNamedIndividual(IRI.create(namespace, 'anna'))
markus = OWLNamedIndividual(IRI.create(namespace, 'markus'))
michelle = OWLNamedIndividual(IRI.create(namespace, 'michelle'))
axiom4 = OWLObjectPropertyAssertionAxiom(subject=thomas, property_=has_child, object_=julian)
axiom5 = OWLObjectPropertyAssertionAxiom(subject=julia, property_=has_child, object_=julian)

# 4. Use loaded class expressions for predictions
predictions = model.predict(individuals=[julia, julian, thomas, anna, markus, michelle],
                            axioms=[axiom1, axiom2, axiom3, axiom4, axiom5],
                            hypotheses=dl_classifiers)
print(predictions)
"""
          (¬female) ⊓ (∃ hasChild.⊤)  male
julia                            0.0   0.0
julian                           0.0   1.0
thomas                           1.0   1.0
anna                             0.0   0.0
markus                           1.0   1.0
michelle                         0.0   0.0
"""
```

We have already many algorithms ready to use
```python
from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.concept_learner import CELOE, OCEL, EvoLearner, Drill
from ontolearn.binders import DLLearnerBinder
from ontolearn.learning_problem import PosNegLPStandard
from ontolearn.metrics import F1
from owlapy.model import OWLNamedIndividual, IRI
from ontolearn.utils import setup_logging

setup_logging()
max_runtime, topk=5, 1
kb = KnowledgeBase(path="KGs/Family/family-benchmark_rich_background.owl")
lp = PosNegLPStandard(pos={OWLNamedIndividual(IRI.create(p)) for p in
                           {"http://www.benchmark.org/family#F10F175",
                            "http://www.benchmark.org/family#F10F177"}},
                      neg={OWLNamedIndividual(IRI.create("http://www.benchmark.org/family#F9M142"))})
preds_drill = list(Drill(knowledge_base=kb, quality_func=F1(), max_runtime=max_runtime).fit(lp).best_hypotheses(n=topk))
preds_dl_celoe = list(DLLearnerBinder(binary_path="examples/dllearner-1.5.0/bin/cli",
                                      kb_path="KGs/Family/family-benchmark_rich_background.owl", model='celoe',
                                      max_runtime=max_runtime).fit(lp).best_hypotheses(n=topk))
preds_evo = list(EvoLearner(knowledge_base=kb, quality_func=F1(), max_runtime=max_runtime).fit(lp,verbose=0).best_hypotheses(n=topk))
preds_celoe = list(CELOE(knowledge_base=kb, quality_func=F1(), max_runtime=max_runtime).fit(lp).best_hypotheses(n=topk))
preds_ocel = list(OCEL(knowledge_base=kb, quality_func=F1(), max_runtime=max_runtime).fit(lp).best_hypotheses(n=topk))

print("RESULTS")
print(f"DRILL:{preds_drill[0]}")
print(f"EvoLearner:{preds_celoe[0]}")
print(f"CELOE:{preds_celoe[0]}")
print(f"OCEL:{preds_ocel[0]}")
print(f"DL-CELOE:{preds_dl_celoe[0]}")
```


Fore more please refer to  the [examples](https://github.com/dice-group/Ontolearn/tree/develop/examples) folder.


## Deployment 

```shell
pip install gradio
```

To deploy **EvoLearner** on the **Family** knowledge graph. Available models to deploy: **EvoLearner**, **NCES**, **CELOE** and **OCEL**.
```shell
python deploy_cl.py --model evolearner --path_knowledge_base KGs/Family/family-benchmark_rich_background.owl
```
Run the help command to see the description on this script usage:

```shell
python deploy_cl.py --help
```

### Citing
Currently, we are working on our manuscript describing our framework. 
If you find our work useful in your research, please consider citing the respective paper:
```
# DRILL
@inproceedings{demir2023drill,
  author = {Demir, Caglar and Ngomo, Axel-Cyrille Ngonga},
  booktitle = {The 32nd International Joint Conference on Artificial Intelligence, IJCAI 2023},
  title = {Neuro-Symbolic Class Expression Learning},
  url = {https://www.ijcai.org/proceedings/2023/0403.pdf},
 year={2023}
}

# NCES2
@inproceedings{kouagou2023nces2,
author={Kouagou, N'Dah Jean and Heindorf, Stefan and Demir, Caglar and Ngonga Ngomo, Axel-Cyrille},
title={Neural Class Expression Synthesis in ALCHIQ(D)},
url = {https://papers.dice-research.org/2023/ECML_NCES2/NCES2_public.pdf},
booktitle={Machine Learning and Knowledge Discovery in Databases},
year={2023},
publisher={Springer Nature Switzerland},
address="Cham"
}

# NCES
@inproceedings{kouagou2023neural,
  title={Neural class expression synthesis},
  author={Kouagou, N’Dah Jean and Heindorf, Stefan and Demir, Caglar and Ngonga Ngomo, Axel-Cyrille},
  booktitle={European Semantic Web Conference},
  pages={209--226},
  year={2023},
  publisher={Springer Nature Switzerland}
}

# EvoLearner
@inproceedings{heindorf2022evolearner,
  title={Evolearner: Learning description logics with evolutionary algorithms},
  author={Heindorf, Stefan and Bl{\"u}baum, Lukas and D{\"u}sterhus, Nick and Werner, Till and Golani, Varun Nandkumar and Demir, Caglar and Ngonga Ngomo, Axel-Cyrille},
  booktitle={Proceedings of the ACM Web Conference 2022},
  pages={818--828},
  year={2022}
}


# CLIP
@inproceedings{kouagou2022learning,
  title={Learning Concept Lengths Accelerates Concept Learning in ALC},
  author={Kouagou, N’Dah Jean and Heindorf, Stefan and Demir, Caglar and Ngonga Ngomo, Axel-Cyrille},
  booktitle={European Semantic Web Conference},
  pages={236--252},
  year={2022},
  publisher={Springer Nature Switzerland}
}
```

In case you have any question, please contact:  ```onto-learn@lists.uni-paderborn.de```
