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
git clone https://github.com/dice-group/Ontolearn.git 
conda create --name onto python=3.9.18 && conda activate onto && pip3 install -e . && python -c "import ontolearn"
# To download knowledge graphs
wget https://files.dice-research.org/projects/Ontolearn/KGs.zip -O ./KGs.zip && unzip KGs.zip
# To download learning problems.
wget https://files.dice-research.org/projects/Ontolearn/LPs.zip -O ./LPs.zip && unzip LPs.zip
```

```shell
pytest -p no:warnings -x # Running 158 tests takes ~ 3 mins
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

Fore more please refer to  the [examples](https://github.com/dice-group/Ontolearn/tree/develop/examples) folder.

## Benchmarking on Family dataset
```shell
# Benchmark learners on the Family benchmark dataset with benchmark learning problems.
python examples/concept_learning_evaluation.py --lps LPs/Family/lps.json --kb KGs/Family/family-benchmark_rich_background.owl --max_runtime 3 --report family_results.csv 
python -c 'import pandas as pd; print(pd.read_csv("family_results.csv", index_col=0).to_markdown(floatfmt=".3f"))'
```
<details> <summary> To see the results </summary>
Each model has 3 second to find a fitting answer. DRILL results are obtained by using F1 score as heuristic function.

| LP                 |   F1-OCEL |   RT-OCEL |   F1-CELOE |   RT-CELOE |   F1-EvoLearner |   RT-EvoLearner |   F1-DRILL |   RT-DRILL |   F1-tDL |   RT-tDL |
|:-------------------|----------:|----------:|-----------:|-----------:|----------------:|----------------:|-----------:|-----------:|---------:|---------:|
| Aunt               |     0.804 |     3.004 |      0.837 |      3.002 |           0.837 |           3.607 |      1.000 |      2.172 |    1.000 |    0.609 |
| Brother            |     1.000 |     0.013 |      1.000 |      0.003 |           1.000 |           0.021 |      1.000 |      0.342 |    1.000 |    0.656 |
| Cousin             |     0.693 |     3.069 |      0.793 |      3.001 |           0.751 |           3.510 |      0.348 |      0.293 |    1.000 |    0.917 |
| Daughter           |     1.000 |     0.011 |      1.000 |      0.004 |           1.000 |           0.027 |      1.000 |      0.480 |    1.000 |    0.737 |
| Father             |     1.000 |     0.004 |      1.000 |      0.002 |           1.000 |           0.008 |      1.000 |      0.342 |    1.000 |    0.778 |
| Granddaughter      |     1.000 |     0.003 |      1.000 |      0.001 |           1.000 |           0.006 |      1.000 |      0.314 |    1.000 |    0.607 |
| Grandfather        |     1.000 |     0.003 |      1.000 |      0.001 |           1.000 |           0.006 |      0.909 |      0.292 |    1.000 |    0.605 |
| Grandgranddaughter |     1.000 |     0.003 |      1.000 |      0.001 |           1.000 |           0.004 |      1.000 |      0.262 |    1.000 |    0.288 |
| Grandgrandfather   |     1.000 |     0.757 |      1.000 |      0.156 |           1.000 |           0.074 |      0.944 |      0.247 |    1.000 |    0.288 |
| Grandgrandmother   |     1.000 |     0.556 |      1.000 |      0.200 |           1.000 |           0.075 |      1.000 |      0.247 |    1.000 |    0.421 |
| Grandgrandson      |     1.000 |     0.512 |      1.000 |      0.163 |           1.000 |           1.818 |      0.486 |      0.266 |    1.000 |    0.359 |
| Grandmother        |     1.000 |     0.004 |      1.000 |      0.002 |           1.000 |           0.012 |      1.000 |      0.265 |    1.000 |    0.494 |
| Grandson           |     1.000 |     0.003 |      1.000 |      0.002 |           1.000 |           0.007 |      0.451 |      0.294 |    1.000 |    0.587 |
| Mother             |     1.000 |     0.004 |      1.000 |      0.002 |           1.000 |           0.009 |      1.000 |      0.271 |    1.000 |    0.808 |
| PersonWithASibling |     1.000 |     0.003 |      1.000 |      0.001 |           0.737 |           3.037 |      0.588 |      0.295 |    1.000 |    0.928 |
| Sister             |     1.000 |     0.003 |      1.000 |      0.001 |           1.000 |           0.017 |      0.800 |      0.269 |    1.000 |    0.568 |
| Son                |     1.000 |     0.004 |      1.000 |      0.002 |           1.000 |           0.008 |      0.732 |      0.269 |    1.000 |    0.689 |
| Uncle              |     0.884 |     3.007 |      0.905 |      3.001 |           0.905 |           3.516 |      0.657 |      0.219 |    1.000 |    0.527 |
</details>

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
