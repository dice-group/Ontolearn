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
python -m venv venv && source venv/bin/activate # for Windows use: .\venv\Scripts\activate 
pip install -r requirements.txt
wget https://files.dice-research.org/projects/Ontolearn/KGs.zip -O ./KGs.zip && unzip KGs.zip
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

## Benchmark Results
```shell
# To download learning problems. # Benchmark learners on the Family benchmark dataset with benchmark learning problems.
wget https://files.dice-research.org/projects/Ontolearn/LPs.zip -O ./LPs.zip && unzip LPs.zip
```

```shell
# To download learning problems. # Benchmark learners on the Family benchmark dataset with benchmark learning problems.
python examples/concept_learning_evaluation.py --lps LPs/Family/lps.json --kb KGs/Family/family-benchmark_rich_background.owl --max_runtime 60 --report family_results.csv  && python -c 'import pandas as pd; print(pd.read_csv("family_results.csv", index_col=0).to_markdown(floatfmt=".3f"))'
```
<details> <summary> To see the results </summary>
Each model has 60 second to find a fitting answer. DRILL results are obtained by using F1 score as heuristic function.

### Family Benchmark Results

| LP                 |   F1-OCEL |   RT-OCEL |   F1-CELOE |   RT-CELOE |   F1-Evo |   RT-Evo |   F1-DRILL |   RT-DRILL |   F1-TDL |   RT-TDL |
|:-------------------|----------:|----------:|-----------:|-----------:|---------:|---------:|-----------:|-----------:|---------:|---------:|
| Aunt               |     0.837 |    13.393 |      0.911 |      7.194 |    1.000 |    1.341 |      0.921 |     60.046 |    1.000 |    0.207 |
| Brother            |     1.000 |     0.030 |      1.000 |      0.008 |    1.000 |    0.344 |      1.000 |      0.076 |    1.000 |    0.187 |
| Cousin             |     0.721 |    10.756 |      0.793 |      8.396 |    0.348 |    0.390 |      0.861 |     60.068 |    1.000 |    0.231 |
| Daughter           |     1.000 |     0.026 |      1.000 |      0.008 |    1.000 |    0.358 |      1.000 |      0.122 |    1.000 |    0.210 |
| Father             |     1.000 |     0.004 |      1.000 |      0.002 |    1.000 |    0.419 |      1.000 |      0.008 |    1.000 |    0.213 |
| Granddaughter      |     1.000 |     0.004 |      1.000 |      0.003 |    1.000 |    0.297 |      1.000 |      0.006 |    1.000 |    0.188 |
| Grandfather        |     1.000 |     0.003 |      1.000 |      0.001 |    0.909 |    0.250 |      1.000 |      0.005 |    1.000 |    0.189 |
| Grandgranddaughter |     1.000 |     0.004 |      1.000 |      0.001 |    1.000 |    0.264 |      1.000 |      0.003 |    1.000 |    0.286 |
| Grandgrandfather   |     1.000 |     0.720 |      1.000 |      0.153 |    0.944 |    0.220 |      1.000 |      0.473 |    1.000 |    0.163 |
| Grandgrandmother   |     1.000 |     2.273 |      1.000 |      0.197 |    0.200 |    0.237 |      1.000 |      0.478 |    1.000 |    0.304 |
| Grandgrandson      |     1.000 |     1.801 |      1.000 |      0.159 |    1.000 |    0.277 |      1.000 |      0.600 |    1.000 |    0.174 |
| Grandmother        |     1.000 |     0.008 |      1.000 |      0.002 |    1.000 |    0.270 |      1.000 |      0.007 |    1.000 |    0.191 |
| Grandson           |     1.000 |     0.003 |      1.000 |      0.002 |    1.000 |    0.398 |      1.000 |      0.006 |    1.000 |    0.196 |
| Mother             |     1.000 |     0.004 |      1.000 |      0.002 |    0.510 |    0.287 |      1.000 |      0.008 |    1.000 |    0.220 |
| PersonWithASibling |     1.000 |     0.003 |      1.000 |      0.001 |    0.700 |    0.314 |      0.737 |     60.060 |    1.000 |    0.224 |
| Sister             |     1.000 |     0.003 |      1.000 |      0.001 |    0.800 |    0.269 |      1.000 |      0.051 |    1.000 |    0.197 |
| Son                |     1.000 |     0.004 |      1.000 |      0.002 |    0.556 |    0.273 |      1.000 |      0.008 |    1.000 |    0.209 |
| Uncle              |     0.905 |    17.389 |      0.905 |      6.406 |    0.483 |    0.232 |      0.950 |     60.051 |    1.000 |    0.199 |


### Mutagenesis Benchmark Results
```shell
python examples/concept_learning_evaluation.py --lps LPs/Mutagenesis/lps.json --kb KGs/Mutagenesis/mutagenesis.owl --max_runtime 60 --report mutagenesis_results.csv && python -c 'import pandas as pd; print(pd.read_csv("mutagenesis_results.csv", index_col=0).to_markdown(floatfmt=".3f"))'
```
| LP       |   F1-OCEL |   RT-OCEL |   F1-CELOE |   RT-CELOE |   F1-Evo |   RT-Evo |   F1-DRILL |   RT-DRILL |   F1-TDL |   RT-TDL |
|:---------|----------:|----------:|-----------:|-----------:|---------:|---------:|-----------:|-----------:|---------:|---------:|
| NotKnown |     0.916 |    60.226 |      0.916 |     41.243 |    0.976 |   40.411 |      0.704 |     60.044 |    1.000 |   49.022 |

### Carcinogenesis Benchmark Results
```shell
python examples/concept_learning_evaluation.py --lps LPs/Carcinogenesis/lps.json --kb KGs/Carcinogenesis/carcinogenesis.owl --max_runtime 60 --report carcinogenesis_results.csv  && python -c 'import pandas as pd; print(pd.read_csv("carcinogenesis_results.csv", index_col=0).to_markdown(floatfmt=".3f"))'
```

| LP       |   F1-OCEL |   RT-OCEL |   F1-CELOE |   RT-CELOE |   F1-Evo |   RT-Evo |   F1-DRILL |   RT-DRILL |   F1-TDL |   RT-TDL |
|:---------|----------:|----------:|-----------:|-----------:|---------:|---------:|-----------:|-----------:|---------:|---------:|
| NOTKNOWN |     0.739 |    64.975 |      0.739 |     60.004 |    0.814 |   60.758 |      0.705 |     60.066 |    1.000 |   56.701 |

</details>

## Deployment 

```shell
pip install gradio # (check `pip show gradio` first)
```

Available models to deploy: **EvoLearner**, **NCES**, **CELOE** and **OCEL**.
To deploy **EvoLearner** on the **Family** knowledge graph:
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
