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
# To download learning problems. # Benchmark learners on the Family benchmark dataset with benchmark learning problems.
wget https://files.dice-research.org/projects/Ontolearn/LPs.zip -O ./LPs.zip && unzip LPs.zip
```

```shell
# To download learning problems. # Benchmark learners on the Family benchmark dataset with benchmark learning problems.
python examples/concept_learning_evaluation.py --lps LPs/Family/lps.json --kb KGs/Family/family-benchmark_rich_background.owl --max_runtime 60 --report family_results.csv 
python -c 'import pandas as pd; print(pd.read_csv("family_results.csv", index_col=0).to_markdown(floatfmt=".3f"))'
```
<details> <summary> To see the results </summary>
Each model has 60 second to find a fitting answer. DRILL results are obtained by using F1 score as heuristic function.

### Family Benchmark Results

| LP                 |   F1-OCEL |   RT-OCEL |   F1-CELOE |   RT-CELOE |   F1-EvoLearner |   RT-EvoLearner |   F1-DRILL |   RT-DRILL |   F1-tDL |   RT-tDL |
|:-------------------|----------:|----------:|-----------:|-----------:|----------------:|----------------:|-----------:|-----------:|---------:|---------:|
| Aunt               |     0.837 |    13.737 |      0.911 |      7.238 |           0.882 |         105.142 |      1.000 |      1.567 |    1.000 |    0.343 |
| Brother            |     1.000 |     0.030 |      1.000 |      0.007 |           1.000 |           0.186 |      1.000 |      0.341 |    1.000 |    0.193 |
| Cousin             |     0.721 |    11.565 |      0.793 |     10.104 |           0.831 |          98.884 |      0.348 |      0.334 |    1.000 |    0.232 |
| Daughter           |     1.000 |     0.025 |      1.000 |      0.008 |           1.000 |           0.324 |      1.000 |      0.587 |    1.000 |    0.335 |
| Father             |     1.000 |     0.005 |      1.000 |      0.002 |           1.000 |           0.010 |      1.000 |      0.400 |    1.000 |    0.246 |
| Granddaughter      |     1.000 |     0.003 |      1.000 |      0.001 |           1.000 |           0.006 |      1.000 |      0.310 |    1.000 |    0.217 |
| Grandfather        |     1.000 |     0.003 |      1.000 |      0.001 |           1.000 |           0.005 |      1.000 |      0.266 |    1.000 |    0.196 |
| Grandgranddaughter |     1.000 |     0.003 |      1.000 |      0.001 |           1.000 |           0.003 |      1.000 |      0.279 |    1.000 |    0.178 |
| Grandgrandfather   |     1.000 |     0.775 |      1.000 |      0.164 |           1.000 |           0.732 |      0.944 |      0.220 |    1.000 |    0.176 |
| Grandgrandmother   |     1.000 |     2.458 |      1.000 |      0.202 |           1.000 |           0.714 |      0.000 |      0.239 |    1.000 |    0.173 |
| Grandgrandson      |     1.000 |     0.547 |      1.000 |      0.165 |           1.000 |           0.610 |      0.486 |      0.289 |    1.000 |    0.185 |
| Grandmother        |     1.000 |     0.004 |      1.000 |      0.002 |           1.000 |           0.007 |      0.654 |      0.282 |    1.000 |    0.194 |
| Grandson           |     1.000 |     0.003 |      1.000 |      0.002 |           1.000 |           0.006 |      0.687 |      0.267 |    1.000 |    0.330 |
| Mother             |     1.000 |     0.004 |      1.000 |      0.002 |           1.000 |           0.008 |      1.000 |      0.282 |    1.000 |    0.227 |
| PersonWithASibling |     1.000 |     0.004 |      1.000 |      0.001 |           0.737 |          85.697 |      0.571 |      0.316 |    1.000 |    0.242 |
| Sister             |     1.000 |     0.003 |      1.000 |      0.001 |           1.000 |           0.150 |      0.800 |      0.282 |    1.000 |    0.525 |
| Son                |     1.000 |     0.004 |      1.000 |      0.002 |           1.000 |           0.008 |      0.556 |      0.268 |    1.000 |    0.217 |
| Uncle              |     0.905 |    29.269 |      0.905 |      8.582 |           0.950 |         103.332 |      0.633 |      0.322 |    1.000 |    0.198 |


```shell
python examples/concept_learning_evaluation.py --lps LPs/Mutagenesis/lps.json --kb KGs/Mutagenesis/mutagenesis.owl --max_runtime 60 --report mutagenesis_results.csv 
python -c 'import pandas as pd; print(pd.read_csv("mutagenesis_results.csv", index_col=0).to_markdown(floatfmt=".3f"))'
```
| LP       |   F1-OCEL |   RT-OCEL |   F1-CELOE |   RT-CELOE |   F1-EvoLearner |   RT-EvoLearner |   F1-DRILL |   RT-DRILL |   F1-tDL |   RT-tDL |
|:---------|----------:|----------:|-----------:|-----------:|----------------:|----------------:|-----------:|-----------:|---------:|---------:|
| NotKnown |     0.916 |    60.002 |      0.916 |     41.288 |           0.856 |         247.329 |      0.976 |     40.502 |    1.000 |   47.646 |


```shell
python examples/concept_learning_evaluation.py --lps LPs/Carcinogenesis/lps.json --kb KGs/Carcinogenesis/carcinogenesis.owl --max_runtime 60 --report carcinogenesis_results.csv 
python -c 'import pandas as pd; print(pd.read_csv("carcinogenesis_results.csv", index_col=0).to_markdown(floatfmt=".3f"))'
```

| LP       |   F1-OCEL |   RT-OCEL |   F1-CELOE |   RT-CELOE |   F1-EvoLearner |   RT-EvoLearner |   F1-DRILL |   RT-DRILL |   F1-tDL |   RT-tDL |
|:---------|----------:|----------:|-----------:|-----------:|----------------:|----------------:|-----------:|-----------:|---------:|---------:|
| NOTKNOWN |     0.734 |    60.307 |      0.739 |     69.639 |           0.745 |        1083.561 |      0.820 |     64.725 |    1.000 |   47.550 |

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
