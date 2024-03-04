# Ontolearn

*Ontolearn* is an open-source software library for description logic learning problem.
Find more in the [Documentation](https://ontolearn-docs-dice-group.netlify.app/usage/01_introduction).

Learning algorithms: 
- **Drill** &rarr; [Neuro-Symbolic Class Expression Learning](https://www.ijcai.org/proceedings/2023/0403.pdf)
- **EvoLearner** &rarr; [EvoLearner: Learning Description Logics with Evolutionary Algorithms](https://dl.acm.org/doi/abs/10.1145/3485447.3511925)
- **NCES2** &rarr; (soon) [Neural Class Expression Synthesis in ALCHIQ(D)](https://papers.dice-research.org/2023/ECML_NCES2/NCES2_public.pdf)
- **NCES** &rarr; [Neural Class Expression Synthesis](https://link.springer.com/chapter/10.1007/978-3-031-33455-9_13) 
- **NERO** &rarr; (soon) [Learning Permutation-Invariant Embeddings for Description Logic Concepts](https://link.springer.com/chapter/10.1007/978-3-031-30047-9_9)
- **CLIP** &rarr; [Learning Concept Lengths Accelerates Concept Learning in ALC](https://link.springer.com/chapter/10.1007/978-3-031-06981-9_14)
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
# To download learning problems and benchmark learners on the Family benchmark dataset with benchmark learning problems.
python examples/concept_learning_evaluation.py --lps LPs/Family/lps.json --kb KGs/Family/family-benchmark_rich_background.owl --max_runtime 60 --report family_results.csv  && python -c 'import pandas as pd; print(pd.read_csv("family_results.csv", index_col=0).to_markdown(floatfmt=".3f"))'
```
<details> <summary> To see the results </summary>

Below, we report the average results of 5 runs.
Each model has 60 second to find a fitting answer. DRILL results are obtained by using F1 score as heuristic function.
Note that F1 scores denote the quality of the find/constructed concept w.r.t. E^+ and E^-.

### Family Benchmark Results

| LP                 |   Train-F1-OCEL |   Test-F1-OCEL |   RT-OCEL |   Train-F1-CELOE |   Test-F1-CELOE |   RT-CELOE |   Train-F1-Evo |   Test-F1-Evo |   RT-Evo |   Train-F1-DRILL |   Test-F1-DRILL |   RT-DRILL |   Train-F1-TDL |   Test-F1-TDL |   RT-TDL |   Train-F1-NCES |   Test-F1-NCES |   RT-NCES |   Train-F1-CLIP |   Test-F1-CLIP |   RT-CLIP |
|:-------------------|----------------:|---------------:|----------:|-----------------:|----------------:|-----------:|---------------:|--------------:|---------:|-----------------:|----------------:|-----------:|---------------:|--------------:|---------:|----------------:|---------------:|----------:|----------------:|---------------:|----------:|
| Aunt               |           0.848 |          0.637 |     9.206 |            0.918 |           0.855 |      9.206 |          0.996 |         0.969 |    3.390 |            0.886 |           0.799 |     60.243 |          0.971 |         0.949 |    6.366 |           0.721 |          0.635 |     0.552 |           0.899 |          0.891 |     5.763 |
| Brother            |           1.000 |          1.000 |     0.005 |            1.000 |           1.000 |      0.005 |          1.000 |         1.000 |    0.281 |            1.000 |           1.000 |      0.020 |          1.000 |         1.000 |    6.216 |           0.978 |          0.975 |     0.450 |           1.000 |          1.000 |     0.692 |
| Cousin             |           0.740 |          0.708 |     7.336 |            0.796 |           0.789 |      7.336 |          1.000 |         1.000 |    1.653 |            0.831 |           0.784 |     60.416 |          0.978 |         0.941 |    7.073 |           0.667 |          0.667 |     0.465 |           0.774 |          0.761 |     6.671 |
| Daughter           |           1.000 |          1.000 |     0.006 |            1.000 |           1.000 |      0.006 |          1.000 |         1.000 |    0.309 |            1.000 |           1.000 |      0.033 |          1.000 |         1.000 |    6.459 |           0.993 |          0.977 |     0.534 |           1.000 |          1.000 |     0.716 |
| Father             |           1.000 |          1.000 |     0.002 |            1.000 |           1.000 |      0.002 |          1.000 |         1.000 |    0.411 |            1.000 |           1.000 |      0.004 |          1.000 |         1.000 |    6.522 |           0.897 |          0.903 |     0.448 |           1.000 |          1.000 |     0.588 |
| Granddaughter      |           1.000 |          1.000 |     0.002 |            1.000 |           1.000 |      0.002 |          1.000 |         1.000 |    0.320 |            1.000 |           1.000 |      0.003 |          1.000 |         1.000 |    6.233 |           0.911 |          0.916 |     0.497 |           1.000 |          1.000 |     0.646 |
| Grandfather        |           1.000 |          1.000 |     0.002 |            1.000 |           1.000 |      0.002 |          1.000 |         1.000 |    0.314 |            1.000 |           1.000 |      0.003 |          1.000 |         1.000 |    6.185 |           0.743 |          0.717 |     0.518 |           1.000 |          1.000 |     0.721 |
| Grandgranddaughter |           1.000 |          1.000 |     0.004 |            1.000 |           1.000 |      0.004 |          1.000 |         1.000 |    0.293 |            1.000 |           1.000 |      0.002 |          1.000 |         1.000 |    5.858 |           0.837 |          0.840 |     0.518 |           1.000 |          1.000 |     0.710 |
| Grandgrandfather   |           1.000 |          1.000 |     0.668 |            1.000 |           1.000 |      0.668 |          1.000 |         1.000 |    0.341 |            1.000 |           1.000 |      0.243 |          0.951 |         0.947 |    5.915 |           0.759 |          0.677 |     0.511 |           1.000 |          1.000 |     1.964 |
| Grandgrandmother   |           1.000 |          1.000 |     0.381 |            1.000 |           1.000 |      0.381 |          1.000 |         1.000 |    0.258 |            1.000 |           1.000 |      0.243 |          0.944 |         0.947 |    5.918 |           0.721 |          0.687 |     0.498 |           0.997 |          1.000 |     2.620 |
| Grandgrandson      |           1.000 |          1.000 |     0.341 |            1.000 |           1.000 |      0.341 |          1.000 |         1.000 |    0.276 |            1.000 |           1.000 |      0.122 |          0.938 |         0.911 |    6.093 |           0.779 |          0.809 |     0.460 |           1.000 |          1.000 |     2.555 |
| Grandmother        |           1.000 |          1.000 |     0.002 |            1.000 |           1.000 |      0.002 |          1.000 |         1.000 |    0.385 |            1.000 |           1.000 |      0.003 |          1.000 |         1.000 |    6.135 |           0.762 |          0.725 |     0.480 |           1.000 |          1.000 |     0.628 |
| Grandson           |           1.000 |          1.000 |     0.002 |            1.000 |           1.000 |      0.002 |          1.000 |         1.000 |    0.299 |            1.000 |           1.000 |      0.003 |          1.000 |         1.000 |    6.301 |           0.896 |          0.903 |     0.552 |           1.000 |          1.000 |     0.765 |
| Mother             |           1.000 |          1.000 |     0.002 |            1.000 |           1.000 |      0.002 |          1.000 |         1.000 |    0.327 |            1.000 |           1.000 |      0.004 |          1.000 |         1.000 |    6.570 |           0.967 |          0.972 |     0.555 |           1.000 |          1.000 |     0.779 |
| PersonWithASibling |           1.000 |          1.000 |     0.002 |            1.000 |           1.000 |      0.002 |          1.000 |         1.000 |    0.377 |            0.737 |           0.725 |     60.194 |          1.000 |         1.000 |    6.548 |           0.927 |          0.928 |     0.648 |           1.000 |          1.000 |     0.999 |
| Sister             |           1.000 |          1.000 |     0.002 |            1.000 |           1.000 |      0.002 |          1.000 |         1.000 |    0.356 |            1.000 |           1.000 |      0.017 |          1.000 |         1.000 |    6.315 |           0.866 |          0.876 |     0.512 |           1.000 |          1.000 |     0.616 |
| Son                |           1.000 |          1.000 |     0.002 |            1.000 |           1.000 |      0.002 |          1.000 |         1.000 |    0.317 |            1.000 |           1.000 |      0.004 |          1.000 |         1.000 |    6.579 |           0.892 |          0.855 |     0.537 |           1.000 |          1.000 |     0.700 |
| Uncle              |           0.903 |          0.891 |    12.441 |            0.907 |           0.891 |     12.441 |          1.000 |         0.971 |    1.675 |            0.951 |           0.894 |     60.337 |          0.894 |         0.896 |    6.310 |           0.667 |          0.665 |     0.619 |           0.928 |          0.942 |     5.577 |


### Mutagenesis Benchmark Results
```shell
python examples/concept_learning_evaluation.py --lps LPs/Mutagenesis/lps.json --kb KGs/Mutagenesis/mutagenesis.owl --max_runtime 60 --report mutagenesis_results.csv && python -c 'import pandas as pd; print(pd.read_csv("mutagenesis_results.csv", index_col=0).to_markdown(floatfmt=".3f"))'
```
| LP       |   Train-F1-OCEL |   Test-F1-OCEL |   RT-OCEL |   Train-F1-CELOE |   Test-F1-CELOE |   RT-CELOE |   Train-F1-Evo |   Test-F1-Evo |   RT-Evo |   Train-F1-DRILL |   Test-F1-DRILL |   RT-DRILL |   Train-F1-TDL |   Test-F1-TDL |   RT-TDL |   Train-F1-NCES |   Test-F1-NCES |   RT-NCES |   Train-F1-CLIP |   Test-F1-CLIP |   RT-CLIP |
|:---------|----------------:|---------------:|----------:|-----------------:|----------------:|-----------:|---------------:|--------------:|---------:|-----------------:|----------------:|-----------:|---------------:|--------------:|---------:|----------------:|---------------:|----------:|----------------:|---------------:|----------:|
| NotKnown |           0.916 |          0.918 |    58.328 |            0.916 |           0.918 |     58.328 |          0.724 |         0.729 |   49.281 |            0.704 |           0.704 |     60.052 |          0.879 |         0.771 |    7.763 |           0.564 |          0.560 |     0.493 |           0.814 |          0.807 |     5.622 |

### Carcinogenesis Benchmark Results
```shell
python examples/concept_learning_evaluation.py --lps LPs/Carcinogenesis/lps.json --kb KGs/Carcinogenesis/carcinogenesis.owl --max_runtime 60 --report carcinogenesis_results.csv  && python -c 'import pandas as pd; print(pd.read_csv("carcinogenesis_results.csv", index_col=0).to_markdown(floatfmt=".3f"))'
```

| LP       |   Train-F1-OCEL |   Test-F1-OCEL |   RT-OCEL |   Train-F1-CELOE |   Test-F1-CELOE |   RT-CELOE |   Train-F1-Evo |   Test-F1-Evo |   RT-Evo |   Train-F1-DRILL |   Test-F1-DRILL |   RT-DRILL |   Train-F1-TDL |   Test-F1-TDL |   RT-TDL |   Train-F1-NCES |   Test-F1-NCES |   RT-NCES |   Train-F1-CLIP |   Test-F1-CLIP |   RT-CLIP |
|:---------|----------------:|---------------:|----------:|-----------------:|----------------:|-----------:|---------------:|--------------:|---------:|-----------------:|----------------:|-----------:|---------------:|--------------:|---------:|----------------:|---------------:|----------:|----------------:|---------------:|----------:|
| NOTKNOWN |           0.738 |          0.711 |    42.936 |            0.740 |           0.701 |     42.936 |          0.744 |         0.733 |   63.465 |            0.705 |           0.704 |     60.069 |          0.879 |         0.682 |    7.260 |           0.415 |          0.396 |     1.911 |           0.720 |          0.700 |    85.037 |



Use `python examples/concept_learning_cv_evaluation.py` to apply stratified k-fold cross validation on learning problems. 

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
