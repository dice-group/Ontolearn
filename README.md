[![Downloads](https://static.pepy.tech/badge/ontolearn)](https://pepy.tech/project/ontolearn)
[![Downloads](https://img.shields.io/pypi/dm/ontolearn)](https://pypi.org/project/ontolearn/)
[![Coverage](https://img.shields.io/badge/coverage-86%25-green)](https://ontolearn-docs-dice-group.netlify.app/usage/09_further_resources#code-coverage)
[![Pypi](https://img.shields.io/badge/pypi-0.9.2-blue)](https://pypi.org/project/ontolearn/0.9.2/)
[![Docs](https://img.shields.io/badge/documentation-0.9.2-yellow)](https://ontolearn-docs-dice-group.netlify.app/usage/01_introduction)
[![Python](https://img.shields.io/badge/python-3.10.13+-4584b6)](https://www.python.org/downloads/release/python-31013/)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/dice-group/Ontolearn)
&nbsp;

![Ontolearn](docs/_static/images/Ontolearn_logo.png)

# Ontolearn: Learning OWL Class Expressions

*Ontolearn* is an open-source software library for learning owl class expressions at large scale.

Given positive and negative [OWL named individual](https://www.w3.org/TR/owl2-syntax/#Individuals) examples
$E^+$ and $E^-$, learning [OWL Class expression](https://www.w3.org/TR/owl2-syntax/#Class_Expressions) problem refers to the following supervised Machine Learning problem

$$\forall p \in E^+\ \mathcal{K} \models H(p) \wedge \forall n \in E^-\ \mathcal{K} \not \models H(n).$$

To tackle this supervised learning problem, ontolearn offers many symbolic, neuro-symbolic and deep learning based Learning algorithms: 
- **TDL** &rarr; Tree-based OWL Class Expression Learner for Large Graphs
- **Drill** &rarr; [Neuro-Symbolic Class Expression Learning](https://www.ijcai.org/proceedings/2023/0403.pdf)
- **EvoLearner** &rarr; [EvoLearner: Learning Description Logics with Evolutionary Algorithms](https://dl.acm.org/doi/abs/10.1145/3485447.3511925)
- **NCES2** &rarr; [Neural Class Expression Synthesis in ALCHIQ(D)](https://papers.dice-research.org/2023/ECML_NCES2/NCES2_public.pdf)
- **ROCES** &rarr; [Robust Class Expression Synthesis in Description Logics via Iterative Sampling](https://www.ijcai.org/proceedings/2024/0479.pdf)
- **NCES** &rarr; [Neural Class Expression Synthesis](https://link.springer.com/chapter/10.1007/978-3-031-33455-9_13) 
- **NERO** &rarr; (soon) [Learning Permutation-Invariant Embeddings for Description Logic Concepts](https://link.springer.com/chapter/10.1007/978-3-031-30047-9_9)
- **CLIP** &rarr; [Learning Concept Lengths Accelerates Concept Learning in ALC](https://link.springer.com/chapter/10.1007/978-3-031-06981-9_14)
- **CELOE** &rarr; [Class Expression Learning for Ontology Engineering](https://www.sciencedirect.com/science/article/abs/pii/S1570826811000023)
- **OCEL** &rarr; A limited version of CELOE

Find more in the [Documentation](https://ontolearn-docs-dice-group.netlify.app/usage/01_introduction).

[DeepWiki](https://deepwiki.com/dice-group/Ontolearn) can also help you get started with Ontolearn.


## Installation

```shell
pip install ontolearn 
```
or
```shell
git clone https://github.com/dice-group/Ontolearn.git 
# To create a virtual python env with conda 
conda create -n venv python=3.10.14 --no-default-packages && conda activate venv && pip install -e .
# To download knowledge graphs
wget https://files.dice-research.org/projects/Ontolearn/KGs.zip -O ./KGs.zip && unzip KGs.zip
# To download learning problems
wget https://files.dice-research.org/projects/Ontolearn/LPs.zip -O ./LPs.zip && unzip LPs.zip
```

## Learning OWL Class Expressions
```python
from ontolearn.learners import TDL
from ontolearn.triple_store import TripleStore
from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.learning_problem import PosNegLPStandard
from owlapy.owl_individual import OWLNamedIndividual
from owlapy import owl_expression_to_sparql, owl_expression_to_dl
# (1) Initialize Triplestore or KnowledgeBase
# sudo docker run -p 3030:3030 -e ADMIN_PASSWORD=pw123 stain/jena-fuseki
# Login http://localhost:3030/#/ with admin and pw123 and upload KGs/Family/family.owl
# kb = TripleStore(url="http://localhost:3030/family")
kb = KnowledgeBase(path="KGs/Family/father.owl")
# (2) Initialize a learner.
model = TDL(knowledge_base=kb, use_nominals=True)
# (3) Define a description logic concept learning problem.
lp = PosNegLPStandard(pos={OWLNamedIndividual("http://example.com/father#stefan")},
                      neg={OWLNamedIndividual("http://example.com/father#heinz"),
                           OWLNamedIndividual("http://example.com/father#anna"),
                           OWLNamedIndividual("http://example.com/father#michelle")})
# (4) Learn description logic concepts best fitting (3).
h = model.fit(learning_problem=lp).best_hypotheses()
print(h) 
print(owl_expression_to_dl(h))
print(owl_expression_to_sparql(expression=h)) 
"""
OWLObjectSomeValuesFrom(property=OWLObjectProperty(IRI('http://example.com/father#','hasChild')),filler=OWLObjectOneOf((OWLNamedIndividual(IRI('http://example.com/father#','markus')),)))

∃ hasChild.{markus}

SELECT
 DISTINCT ?x WHERE { 
?x <http://example.com/father#hasChild> ?s_1 . 
 FILTER ( ?s_1 IN ( 
<http://example.com/father#markus>
 ) )
 }
"""
print(model.classification_report)
"""
Classification Report: Negatives: -1 and Positives 1 
              precision    recall  f1-score   support

    Negative       1.00      1.00      1.00         3
    Positive       1.00      1.00      1.00         1

    accuracy                           1.00         4
   macro avg       1.00      1.00      1.00         4
weighted avg       1.00      1.00      1.00         4
"""
```

## Learning OWL Class Expressions over DBpedia
```python
from ontolearn.learners import TDL, Drill
from ontolearn.triple_store import TripleStore
from ontolearn.learning_problem import PosNegLPStandard
from owlapy.owl_individual import OWLNamedIndividual
from owlapy import owl_expression_to_sparql, owl_expression_to_dl
from ontolearn.utils.static_funcs import save_owl_class_expressions
# (1) Initialize Triplestore
kb = TripleStore(url="https://dbpedia.data.dice-research.org/sparql")
# (3) Initialize a learner.
model = Drill(knowledge_base=kb) #  or  TDL(knowledge_base=kb)
# (4) Define a description logic concept learning problem.
lp = PosNegLPStandard(pos={OWLNamedIndividual("http://dbpedia.org/resource/Angela_Merkel")},
                      neg={OWLNamedIndividual("http://dbpedia.org/resource/Barack_Obama")})
# (5) Learn description logic concepts best fitting (4).
h = model.fit(learning_problem=lp).best_hypotheses()
print(h)
print(owl_expression_to_dl(h))
print(owl_expression_to_sparql(expression=h))
save_owl_class_expressions(expressions=h,path="#owl_prediction")
```

Fore more please refer to the [examples](https://github.com/dice-group/Ontolearn/tree/develop/examples) folder.

## ontolearn-webservice 

<details><summary> Click me! </summary>

Load an RDF knowledge graph 
```shell
ontolearn-webservice --path_knowledge_base KGs/Mutagenesis/mutagenesis.owl
```
or launch a Tentris instance https://github.com/dice-group/tentris over Mutagenesis.
```shell
ontolearn-webservice --endpoint_triple_store http://0.0.0.0:9080/sparql
```
The below code trains DRILL with 6 randomly generated learning problems
provided that **path_to_pretrained_drill** does not lead to a directory containing pretrained DRILL.
Thereafter, trained DRILL is saved in the directory **path_to_pretrained_drill**.
Finally, trained DRILL will learn an OWL class expression.
```python
import json
import requests
with open(f"LPs/Mutagenesis/lps.json") as json_file:
    learning_problems = json.load(json_file)["problems"]
for str_target_concept, examples in learning_problems.items():
    response = requests.get('http://0.0.0.0:8000/cel',
                            headers={'accept': 'application/json', 'Content-Type': 'application/json'},
                            json={"pos": examples['positive_examples'],
                                  "neg": examples['negative_examples'],
                                  "model": "Drill",
                                  "path_embeddings": "mutagenesis_embeddings/Keci_entity_embeddings.csv",
                                  "path_to_pretrained_drill": "pretrained_drill",
                                  # if pretrained_drill exists, upload, otherwise train one and save it there
                                  "num_of_training_learning_problems": 2,
                                  "num_of_target_concepts": 3,
                                  "max_runtime": 60000,  # seconds
                                  "iter_bound": 1  # number of iterations/applied refinement opt.
                                  })
    print(response.json())  # {'Prediction': '∀ hasAtom.(¬Nitrogen-34)', 'F1': 0.7283582089552239, 'saved_prediction': 'Predictions.owl'}
```
TDL (a more scalable learner) can also be used as follows
```python
import json
import requests
response = requests.get('http://0.0.0.0:8000/cel',
                        headers={'accept': 'application/json', 'Content-Type': 'application/json'},
                        json={"pos": examples['positive_examples'],
                              "neg": examples['negative_examples'],
                              "model": "TDL"})
print(response.json())
```
NCES (another scalable learner). The following will first train NCES if the provided path `path_to_pretrained_nces` does not exist
```python
import json
import requests
with open(f"LPs/Mutagenesis/lps.json") as json_file:
    learning_problems = json.load(json_file)["problems"]
## This trains NCES before solving the provided learning problems. Expect poor performance for this number of epochs, and this training data size.
## If GPU is available, set `num_of_training_learning_problems` t0 10_000 or more. Set `nces_train_epochs` to 300 or more, and increase `nces_batch_size`.
for str_target_concept, examples in learning_problems.items():
    response = requests.get('http://0.0.0.0:8000/cel',
                            headers={'accept': 'application/json', 'Content-Type': 'application/json'},
                            json={"pos": examples['positive_examples'],
                                  "neg": examples['negative_examples'],
                                  "model": "NCES",
                                  "path_embeddings": "mutagenesis_embeddings/Keci_entity_embeddings.csv",
                                  "path_to_pretrained_nces": None,
                                  # if pretrained_nces exists, load weghts, otherwise train one and save it
                                  "num_of_training_learning_problems": 100,
                                  "nces_train_epochs": 5,
                                  "nces_batch_size": 16
                                  })
    print(response.json())
```

Now this will use pretrained weights for NCES

```python
import json
import requests
with open(f"LPs/Mutagenesis/lps.json") as json_file:
    learning_problems = json.load(json_file)["problems"]
for str_target_concept, examples in learning_problems.items():
    response = requests.get('http://0.0.0.0:8000/cel',
                            headers={'accept': 'application/json', 'Content-Type': 'application/json'},
                            json={"pos": examples['positive_examples'],
                                  "neg": examples['negative_examples'],
                                  "model": "NCES",
                                  "path_embeddings": "./NCESData/mutagenesis/embeddings/ConEx_entity_embeddings.csv",
                                  "path_to_pretrained_nces": "./NCESData/mutagenesis/trained_models/",
                                  # if pretrained_nces exists, load weghts, otherwise train one and save it
                                  "num_of_training_learning_problems": 100,
                                  "nces_train_epochs": 5,
                                  "nces_batch_size": 16
                                  })
    print(response.json())
```

</details>

## Benchmark Results

<details> <summary> To see the results </summary>

```shell
# To download learning problems. # Benchmark learners on the Family benchmark dataset with benchmark learning problems.
wget https://files.dice-research.org/projects/Ontolearn/LPs.zip -O ./LPs.zip && unzip LPs.zip
```

### 10-Fold Cross Validation Family Benchmark Results

Here we apply 10-fold cross validation technique on each benchmark learning problem with max runtime of 60 seconds to measure the training and testing performance of learners.
In the evaluation, from a given single learning problem (a set of positive and negative examples), a learner learns an OWL Class Expression (H) on a given 9 fold of positive and negative examples.
To compute the training performance, We compute F1-score of H train positive and negative examples.
To compute the test performance, we compute F1-score of H w.r.t. test positive and negative examples.
  
```shell
# To download learning problems and benchmark learners on the Family benchmark dataset with benchmark learning problems.
python examples/concept_learning_cv_evaluation.py --kb ./KGs/Family/family-benchmark_rich_background.owl --lps ./LPs/Family/lps_difficult.json --path_of_nces_embeddings ./NCESData/family/embeddings/ConEx_entity_embeddings.csv --path_of_clip_embeddings ./CLIPData/family/embeddings/ConEx_entity_embeddings.csv --max_runtime 60 --report family_results.csv 
```

```shell
# To download learning problems and benchmark with selected learners on the Family benchmark dataset with benchmark learning problems.
python examples/concept_learning_cv_evaluation.py --kb ./KGs/Family/family-benchmark_rich_background.owl --lps ./LPs/Family/lps_difficult.json --learner_types ocel drill tdl nces --path_of_nces_embeddings ./NCESData/family/embeddings/ConEx_entity_embeddings.csv --path_of_clip_embeddings ./CLIPData/family/embeddings/ConEx_entity_embeddings.csv --max_runtime 60 --report family_results.csv 
```

```shell
# To download learning problems and benchmark with a single learner on the Family benchmark dataset with benchmark learning problems.
python examples/concept_learning_cv_evaluation.py --kb ./KGs/Family/family-benchmark_rich_background.owl --lps ./LPs/Family/lps_difficult.json --learner_types nces --path_of_nces_embeddings ./NCESData/family/embeddings/ConEx_entity_embeddings.csv --path_of_clip_embeddings ./CLIPData/family/embeddings/ConEx_entity_embeddings.csv --max_runtime 60 --report family_results.csv 
```
In the following python script, the results are summarized and the markdown displayed below generated.
```python
import pandas as pd
df=pd.read_csv("family_results.csv").groupby("LP").mean()
print(df[[col for col in df if col.startswith('Test-F1') or col.startswith('RT')]].to_markdown(floatfmt=".3f"))
```
**Note that DRILL is untrained and we simply used accuracy driven heuristics to learn an OWL class expression.**

Below, we report the average test F1 score and the average runtimes of learners.

|         LP         | Test-F1-OCEL | RT-OCEL | Test-F1-CELOE | RT-CELOE | Test-F1-Evo | RT-Evo | Test-F1-DRILL | RT-DRILL | Test-F1-TDL | RT-TDL |   Test-F1-NCES |   RT-NCES |   Test-F1-NCES2 |   RT-NCES2 |   Test-F1-ROCES |   RT-ROCES | Test-F1-CLIP | RT-CLIP |
|:------------------:|-------------:|--------:|--------------:|---------:|------------:|-------:|--------------:|---------:|------------:|-------:|---------------:|----------:|----------------:|-----------:|----------------:|-----------:|-------------:|--------:|
|        Aunt        |        0.614 |  13.697 |         0.855 |   13.697 |       0.978 |  5.278 |         0.811 |   60.351 |       0.956 |  0.118 |          0.805 |     0.632 |           0.812 |      1.136 |           0.812 |      1.119 |        0.855 |   14.059 |
|       Cousin       |        0.712 |  10.846 |         0.789 |   10.846 |       0.993 |  3.311 |         0.701 |   60.485 |       0.820 |  0.176 |          0.608 |     0.628 |           0.680 |      1.177 |           0.695 |      1.086 |        0.779 |   9.050 |
| Grandgranddaughter |        1.000 |   0.013 |         1.000 |    0.013 |       1.000 |  0.426 |         0.980 |   17.486 |       1.000 |  0.050 |          1.000 |     0.507 |           1.000 |      0.955 |           1.000 |      0.917 |        1.000 |   0.639 |
|  Grandgrandfather  |        1.000 |   0.897 |         1.000 |    0.897 |       1.000 |  0.404 |         0.947 |   55.728 |       0.947 |  0.059 |          0.927 |     0.505 |           0.947 |      0.944 |           0.927 |      0.924 |        1.000 |   0.746  |
|  Grandgrandmother  |        1.000 |   4.173 |         1.000 |    4.173 |       1.000 |  0.442 |         0.893 |   50.329 |       0.947 |  0.060 |          0.947 |     0.633 |           0.933 |      1.323 |           0.947 |      1.306 |        1.000 |   0.817 |
|   Grandgrandson    |        1.000 |   1.632 |         1.000 |    1.632 |       1.000 |  0.452 |         0.931 |   60.358 |       0.911 |  0.070 |          0.909 |     0.598 |           0.931 |      1.171 |           0.931 |      1.146 |        1.000 |   0.939 |
|       Uncle        |        0.876 |  16.244 |         0.891 |   16.244 |       0.964 |  4.516 |         0.876 |   60.416 |       0.933 |  0.098 |          0.854 |     0.538 |           0.891 |      0.948 |           0.891 |      0.905 |        0.928 |   17.682 |


|         LP         | Train-F1-OCEL | Train-F1-CELOE | Train-F1-Evo | Train-F1-DRILL | Train-F1-TDL |   Train-F1-NCES |   Train-F1-NCES2 |   Train-F1-ROCES |   Train-F1-CLIP |
|:------------------:|--------------:|---------------:|-------------:|---------------:|-------------:|----------------:|-----------------:|-----------------:|----------------:|
|        Aunt        |         0.835 |          0.918 |        0.995 |          0.837 |        1.000 |           0.759 |            0.804 |            0.804 |           0.918 |
|       Cousin       |         0.746 |          0.796 |        1.000 |          0.732 |        1.000 |           0.680 |            0.696 |            0.728 |           0.798 |
| Grandgranddaughter |         1.000 |          1.000 |        1.000 |          1.000 |        1.000 |           1.000 |            1.000 |            1.000 |           1.000 |
|  Grandgrandfather  |         1.000 |          1.000 |        1.000 |          0.968 |        1.000 |           0.910 |            0.944 |            0.942 |           1.000 |
|  Grandgrandmother  |         1.000 |          1.000 |        1.000 |          0.975 |        1.000 |           0.923 |            0.941 |            0.944 |           1.000 |
|   Grandgrandson    |         1.000 |          1.000 |        1.000 |          0.962 |        1.000 |           0.911 |            0.923 |            0.923 |           1.000 |
|       Uncle        |         0.904 |          0.907 |        0.996 |          0.908 |        1.000 |           0.823 |            0.886 |            0.884 |           0.940 |


### 10-Fold Cross Validation Mutagenesis Benchmark Results
```shell
python examples/concept_learning_cv_evaluation.py --kb ./KGs/Mutagenesis/mutagenesis.owl --lps ./LPs/Mutagenesis/lps.json --path_of_nces_embeddings ./NCESData/mutagenesis/embeddings/ConEx_entity_embeddings.csv --path_of_clip_embeddings ./CLIPData/mutagenesis/embeddings/ConEx_entity_embeddings.csv --max_runtime 60 --report mutagenesis_results.csv 
```

| LP       | Train-F1-OCEL | Test-F1-OCEL | RT-OCEL | Train-F1-CELOE | Test-F1-CELOE | RT-CELOE | Train-F1-Evo | Test-F1-Evo | RT-Evo | Train-F1-DRILL | Test-F1-DRILL | RT-DRILL | Train-F1-TDL | Test-F1-TDL | RT-TDL |   Train-F1-NCES |   Test-F1-NCES |   RT-NCES |   Train-F1-NCES2 |   Test-F1-NCES2 |   RT-NCES2 |   Train-F1-ROCES |   Test-F1-ROCES |   RT-ROCES | Train-F1-CLIP | Test-F1-CLIP | RT-CLIP |
|:---------|--------------:|-------------:|--------:|---------------:|--------------:|---------:|-------------:|------------:|-------:|---------------:|--------------:|---------:|-------------:|------------:|-------:|----------------:|---------------:|----------:|-----------------:|----------------:|-----------:|-----------------:|----------------:|-----------:|--------------:|-------------:|--------:|
| NotKnown |         0.916 |        0.918 |  60.705 |          0.916 |         0.918 |   60.705 |        0.975 |       0.970 | 51.870 |          0.809 |         0.804 |   60.140 |        1.000 |       0.852 | 13.569 |           0.704 |          0.704 |     2.605 |            0.704 |           0.704 |      1.841 |            0.704 |           0.704 |      1.711 |         0.916 |        0.918 |   26.312|



### 10-Fold Cross Validation Carcinogenesis Benchmark Results
```shell
python examples/concept_learning_cv_evaluation.py --kb ./KGs/Carcinogenesis/carcinogenesis.owl --lps ./LPs/Carcinogenesis/lps.json --path_of_nces_embeddings ./NCESData/carcinogenesis/embeddings/ConEx_entity_embeddings.csv --path_of_clip_embeddings ./CLIPData/carcinogenesis/embeddings/ConEx_entity_embeddings.csv --max_runtime 60 --report carcinogenesis_results.csv 
```

| LP       | Train-F1-OCEL | Test-F1-OCEL | RT-OCEL | Train-F1-CELOE | Test-F1-CELOE | RT-CELOE | Train-F1-Evo | Test-F1-Evo | RT-Evo | Train-F1-DRILL | Test-F1-DRILL | RT-DRILL | Train-F1-TDL | Test-F1-TDL | RT-TDL |   Train-F1-NCES |   Test-F1-NCES |   RT-NCES |   Train-F1-NCES2 |   Test-F1-NCES2 |   RT-NCES2 |   Train-F1-ROCES |   Test-F1-ROCES |   RT-ROCES | Train-F1-CLIP | Test-F1-CLIP | RT-CLIP |
|:---------|--------------:|-------------:|--------:|---------------:|--------------:|---------:|-------------:|------------:|-------:|---------------:|--------------:|---------:|-------------:|------------:|-------:|----------------:|---------------:|----------:|-----------------:|----------------:|-----------:|-----------------:|----------------:|-----------:|--------------:|-------------:|--------:|
| NOTKNOWN |         0.737 |        0.711 |  62.048 |          0.740 |         0.701 |   62.048 |        0.822 |       0.628 | 64.508 |          0.740 |         0.707 |   60.120 |        1.000 |       0.616 |  5.196 |           0.709 |          0.709 |     2.718 |            0.705 |           0.704 |      1.912 |            0.705 |           0.704 |      1.774 |        0.740 |        0.701 |   48.475|


### Benchmark Results on DBpedia. Results are based on the training examples only

```shell
python examples/owl_class_expression_learning_dbpedia.py --model Drill && python examples/owl_class_expression_learning_dbpedia.py --model TDL 
```
|           LP-Type         | Train-F1-Drill |   RT-Drill   |  Train-F1-TDL  |     RT-TDL    |
|:--------------------------|---------------:|-------------:|---------------:|--------------:|
|  OWLObjectAllValuesFrom   |          0.438 |  240.331     |          1.000 |       206.288 |
|  OWLObjectIntersectionOf  |          0.213 |  202.558     |          0.717 |        91.660 |
|  OWLObjectUnionOf         |          0.546 |  187.144     |          0.967 |       129.700 |

</details>

## Development


<details> <summary> To see the results </summary>

Creating a feature branch **refactoring** from development branch

```shell
git branch refactoring develop
```

Each feature branch must be merged to develop branch. To this end, the tests must run without a problem:
```shell
# To download knowledge graphs
wget https://files.dice-research.org/projects/Ontolearn/KGs.zip -O ./KGs.zip && unzip KGs.zip
# To download learning problems
wget https://files.dice-research.org/projects/Ontolearn/LPs.zip -O ./LPs.zip && unzip LPs.zip
# Download weights for some model for few tests
wget https://files.dice-research.org/projects/NCES/NCES_Ontolearn_Data/NCESData.zip -O ./NCESData.zip && unzip NCESData.zip && rm NCESData.zip
wget https://files.dice-research.org/projects/Ontolearn/CLIP/CLIPData.zip && unzip CLIPData.zip && rm CLIPData.zip 
pytest -p no:warnings -x # Running 76 tests takes ~ 17 mins
```



</details>

## References
Currently, we are working on our manuscript describing our framework. 
If you find our work useful in your research, please consider citing the respective paper:
```
# Ontolearn
@article{demir2025ontolearn,
  title={Ontolearn---A Framework for Large-scale OWL Class Expression Learning in Python},
  author={Demir, Caglar and Baci, Alkid and Kouagou, N'Dah Jean and Sieger, Leonie Nora and Heindorf, Stefan and Bin, Simon and Bl{\"u}baum, Lukas and Bigerl, Alexander and Ngomo, Axel-Cyrille Ngonga},
  journal={Journal of Machine Learning Research},
  volume={26},
  number={63},
  pages={1--6},
  year={2025}
}

# ROCES
@inproceedings{kouagou2024roces,
  title     = {ROCES: Robust Class Expression Synthesis in Description Logics via Iterative Sampling},
  author    = {Kouagou, N'Dah Jean and Heindorf, Stefan and Demir, Caglar and Ngonga Ngomo, Axel-Cyrille},
  booktitle = {Proceedings of the Thirty-Third International Joint Conference on
               Artificial Intelligence, {IJCAI-24}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  editor    = {Kate Larson},
  pages     = {4335--4343},
  year      = {2024},
  month     = {8},
  note      = {Main Track},
  doi       = {10.24963/ijcai.2024/479},
  url       = {https://doi.org/10.24963/ijcai.2024/479},
}

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

In case you have any question, please contact: ```caglar.demir@upb.de``` or ```caglardemir8@gmail.com```
