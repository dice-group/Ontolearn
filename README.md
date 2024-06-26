# Ontolearn: Learning OWL Class Expression

*Ontolearn* is an open-source software library for learning owl class expressions at large scale.

Given positive and negative [OWL named individual](https://www.w3.org/TR/owl2-syntax/#Individuals) examples
$E^+$ and $E^-$, learning [OWL Class expression](https://www.w3.org/TR/owl2-syntax/#Class_Expressions) problem refers to the following supervised Machine Learning problem

$$\forall p \in E^+\ \mathcal{K} \models H(p) \wedge \forall n \in E^-\ \mathcal{K} \not \models H(n).$$

To tackle this supervised learnign problem, ontolearn offers many symbolic, neuro-sybmoloc and deep learning based Learning algorithms: 
- **Drill** &rarr; [Neuro-Symbolic Class Expression Learning](https://www.ijcai.org/proceedings/2023/0403.pdf)
- **EvoLearner** &rarr; [EvoLearner: Learning Description Logics with Evolutionary Algorithms](https://dl.acm.org/doi/abs/10.1145/3485447.3511925)
- **NCES2** &rarr; (soon) [Neural Class Expression Synthesis in ALCHIQ(D)](https://papers.dice-research.org/2023/ECML_NCES2/NCES2_public.pdf)
- **NCES** &rarr; [Neural Class Expression Synthesis](https://link.springer.com/chapter/10.1007/978-3-031-33455-9_13) 
- **NERO** &rarr; (soon) [Learning Permutation-Invariant Embeddings for Description Logic Concepts](https://link.springer.com/chapter/10.1007/978-3-031-30047-9_9)
- **CLIP** &rarr; [Learning Concept Lengths Accelerates Concept Learning in ALC](https://link.springer.com/chapter/10.1007/978-3-031-06981-9_14)
- **CELOE** &rarr; [Class Expression Learning for Ontology Engineering](https://www.sciencedirect.com/science/article/abs/pii/S1570826811000023)
- **OCEL** &rarr; A limited version of CELOE

Find more in the [Documentation](https://ontolearn-docs-dice-group.netlify.app/usage/01_introduction).

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
```shell
pytest -p no:warnings -x # Running 64 tests takes ~ 6 mins
```

## Learning OWL Class Expression
```python
from ontolearn.learners import TDL
from ontolearn.triple_store import TripleStore
from ontolearn.learning_problem import PosNegLPStandard
from owlapy.owl_individual import OWLNamedIndividual
from owlapy import owl_expression_to_sparql, owl_expression_to_dl
# (1) Initialize Triplestore
# sudo docker run -p 3030:3030 -e ADMIN_PASSWORD=pw123 stain/jena-fuseki
# Login http://localhost:3030/#/ with admin and pw123
# Create a new dataset called family and upload KGs/Family/family.owl
kb = TripleStore(url="http://localhost:3030/family")
# (2) Initialize a learner.
model = TDL(knowledge_base=kb)
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
```

## Learning OWL Class Expression over DBpedia
```python
from ontolearn.utils.static_funcs import save_owl_class_expressions

# (1) Initialize Triplestore
kb = TripleStore(url="http://dice-dbpedia.cs.upb.de:9080/sparql")
# (3) Initialize a learner.
model = TDL(knowledge_base=kb)
# (4) Define a description logic concept learning problem.
lp = PosNegLPStandard(pos={OWLNamedIndividual("http://dbpedia.org/resource/Angela_Merkel")},
                      neg={OWLNamedIndividual("http://dbpedia.org/resource/Barack_Obama")})
# (5) Learn description logic concepts best fitting (4).
h = model.fit(learning_problem=lp).best_hypotheses()
print(h)
print(owl_expression_to_dl(h))
print(owl_expression_to_sparql(expression=h))
save_owl_class_expressions(expressions=h,path="owl_prediction")
```

Fore more please refer to  the [examples](https://github.com/dice-group/Ontolearn/tree/develop/examples) folder.

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
with open(f"LPs/Mutagenesis/lps.json") as json_file:
    learning_problems = json.load(json_file)["problems"]
for str_target_concept, examples in learning_problems.items():
    response = requests.get('http://0.0.0.0:8000/cel',
                            headers={'accept': 'application/json', 'Content-Type': 'application/json'},
                            json={"pos": examples['positive_examples'],
                                  "neg": examples['negative_examples'],
                                  "model": "TDL"})
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
python examples/concept_learning_cv_evaluation.py --lps LPs/Family/lps_difficult.json --kb KGs/Family/family-benchmark_rich_background.owl --max_runtime 60 --report family_results.csv 
```
In the following python script, the results are summarized and the markdown displayed below generated.
```python
import pandas as pd
df=pd.read_csv("family_results.csv").groupby("LP").mean()
print(df[[col for col in df if col.startswith('Test-F1') or col.startswith('RT')]].to_markdown(floatfmt=".3f"))
```
**Note that DRILL is untrained and we simply used accuracy driven heuristics to learn an OWL class expression.**

Below, we report the average test F1 score and the average runtimes of learners.

|         LP         | Test-F1-OCEL | RT-OCEL | Test-F1-CELOE | RT-CELOE | Test-F1-Evo | RT-Evo | Test-F1-DRILL | RT-DRILL | Test-F1-TDL | RT-TDL | Test-F1-NCES | RT-NCES |
|:------------------:|-------------:|--------:|--------------:|---------:|------------:|-------:|--------------:|---------:|------------:|-------:|-------------:|--------:|
|        Aunt        |        0.614 |  13.697 |         0.855 |   13.697 |       0.978 |  5.278 |         0.811 |   60.351 |       0.956 |  0.118 |        0.681 |   0.234 |
|       Cousin       |        0.712 |  10.846 |         0.789 |   10.846 |       0.993 |  3.311 |         0.701 |   60.485 |       0.820 |  0.176 |        0.667 |   0.232 |
| Grandgranddaughter |        1.000 |   0.013 |         1.000 |    0.013 |       1.000 |  0.426 |         0.980 |   17.486 |       1.000 |  0.050 |        0.800 |   0.224 |
|  Grandgrandfather  |        1.000 |   0.897 |         1.000 |    0.897 |       1.000 |  0.404 |         0.947 |   55.728 |       0.947 |  0.059 |        0.707 |   0.231 |
|  Grandgrandmother  |        1.000 |   4.173 |         1.000 |    4.173 |       1.000 |  0.442 |         0.893 |   50.329 |       0.947 |  0.060 |        0.707 |   0.229 |
|   Grandgrandson    |        1.000 |   1.632 |         1.000 |    1.632 |       1.000 |  0.452 |         0.931 |   60.358 |       0.911 |  0.070 |        0.817 |   0.235 |
|       Uncle        |        0.876 |  16.244 |         0.891 |   16.244 |       0.964 |  4.516 |         0.876 |   60.416 |       0.933 |  0.098 |        0.687 |   0.253 |


|         LP         | Train-F1-OCEL | Train-F1-CELOE | Train-F1-Evo | Train-F1-DRILL | Train-F1-TDL | Train-F1-NCES | 
|:------------------:|--------------:|---------------:|-------------:|---------------:|-------------:|--------------:|
|        Aunt        |         0.835 |          0.918 |        0.995 |          0.837 |        1.000 |         0.712 |
|       Cousin       |         0.746 |          0.796 |        1.000 |          0.732 |        1.000 |         0.667 |
| Grandgranddaughter |         1.000 |          1.000 |        1.000 |          1.000 |        1.000 |         0.825 |
|  Grandgrandfather  |         1.000 |          1.000 |        1.000 |          0.968 |        1.000 |         0.741 |
|  Grandgrandmother  |         1.000 |          1.000 |        1.000 |          0.975 |        1.000 |         0.702 |
|   Grandgrandson    |         1.000 |          1.000 |        1.000 |          0.962 |        1.000 |         0.824 |
|       Uncle        |         0.904 |          0.907 |        0.996 |          0.908 |        1.000 |         0.696 |


### 10-Fold Cross Validation Mutagenesis Benchmark Results
```shell
python examples/concept_learning_cv_evaluation.py --lps LPs/Mutagenesis/lps.json --kb KGs/Mutagenesis/mutagenesis.owl --max_runtime 60 --report mutagenesis_results.csv 
```

| LP       | Train-F1-OCEL | Test-F1-OCEL | RT-OCEL | Train-F1-CELOE | Test-F1-CELOE | RT-CELOE | Train-F1-Evo | Test-F1-Evo | RT-Evo | Train-F1-DRILL | Test-F1-DRILL | RT-DRILL | Train-F1-TDL | Test-F1-TDL | RT-TDL | Train-F1-NCES | Test-F1-NCES | RT-NCES |
|:---------|--------------:|-------------:|--------:|---------------:|--------------:|---------:|-------------:|------------:|-------:|---------------:|--------------:|---------:|-------------:|------------:|-------:|--------------:|-------------:|--------:|
| NotKnown |         0.916 |        0.918 |  60.705 |          0.916 |         0.918 |   60.705 |        0.975 |       0.970 | 51.870 |          0.809 |         0.804 |   60.140 |        1.000 |       0.852 | 13.569 |         0.634 |        0.632 |   1.223 |



### 10-Fold Cross Validation Carcinogenesis Benchmark Results
```shell
python examples/concept_learning_cv_evaluation.py --lps LPs/Carcinogenesis/lps.json --kb KGs/Carcinogenesis/carcinogenesis.owl --max_runtime 60 --report carcinogenesis_results.csv 
```
| LP       | Train-F1-OCEL | Test-F1-OCEL | RT-OCEL | Train-F1-CELOE | Test-F1-CELOE | RT-CELOE | Train-F1-Evo | Test-F1-Evo | RT-Evo | Train-F1-DRILL | Test-F1-DRILL | RT-DRILL | Train-F1-TDL | Test-F1-TDL | RT-TDL | Train-F1-NCES | Test-F1-NCES | RT-NCES |
|:---------|--------------:|-------------:|--------:|---------------:|--------------:|---------:|-------------:|------------:|-------:|---------------:|--------------:|---------:|-------------:|------------:|-------:|--------------:|-------------:|--------:|
| NOTKNOWN |         0.737 |        0.711 |  62.048 |          0.740 |         0.701 |   62.048 |        0.822 |       0.628 | 64.508 |          0.740 |         0.707 |   60.120 |        1.000 |       0.616 |  5.196 |         0.375 |        0.407 |   1.246 |


</details>

## Deployment 

<details> <summary> To see the results </summary>

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

</details>

## Development

<details> <summary> To see the results </summary>
  
Creating a feature branch **refactoring** from development branch

```shell
git branch refactoring develop
```

</details>

## References
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

In case you have any question, please contact: ```caglar.demir@upb.de``` or ```caglardemir8@gmail.com```
