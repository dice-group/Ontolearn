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

The following script will apply 10-fold cross validation on  each benchmark learning problem.
```shell
# To download learning problems and benchmark learners on the Family benchmark dataset with benchmark learning problems.
python examples/concept_learning_cv_evaluation.py --lps LPs/Family/lps.json --kb KGs/Family/family-benchmark_rich_background.owl --max_runtime 3 --report family.csv 
```
```python
import pandas as pd
df=pd.read_csv("family.csv").groupby("LP").mean()
filter_col = [col for col in df if col.startswith('Test-F1') or col.startswith('RT')]
print(df[filter_col].to_markdown(floatfmt=".3f"))
```

| LP                 |   Test-F1-OCEL |   RT-OCEL |   Test-F1-CELOE |   RT-CELOE |   Test-F1-Evo |   RT-Evo |   Test-F1-DRILL |   RT-DRILL |   Test-F1-TDL |   RT-TDL |   Test-F1-NCES |   RT-NCES |
|:-------------------|---------------:|----------:|----------------:|-----------:|--------------:|---------:|----------------:|-----------:|--------------:|---------:|---------------:|----------:|
| Aunt               |          0.641 |     3.037 |           0.834 |      3.037 |         0.947 |    2.265 |           0.841 |      3.061 |         0.967 |    0.135 |          0.692 |     0.254 |
| Brother            |          1.000 |     0.005 |           1.000 |      0.005 |         1.000 |    0.495 |           1.000 |      0.006 |         1.000 |    0.106 |          0.947 |     0.235 |
| Cousin             |          0.723 |     3.023 |           0.795 |      3.023 |         0.979 |    2.780 |           0.711 |      3.040 |         0.797 |    0.169 |          0.667 |     0.225 |
| Daughter           |          1.000 |     0.005 |           1.000 |      0.005 |         1.000 |    0.499 |           1.000 |      0.006 |         1.000 |    0.163 |          0.942 |     0.217 |
| Father             |          1.000 |     0.003 |           1.000 |      0.003 |         1.000 |    0.511 |           1.000 |      0.016 |         1.000 |    0.149 |          0.900 |     0.222 |
| Granddaughter      |          1.000 |     0.003 |           1.000 |      0.003 |         1.000 |    0.478 |           0.971 |      3.033 |         1.000 |    0.124 |          0.905 |     0.218 |
| Grandfather        |          1.000 |     0.003 |           1.000 |      0.003 |         0.986 |    0.599 |           1.000 |      0.016 |         1.000 |    0.116 |          0.672 |     0.273 |
| Grandgranddaughter |          1.000 |     0.008 |           1.000 |      0.008 |         1.000 |    0.458 |           0.967 |      2.794 |         1.000 |    0.057 |          0.847 |     0.254 |
| Grandgrandfather   |          1.000 |     1.011 |           1.000 |      1.011 |         1.000 |    0.485 |           0.847 |      3.038 |         0.947 |    0.056 |          0.743 |     0.255 |
| Grandgrandmother   |          1.000 |     0.661 |           1.000 |      0.661 |         0.980 |    0.484 |           0.810 |      3.090 |         0.880 |    0.053 |          0.657 |     0.236 |
| Grandgrandson      |          1.000 |     0.547 |           1.000 |      0.547 |         1.000 |    0.471 |           0.891 |      3.032 |         0.878 |    0.071 |          0.835 |     0.244 |
| Grandmother        |          1.000 |     0.004 |           1.000 |      0.004 |         1.000 |    0.513 |           0.958 |      1.524 |         1.000 |    0.110 |          0.795 |     0.237 |
| Grandson           |          1.000 |     0.004 |           1.000 |      0.004 |         1.000 |    0.530 |           0.867 |      2.784 |         1.000 |    0.112 |          0.960 |     0.236 |
| Mother             |          1.000 |     0.003 |           1.000 |      0.003 |         1.000 |    0.546 |           1.000 |      0.017 |         1.000 |    0.154 |          0.966 |     0.229 |
| PersonWithASibling |          1.000 |     0.003 |           1.000 |      0.003 |         1.000 |    0.586 |           1.000 |      0.207 |         1.000 |    0.207 |          0.904 |     0.236 |
| Sister             |          1.000 |     0.003 |           1.000 |      0.003 |         1.000 |    0.639 |           0.986 |      0.076 |         1.000 |    0.150 |          0.942 |     0.247 |
| Son                |          1.000 |     0.004 |           1.000 |      0.004 |         1.000 |    0.596 |           0.907 |      3.044 |         1.000 |    0.144 |          0.920 |     0.240 |
| Uncle              |          0.891 |     3.064 |           0.891 |      3.064 |         0.949 |    2.620 |           0.863 |      3.089 |         0.922 |    0.102 |          0.693 |     0.253 |


Use `python examples/concept_learning_cv_evaluation.py` to apply stratified k-fold cross validation on learning problems. 

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
