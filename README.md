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
# To download learning problems and benchmark learners on the Family benchmark dataset with benchmark learning problems.
python examples/concept_learning_evaluation.py --lps LPs/Family/lps.json --kb KGs/Family/family-benchmark_rich_background.owl --max_runtime 60 --report family_results.csv  && python -c 'import pandas as pd; print(pd.read_csv("family_results.csv", index_col=0).to_markdown(floatfmt=".3f"))'
```
<details> <summary> To see the results </summary>

Below, we report the average results of 5 runs.
Each model has 60 second to find a fitting answer. DRILL results are obtained by using F1 score as heuristic function.
Note that F1 scores denote the quality of the find/constructed concept w.r.t. E^+ and E^-.

### Family Benchmark Results

| LP                 |   F1-OCEL |   RT-OCEL |   F1-CELOE |   RT-CELOE |   F1-Evo |   RT-Evo |   F1-DRILL |   RT-DRILL |   F1-TDL |   RT-TDL |
|:-------------------|----------:|----------:|-----------:|-----------:|---------:|---------:|-----------:|-----------:|---------:|---------:|
| Aunt               |     0.837 |    13.966 |      0.911 |      7.395 |    1.000 |    3.929 |      0.888 |     60.088 |    1.000 |    0.560 |
| Brother            |     1.000 |     0.011 |      1.000 |      0.010 |    1.000 |    0.504 |      1.000 |      0.107 |    1.000 |    0.380 |
| Cousin             |     0.721 |    12.431 |      0.793 |      8.843 |    1.000 |    4.590 |      0.836 |     60.186 |    1.000 |    0.552 |
| Daughter           |     1.000 |     0.039 |      1.000 |      0.010 |    1.000 |    0.552 |      1.000 |      0.105 |    1.000 |    0.426 |
| Father             |     1.000 |     0.012 |      1.000 |      0.010 |    1.000 |    0.503 |      1.000 |      0.102 |    1.000 |    0.479 |
| Granddaughter      |     1.000 |     0.042 |      1.000 |      0.010 |    1.000 |    0.457 |      1.000 |      0.126 |    1.000 |    0.364 |
| Grandfather        |     1.000 |     0.012 |      1.000 |      0.010 |    1.000 |    0.448 |      1.000 |      0.161 |    1.000 |    0.425 |
| Grandgranddaughter |     1.000 |     0.013 |      1.000 |      0.011 |    1.000 |    0.433 |      1.000 |      0.095 |    1.000 |    0.339 |
| Grandgrandfather   |     1.000 |     0.784 |      1.000 |      0.202 |    1.000 |    0.488 |      1.000 |      0.846 |    1.000 |    0.379 |
| Grandgrandmother   |     1.000 |     1.293 |      1.000 |      0.247 |    1.000 |    0.411 |      1.000 |      0.852 |    1.000 |    0.417 |
| Grandgrandson      |     1.000 |     1.314 |      1.000 |      0.208 |    1.000 |    0.473 |      1.000 |      0.399 |    1.000 |    0.425 |
| Grandmother        |     1.000 |     0.012 |      1.000 |      0.011 |    1.000 |    0.446 |      1.000 |      0.099 |    1.000 |    0.442 |
| Grandson           |     1.000 |     0.013 |      1.000 |      0.011 |    1.000 |    0.470 |      1.000 |      0.135 |    1.000 |    0.448 |
| Mother             |     1.000 |     0.013 |      1.000 |      0.011 |    1.000 |    0.542 |      1.000 |      0.103 |    1.000 |    0.471 |
| PersonWithASibling |     1.000 |     0.011 |      1.000 |      0.009 |    1.000 |    0.536 |      0.737 |     60.149 |    1.000 |    0.482 |
| Sister             |     1.000 |     0.012 |      1.000 |      0.010 |    1.000 |    0.522 |      1.000 |      0.100 |    1.000 |    0.376 |
| Son                |     1.000 |     0.014 |      1.000 |      0.012 |    1.000 |    0.504 |      1.000 |      0.101 |    1.000 |    0.516 |
| Uncle              |     0.905 |    17.918 |      0.905 |      6.465 |    1.000 |    3.105 |      0.941 |     60.175 |    1.000 |    0.491 |


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
