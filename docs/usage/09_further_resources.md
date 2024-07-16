# Further Resources

You can find more details in the related papers for each algorithm:

Concept Learning:

- **NCES2** &rarr; (soon) [Neural Class Expression Synthesis in ALCHIQ(D)](https://papers.dice-research.org/2023/ECML_NCES2/NCES2_public.pdf)
- **Drill** &rarr; [Deep Reinforcement Learning for Refinement Operators in ALC](https://arxiv.org/pdf/2106.15373.pdf)
- **NCES** &rarr; [Neural Class Expression Synthesis](https://link.springer.com/chapter/10.1007/978-3-031-33455-9_13)
- **NERO** &rarr; (soon) [Learning Permutation-Invariant Embeddings for Description Logic Concepts](https://github.com/dice-group/Nero)
- **EvoLearner** &rarr; [An evolutionary approach to learn concepts in ALCQ(D)](https://dl.acm.org/doi/abs/10.1145/3485447.3511925)
- **CLIP** &rarr; [Learning Concept Lengths Accelerates Concept Learning in ALC](https://link.springer.com/chapter/10.1007/978-3-031-06981-9_14)
- **CELOE** &rarr; [Class Expression Learning for Ontology Engineering](https://www.sciencedirect.com/science/article/abs/pii/S1570826811000023)

Sampling:
- **OntoSample** &rarr; [Accelerating Concept Learning via Sampling](https://dl.acm.org/doi/10.1145/3583780.3615158)

Also check OWLAPY's documentation [here](https://dice-group.github.io/owlapy/usage/main.html).

## Citing

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

# OntoSample
@inproceedings{10.1145/3583780.3615158,
  author = {Baci, Alkid and Heindorf, Stefan},
  title = {Accelerating Concept Learning via Sampling},
  year = {2023},
  isbn = {9798400701245},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3583780.3615158},
  doi = {10.1145/3583780.3615158},
  abstract = {Node classification is an important task in many fields, e.g., predicting entity types in knowledge graphs, classifying papers in citation graphs, or classifying nodes in social networks. In many cases, it is crucial to explain why certain predictions are made. Towards this end, concept learning has been proposed as a means of interpretable node classification: given positive and negative examples in a knowledge base, concepts in description logics are learned that serve as classification models. However, state-of-the-art concept learners, including EvoLearner and CELOE exhibit long runtimes. In this paper, we propose to accelerate concept learning with graph sampling techniques. We experiment with seven techniques and tailor them to the setting of concept learning. In our experiments, we achieve a reduction in training size by over 90\% while maintaining a high predictive performance.},
  booktitle = {Proceedings of the 32nd ACM International Conference on Information and Knowledge Management},
  pages = {3733–3737},
  numpages = {5},
  keywords = {concept learning, graph sampling, knowledge bases},
  location = {, Birmingham, United Kingdom, },
  series = {CIKM '23}
}
```

## More Inside the Project

Examples and test cases provide a good starting point to get to know
the project better. Find them in the folders 
[examples](https://github.com/dice-group/Ontolearn/tree/develop/examples) and [tests](https://github.com/dice-group/Ontolearn/tree/develop/tests).

## Contribution

Feel free to create a pull request and our developers will take a look on it. 
We appreciate your commitment. 

## Questions

In case you have any question, please contact:  `onto-learn@lists.uni-paderborn.de`
or open an issue on our [GitHub issues page](https://github.com/dice-group/Ontolearn/issues).

## Code Coverage

The coverage report is generated using [coverage.py](https://coverage.readthedocs.io/en).

```
Name                                       Stmts   Miss  Cover   Missing
------------------------------------------------------------------------
ontolearn/__init__.py                          1      0   100%
ontolearn/abstracts.py                        60      0   100%
ontolearn/base_concept_learner.py            160      2    99%   313, 317
ontolearn/base_nces.py                        38      0   100%
ontolearn/clip_architectures.py               93      0   100%
ontolearn/clip_trainer.py                     94      7    93%   85, 94, 97, 102, 109, 122, 145
ontolearn/concept_generator.py                95      2    98%   68, 84
ontolearn/concept_learner.py                 699     43    94%   291, 336, 411, 466-467, 533, 972-973, 1033, 1044, 1053, 1065, 1213, 1235, 1237, 1242, 1282-1286, 1325, 1336, 1371, 1374, 1379, 1389, 1391, 1446, 1452, 1457, 1502-1506, 1533, 1542-1545, 1552-1554, 1637, 1639
ontolearn/data_struct.py                       5      0   100%
ontolearn/ea_algorithms.py                    57      2    96%   93, 96
ontolearn/ea_initialization.py               216      7    97%   93, 97, 310-315
ontolearn/ea_utils.py                         88      5    94%   93, 110-111, 114-115
ontolearn/fitness_functions.py                13      0   100%
ontolearn/heuristics.py                       45      0   100%
ontolearn/knowledge_base.py                  342     38    89%   120, 130, 155-156, 158, 161, 168, 172-173, 478-479, 519, 527, 530, 536, 574, 588, 596, 614, 618, 622, 641, 643, 657, 712, 722, 728-733, 782, 1030, 1039, 1049, 1058, 1107
ontolearn/learners/__init__.py                 2      0   100%
ontolearn/learners/drill.py                   30      0   100%
ontolearn/learners/tree_learner.py           173     13    92%   193, 377, 412, 418, 421-427, 435-437, 440, 443, 469
ontolearn/learning_problem.py                 31      1    97%   98
ontolearn/learning_problem_generator.py       16      0   100%
ontolearn/lp_generator/__init__.py             2      0   100%
ontolearn/lp_generator/generate_data.py       10      0   100%
ontolearn/lp_generator/helper_classes.py     125     14    89%   76, 85-93, 116, 135, 169-170
ontolearn/metrics.py                          50      0   100%
ontolearn/model_adapter.py                    33      0   100%
ontolearn/nces_architectures.py               72      0   100%
ontolearn/nces_modules.py                     53      5    91%   44-45, 68-69, 72
ontolearn/nces_trainer.py                    127     10    92%   70, 74, 83, 87, 147, 156, 159, 164, 173, 185
ontolearn/nces_utils.py                       24      0   100%
ontolearn/owl_neural_reasoner.py             214     11    95%   63, 96, 124, 129, 140, 205, 290, 485, 498-501
ontolearn/refinement_operators.py            521     26    95%   167-168, 226, 299, 400-401, 447, 541, 565, 599-601, 746, 782, 888, 916, 961-963, 970, 991-993, 995, 997, 1065, 1087
ontolearn/search.py                          293     25    91%   70, 133, 196, 216, 303, 307, 310, 339, 392, 429, 433, 441, 457, 467, 482, 484, 509, 511, 576-577, 666-667, 762, 766, 770
ontolearn/utils/__init__.py                   33      2    94%   55, 95
ontolearn/utils/log_config.py                 19      0   100%
ontolearn/utils/oplogging.py                   8      0   100%
ontolearn/utils/static_funcs.py               43      2    95%   55, 86
ontolearn/value_splitter.py                  159      6    96%   111-113, 118, 127, 130
------------------------------------------------------------------------
TOTAL                                       4044    221    95%
```