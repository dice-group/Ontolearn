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
ontolearn/base_concept_learner.py            158      2    99%   311, 315
ontolearn/base_nces.py                        38      0   100%
ontolearn/clip_architectures.py               93     77    17%   33-41, 45-56, 61-69, 73-84, 90-101, 105-119, 125-131, 137-141
ontolearn/clip_trainer.py                     94     76    19%   45-50, 53-55, 69-75, 78-151
ontolearn/concept_generator.py                95      2    98%   68, 84
ontolearn/concept_learner.py                 748    173    77%   219, 294, 339, 414, 469-470, 536, 975-976, 1036, 1047, 1056, 1068, 1187-1211, 1214-1242, 1245, 1282-1298, 1301-1314, 1320-1382, 1387-1397, 1450, 1458-1463, 1469-1490, 1497-1499, 1544-1548, 1575, 1586-1589, 1596-1598, 1672-1678, 1688-1689, 1694, 1696
ontolearn/data_struct.py                       5      0   100%
ontolearn/ea_algorithms.py                    57      1    98%   93
ontolearn/ea_initialization.py               216      7    97%   93, 97, 310-315
ontolearn/ea_utils.py                         88      5    94%   93, 110-111, 114-115
ontolearn/fitness_functions.py                13      0   100%
ontolearn/heuristics.py                       45      0   100%
ontolearn/knowledge_base.py                  340     53    84%   120, 130, 153-154, 156, 159, 166, 170-171, 175, 479-480, 512, 520, 528, 531, 537, 571, 574-582, 587-588, 595-597, 618, 622, 626, 641-643, 647, 662, 711, 721, 727-732, 779, 1027, 1036, 1046, 1055, 1104
ontolearn/learners/__init__.py                 2      0   100%
ontolearn/learners/drill.py                   30      0   100%
ontolearn/learners/tree_learner.py           205     28    86%   190, 273-303, 391, 398, 400-404, 420, 423, 444, 453
ontolearn/learning_problem.py                 31      1    97%   98
ontolearn/learning_problem_generator.py       16      0   100%
ontolearn/lp_generator/__init__.py             2      0   100%
ontolearn/lp_generator/generate_data.py       10      0   100%
ontolearn/lp_generator/helper_classes.py     125     14    89%   76, 85-93, 116, 135, 169-170
ontolearn/metrics.py                          50      0   100%
ontolearn/nces_architectures.py               72      0   100%
ontolearn/nces_modules.py                     53      5    91%   44-45, 68-69, 72
ontolearn/nces_trainer.py                    127     11    91%   48, 70, 74, 83, 87, 147, 156, 159, 164, 173, 185
ontolearn/nces_utils.py                       24      0   100%
ontolearn/owl_neural_reasoner.py             215     11    95%   57, 93, 121, 126, 137, 193, 281, 475, 488-491
ontolearn/refinement_operators.py            521     31    94%   167-168, 226, 299, 400-401, 447, 541, 565, 599-601, 746, 782, 867-868, 888, 916, 935, 961-963, 967-968, 970, 991-993, 995, 997, 1065, 1087
ontolearn/search.py                          293     25    91%   70, 133, 196, 216, 303, 307, 310, 339, 392, 429, 433, 441, 457, 467, 482, 484, 509, 511, 576-577, 666-667, 762, 766, 770
ontolearn/utils/__init__.py                   33      2    94%   58, 98
ontolearn/utils/log_config.py                 19      0   100%
ontolearn/utils/oplogging.py                   8      0   100%
ontolearn/utils/static_funcs.py               77     31    60%   63-79, 102-106, 124-135, 151, 180
ontolearn/value_splitter.py                  159      6    96%   111-113, 118, 127, 130
------------------------------------------------------------------------
TOTAL                                       4123    561    86%
```