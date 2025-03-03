# Further Resources

You can find more details in the related papers for each algorithm:

Concept Learning:

- **TDL** &rarr; Tree-based OWL Class Expression Learner for Large Graphs (manuscript will be added soon)
- **Drill** &rarr; [Neuro-Symbolic Class Expression Learning](https://www.ijcai.org/proceedings/2023/0403.pdf)
- **EvoLearner** &rarr; [EvoLearner: Learning Description Logics with Evolutionary Algorithms](https://dl.acm.org/doi/abs/10.1145/3485447.3511925)
- **NCES2** &rarr; [Neural Class Expression Synthesis in ALCHIQ(D)](https://papers.dice-research.org/2023/ECML_NCES2/NCES2_public.pdf)
- **ROCES** &rarr; [Robust Class Expression Synthesis in Description Logics via Iterative Sampling](https://www.ijcai.org/proceedings/2024/0479.pdf)
- **NCES** &rarr; [Neural Class Expression Synthesis](https://link.springer.com/chapter/10.1007/978-3-031-33455-9_13) 
- **NERO*** &rarr; (soon) [Learning Permutation-Invariant Embeddings for Description Logic Concepts](https://link.springer.com/chapter/10.1007/978-3-031-30047-9_9)
- **CLIP** &rarr; [Learning Concept Lengths Accelerates Concept Learning in ALC](https://link.springer.com/chapter/10.1007/978-3-031-06981-9_14)
- **CELOE** &rarr; [Class Expression Learning for Ontology Engineering](https://www.sciencedirect.com/science/article/abs/pii/S1570826811000023)
- **OCEL** &rarr; A limited version of CELOE

&ast;  _Not implemented in our library yet._

Sampling:
- **OntoSample** &rarr; [Accelerating Concept Learning via Sampling](https://dl.acm.org/doi/10.1145/3583780.3615158)

Also check Owlapy's documentation [here](https://dice-group.github.io/owlapy/usage/main.html).


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
[examples](https://github.com/dice-group/Ontolearn/tree/master/examples) and 
[tests](https://github.com/dice-group/Ontolearn/tree/master/tests).

## Contribution

We try to keep documentation up to day to the latest changes, but sometimes we may
overlook some details or make mistakes. If you notice any of such things please let us know :).
As for coding part, feel free to create a pull request and our developers will take a look 
on it. We appreciate your commitment. 

## Questions

In case you have any question or issue you are welcomed to open an issue on our [GitHub issues page](https://github.com/dice-group/Ontolearn/issues).
You can also reach us privately in any of the emails below:

- [cdemir@mail.uni-paderborn.de](mailto:cdemir@mail.uni-paderborn.de) 
- [alkid@mail.uni-paderborn.de](mailto:alkid@mail.uni-paderborn.de)


## Code Coverage

The coverage report is generated using [coverage.py](https://coverage.readthedocs.io/en) for Ontolearn v0.9.0.


```
Name                                          Stmts   Miss  Cover   Missing
---------------------------------------------------------------------------
examples/retrieval_eval.py                      112     16    86%   78, 83, 123, 221, 277-290
examples/retrieval_eval_under_incomplete.py     124     31    75%   78-83, 116, 141-144, 196-219, 235-247
ontolearn/__init__.py                             1      0   100%
ontolearn/abstracts.py                           59      3    95%   193-195
ontolearn/base_concept_learner.py               154      2    99%   310, 314
ontolearn/base_nces.py                           78      5    94%   66, 91, 104-105, 113
ontolearn/clip_architectures.py                  91      0   100%
ontolearn/clip_trainer.py                        89      7    92%   79, 88, 91, 96, 103, 116, 139
ontolearn/concept_generator.py                   95     26    73%   63-72, 78-88, 173-174, 221-222, 251-252
ontolearn/concept_learner.py                    811    120    85%   370-371, 431, 442, 451, 612, 634, 636, 641, 682-686, 723, 734, 754, 769, 777, 787, 789, 831, 838, 843-845, 868-869, 883-885, 903-905, 909-923, 961-964, 969-976, 996-997, 1007-1011, 1051-1052, 1054-1057, 1064-1066, 1157, 1218, 1240-1241, 1245-1263, 1279-1283, 1307-1325, 1341-1342, 1351-1355, 1402, 1409-1411, 1506
ontolearn/data_struct.py                        132     53    60%   179-180, 411, 417-445, 464, 470-499, 516-518
ontolearn/ea_algorithms.py                       57      1    98%   93
ontolearn/ea_initialization.py                  216      7    97%   93, 97, 310-315
ontolearn/ea_utils.py                            88      5    94%   93, 110-111, 114-115
ontolearn/fitness_functions.py                   13      0   100%
ontolearn/heuristics.py                          45      0   100%
ontolearn/incomplete_kb.py                       79     66    16%   47-74, 115, 134-223
ontolearn/knowledge_base.py                     234     18    92%   107-108, 115, 400-401, 436, 444, 447, 453, 516, 561, 639, 773-774, 804, 814, 823, 872
ontolearn/learners/__init__.py                    5      0   100%
ontolearn/learners/celoe.py                     167     25    85%   158, 183, 237, 241, 314-318, 332, 335-360
ontolearn/learners/drill.py                      31      0   100%
ontolearn/learners/ocel.py                       21      0   100%
ontolearn/learners/tree_learner.py              193     28    85%   160, 243-273, 361, 368, 370-374, 390, 393, 414, 423
ontolearn/learning_problem.py                    55      9    84%   98, 119, 129, 135-140
ontolearn/learning_problem_generator.py          17      0   100%
ontolearn/lp_generator/__init__.py                2      0   100%
ontolearn/lp_generator/generate_data.py           8      0   100%
ontolearn/lp_generator/helper_classes.py        106      4    96%   85, 111, 145-146
ontolearn/metrics.py                             50      0   100%
ontolearn/nces_architectures.py                  73      0   100%
ontolearn/nces_modules.py                       143     29    80%   44-45, 68-69, 72, 200-203, 213-242, 245-246
ontolearn/nces_trainer.py                       196     12    94%   72, 76, 85, 89, 174, 181-183, 204, 219-221
ontolearn/nces_utils.py                          99     62    37%   58-59, 64-82, 89-141, 147, 156
ontolearn/owl_neural_reasoner.py                178     21    88%   94, 101, 121, 127, 133, 137, 165-173, 196, 240, 251, 256, 271, 399-402
ontolearn/quality_funcs.py                       39     27    31%   32-56, 60-69
ontolearn/refinement_operators.py               519     33    94%   165-166, 217-226, 296, 397-398, 444, 538, 562, 596-598, 743, 779, 885, 913, 958-960, 967, 988-990, 992, 994, 1062, 1084
ontolearn/search.py                             293     43    85%   69, 132, 163-170, 195, 215, 264, 302, 306, 309, 338, 391, 411, 428, 432, 440, 451-452, 455-463, 466, 481, 483, 508, 510, 575-576, 665-666, 761, 765, 769
ontolearn/semantic_caching.py                   379     80    79%   57-156, 174-179, 200, 202, 208-218, 228, 251, 268, 281, 285, 326-327, 343, 352-353, 358-360, 383-385, 394, 403, 411-413, 422, 475-477, 488-489, 497, 526, 545, 560, 640
ontolearn/utils/__init__.py                      33      1    97%   98
ontolearn/utils/log_config.py                    19      0   100%
ontolearn/utils/oplogging.py                      8      0   100%
ontolearn/utils/static_funcs.py                 113     26    77%   55, 66, 140, 172-177, 218-219, 234-251
ontolearn/value_splitter.py                     159      6    96%   111-113, 118, 127, 130
---------------------------------------------------------------------------
TOTAL                                          5384    766    86%
```