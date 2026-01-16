# Further Resources

You can find more details in the related papers for each algorithm:

Concept Learning:

- **TDL** &rarr; [Tree-based OWL Class Expression Learner for Large Graphs](https://dl.acm.org/doi/10.1007/978-3-032-06066-2_29)
- **Drill** &rarr; [Neuro-Symbolic Class Expression Learning](https://www.ijcai.org/proceedings/2023/0403.pdf)
- **EvoLearner** &rarr; [EvoLearner: Learning Description Logics with Evolutionary Algorithms](https://dl.acm.org/doi/abs/10.1145/3485447.3511925)
- **NCES2** &rarr; [Neural Class Expression Synthesis in ALCHIQ(D)](https://papers.dice-research.org/2023/ECML_NCES2/NCES2_public.pdf)
- **ROCES** &rarr; [Robust Class Expression Synthesis in Description Logics via Iterative Sampling](https://www.ijcai.org/proceedings/2024/0479.pdf)
- **NCES** &rarr; [Neural Class Expression Synthesis](https://link.springer.com/chapter/10.1007/978-3-031-33455-9_13) 
- **NERO** &rarr; [Learning Permutation-Invariant Embeddings for Description Logic Concepts](https://link.springer.com/chapter/10.1007/978-3-031-30047-9_9)
- **CLIP** &rarr; [Learning Concept Lengths Accelerates Concept Learning in ALC](https://link.springer.com/chapter/10.1007/978-3-031-06981-9_14)
- **CELOE** &rarr; [Class Expression Learning for Ontology Engineering](https://www.sciencedirect.com/science/article/abs/pii/S1570826811000023)
- **OCEL** &rarr; A limited version of CELOE
- **SPELL** &rarr; [SAT-Based PAC Learning of Description Logic Concepts](https://www.ijcai.org/proceedings/2023/0373.pdf)
- **ALCSAT** &rarr; [SAT-Based Bounded Fitting for the Description Logic $\mathcal{ALC}$](https://arxiv.org/pdf/2507.21752)


Sampling:
- **OntoSample** &rarr; [Accelerating Concept Learning via Sampling](https://dl.acm.org/doi/10.1145/3583780.3615158)

Also check Owlapy's documentation [here](https://dice-group.github.io/owlapy/usage/main.html).


## Citing
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

# TDL
@InProceedings{10.1007/978-3-032-06066-2_29,
author={Demir, Caglar and Yekini, Moshood and R{\"o}der, Michael and Mahmood, Yasir and Ngonga Ngomo, Axel-Cyrille},
editor={Ribeiro, Rita P. and Pfahringer, Bernhard and Japkowicz, Nathalie and Larra{\~{n}}aga, Pedro and Jorge, Al{\'i}pio M. and Soares, Carlos and Abreu, Pedro H. and Gama, Jo{\~a}o},
title={Tree-Based OWL Class Expression Learner over Large Graphs},
booktitle={Machine Learning and Knowledge Discovery in Databases. Research Track},
year={2026},
publisher={Springer Nature Switzerland},
address={Cham},
pages={495--511},
isbn={978-3-032-06066-2}
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

# NERO
@InProceedings{10.1007/978-3-031-30047-9_9,
author="Demir, Caglar
and Ngonga Ngomo, Axel-Cyrille",
editor="Cr{\'e}milleux, Bruno
and Hess, Sibylle
and Nijssen, Siegfried",
title="Learning Permutation-Invariant Embeddings for Description Logic Concepts",
booktitle="Advances in Intelligent Data Analysis XXI",
year="2023",
publisher="Springer Nature Switzerland",
address="Cham",
pages="103--115",
abstract="Concept learning deals with learning description logic concepts from a background knowledge and input examples. The goal is to learn a concept that covers all positive examples, while not covering any negative examples. This non-trivial task is often formulated as a search problem within an infinite quasi-ordered concept space. Although state-of-the-art models have been successfully applied to tackle this problem, their large-scale applications have been severely hindered due to their excessive exploration incurring impractical runtimes. Here, we propose a remedy for this limitation. We reformulate the learning problem as a multi-label classification problem and propose a neural embedding model (NERO) that learns permutation-invariant embeddings for sets of examples tailored towards predicting {\$}{\$}F{\_}1{\$}{\$}F1scores of pre-selected description logic concepts. By ranking such concepts in descending order of predicted scores, a possible goal concept can be detected within few retrieval operations, i.e., no excessive exploration. Importantly, top-ranked concepts can be used to start the search procedure of state-of-the-art symbolic models in multiple advantageous regions of a concept space, rather than starting it in the most general concept {\$}{\$}{\backslash}top {\$}{\$}⊤. Our experiments on 5 benchmark datasets with 770 learning problems firmly suggest that NERO significantly (p-value {\$}{\$}<1{\backslash}{\%}{\$}{\$}<1{\%}) outperforms the state-of-the-art models in terms of {\$}{\$}F{\_}1{\$}{\$}F1score, the number of explored concepts, and the total runtime. We provide an open-source implementation of our approach (https://github.com/dice-group/Nero).",
isbn="978-3-031-30047-9"
}

# OWLAPY
@misc{baci2025owlapypythonicframeworkowl,
      title={OWLAPY: A Pythonic Framework for OWL Ontology Engineering}, 
      author={Alkid Baci and Luke Friedrichs and Caglar Demir and Axel-Cyrille Ngonga Ngomo},
      year={2025},
      eprint={2511.08232},
      archivePrefix={arXiv},
      primaryClass={cs.SE},
      url={https://arxiv.org/abs/2511.08232}, 
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

We try to keep documentation up to date with the latest changes, but sometimes we may
overlook some details or make mistakes. If you notice any of such things please let us know :).
As for coding part, feel free to create a pull request and our developers will take a look 
at it. We appreciate your commitment. 

## Questions

In case you have any question or issue you are welcomed to open an issue on our [GitHub issues page](https://github.com/dice-group/Ontolearn/issues).
You can also reach us privately in any of the emails below:

- [cdemir@mail.uni-paderborn.de](mailto:cdemir@mail.uni-paderborn.de) 
- [alkid@mail.uni-paderborn.de](mailto:alkid@mail.uni-paderborn.de)


## Code Coverage

The coverage report is generated using [coverage.py](https://coverage.readthedocs.io/en) for Ontolearn v0.10.0.


```
Name                                          Stmts   Miss  Cover   Missing
---------------------------------------------------------------------------
examples/retrieval_eval.py                      112     16    86%   78, 83, 123, 221, 277-290
examples/retrieval_eval_under_incomplete.py     119    102    14%   52-83, 87-224, 230-242
ontolearn/__init__.py                             1      0   100%
ontolearn/abstracts.py                           59      3    95%   193-195
ontolearn/base_concept_learner.py               153     13    92%   307, 311, 351-352, 390-398
ontolearn/base_nces.py                           78      5    94%   66, 91, 104-105, 113
ontolearn/clip_architectures.py                  91      0   100%
ontolearn/clip_trainer.py                        89      7    92%   79, 88, 91, 96, 103, 116, 139
ontolearn/concept_generator.py                   95     26    73%   63-72, 78-88, 173-174, 221-222, 251-252
ontolearn/concept_learner.py                    813    122    85%   372-373, 433, 444, 453, 614, 636, 638, 643, 684-688, 725, 736, 756, 771, 779, 789, 791, 833, 840, 845-847, 870-871, 885-887, 905-907, 911-925, 963-966, 971-978, 998-999, 1009-1013, 1053-1054, 1056-1059, 1066-1068, 1159, 1220, 1242-1243, 1247-1265, 1281-1285, 1309-1327, 1343-1344, 1353-1357, 1404, 1411-1413, 1508, 1536-1537
ontolearn/data_struct.py                        132     53    60%   179-180, 411, 417-445, 464, 470-499, 516-518
ontolearn/ea_algorithms.py                       57      1    98%   93
ontolearn/ea_initialization.py                  219      8    96%   94, 98, 246, 313-318
ontolearn/ea_utils.py                            88      5    94%   93, 110-111, 114-115
ontolearn/fitness_functions.py                   13      0   100%
ontolearn/heuristics.py                          45      0   100%
ontolearn/incomplete_kb.py                       79     73     8%   47-74, 99-118, 134-223
ontolearn/knowledge_base.py                     238     20    92%   99-103, 109, 407-408, 442, 450, 453, 459, 522, 567, 645, 779-780, 810, 820, 829, 878, 968
ontolearn/learners/__init__.py                    5      0   100%
ontolearn/learners/celoe.py                     167     25    85%   158, 183, 237, 241, 314-318, 332, 335-360
ontolearn/learners/drill.py                      31      0   100%
ontolearn/learners/ocel.py                       21      0   100%
ontolearn/learners/tree_learner.py              193     27    86%   160, 243-273, 361, 368, 370-374, 390, 393, 414
ontolearn/learning_problem.py                    55      9    84%   98, 119, 129, 135-140
ontolearn/learning_problem_generator.py          16      0   100%
ontolearn/lp_generator/__init__.py                2      0   100%
ontolearn/lp_generator/generate_data.py           8      0   100%
ontolearn/lp_generator/helper_classes.py        106      4    96%   85, 111, 145-146
ontolearn/metrics.py                             50      0   100%
ontolearn/nces_architectures.py                  73      0   100%
ontolearn/nces_modules.py                       143     29    80%   44-45, 68-69, 72, 200-203, 213-242, 245-246
ontolearn/nces_trainer.py                       196     12    94%   72, 76, 85, 89, 174, 181-183, 204, 219-221
ontolearn/nces_utils.py                          99     60    39%   64-82, 89-141, 147, 156
ontolearn/owl_neural_reasoner.py                178     22    88%   72-94, 101, 121, 127, 133, 137, 165-173, 196, 240, 251, 256, 271, 399-402
ontolearn/quality_funcs.py                       39     27    31%   32-56, 60-69
ontolearn/refinement_operators.py               519     25    95%   165-166, 296, 397-398, 444, 538, 562, 596-598, 743, 779, 885, 913, 958-960, 967, 988-990, 992, 994, 1062, 1084
ontolearn/search.py                             293     43    85%   69, 132, 163-170, 195, 215, 264, 302, 306, 309, 338, 391, 411, 428, 432, 440, 451-452, 455-463, 466, 481, 483, 508, 510, 575-576, 665-666, 761, 765, 769
ontolearn/triple_store.py                       501    237    53%   102-103, 121-122, 134-135, 151, 154-160, 167, 215-218, 225-230, 233-235, 242, 248-254, 284-286, 289, 304, 307-311, 323-327, 330-334, 339-341, 350-353, 364-368, 371-373, 376-385, 388-390, 397-398, 402-411, 454-456, 484-495, 503-515, 518-522, 525-529, 532-533, 536-537, 541-549, 553-561, 567, 571, 586-590, 594, 642, 646, 652, 663-667, 712, 738, 751, 795, 798, 809-811, 814, 818-830, 833, 836, 839, 842-844, 856-860, 889-890, 893, 896-897, 900-901, 904-905, 909, 912-913, 917, 921, 924, 928-933, 937-951, 959-965, 974-981, 985-991, 994, 997, 1000, 1003, 1006, 1009, 1012, 1015, 1020-1025, 1030-1035, 1041-1046, 1052-1057, 1060, 1068, 1072-1073, 1076-1077, 1082, 1087
ontolearn/utils/__init__.py                      33      2    94%   58, 98
ontolearn/utils/log_config.py                    19      0   100%
ontolearn/utils/oplogging.py                      8      0   100%
ontolearn/utils/static_funcs.py                 111     26    77%   53, 64, 138, 170-175, 216-217, 232-249
ontolearn/value_splitter.py                     159      6    96%   111-113, 118, 127, 130
---------------------------------------------------------------------------
TOTAL                                          5506   1008    82%
```