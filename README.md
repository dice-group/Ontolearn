# PruneCEL2: An Efficient Recursive Concept Learner for ALCIQ(D)

> **Acknowledgement** — This project is built upon the open-source framework [Ontolearn](https://github.com/dice-group/Ontolearn).  
> We sincerely thank the Ontolearn development team for their excellent and highly readable codebase, which made this work possible.



## Overview

This repository contains the supplementary material for the paper
"An Efficient Recursive Concept Learner for ALCIQ(D)" 

It provides:

* The implementation of PruneCEL2, a scalable concept learning approach for
  OWL class expressions in the description logic ALCIQ(D).
* The implementation of state-of-the-art concept learning approaches: 
  CELOE, Drill, EvoLearner, NCES2, TDL, ALCSAT.
* The learning problems and knowledge bases required to reproduce our experiments.


## 1. Repository Structure
The following directories and files can be found within this project:
```
- lps:                   The learning problems of Experiment I and Experiment II in json format
- examples:              Implementations and configurations of state-of-the-art concept learners used for comparison experiments.
- PruneCEL_ALCIQD.py:    The source code of PruneCEL2
- ontolearn:             The required dependency code from Ontolearn.
- Update_Query.txt       The SPARQL update queries for materializing the knowledge graph in a triple store.
```

## 2. Running Experiments

### Experiment Setup

PruneCEL2 uses SPARQL queries to retrieve data from the underlying knowledge base. 
For our experiments, we used the triple store [Tentris](https://github.com/dice-group/Tentris). 
However, the experiments can be run with any other triple store ([Fuseki](https://jena.apache.org/documentation/fuseki2/)).
However, using a different triple store can lead to different results since PruneCEL2 moves a large amount of the work to the triple store, serving as oracle.
**Note:** Before running the experiments, the knowledge graph must be fully materialized 
in the triple store. The required SPARQL update queries for materialization are provided 
in the file `Update_Query.txt`.


For the experiments, implementations of the compared approaches (CELOE, DRILL, EvoLearner, NCES2, TDL, and ALCSAT) are provided in the `examples` directory.
All implementations are provided by the [Ontolearn](https://github.com/dice-group/ontolearn) project. 
During our experiments, CELOE, DRILL, ALCSAT, TDL and PruneCEL were set up in a similar way as PruneCEL2, i.e., we provided the address of the SPARQL endpoint and all approaches used SPARQL queries to retrieve the necessary data. 
However, the implementations of EvoLearner and NCES2 do not seem to support this feature at the moment and both have to load the data into memory before they start.
Note that we did not take this loading time into consideration when measuring the runtime of these approaches.


## 3. Installation

### 3.1 Download the source code

The anonymized source code is available at:

https://anonymous.4open.science/r/Ontolearn-C58D

Please download the repository as a ZIP archive from the anonymous repository page and extract it to a local directory.

### 3.2 Install from source

After extracting the project, open a terminal in the project root directory and run:

```shell
conda create -n venv python=3.11 --no-default-packages
conda activate venv
pip install -e .
```



## 4, Run Experiment I 
### 4.1 Overview

We compare PruneCEL2 to PruneCEL, CELOE, Drill, Evolearner, NCES2, TDL, and ALCSAT on the 12 benchmarking datasets 
Family, BioPax, Animal, Mutagenesis, Carcinogenesis, Lymphography, Nctrer, Premier League, Pyrimidine, Hepatitis, Mammographic and Suramin.
The knowledge base and learning problems are provided by [SMLBench, Ontolearn, DLFoil].
We run all approaches with their default configuration and set their maximum runtime for a single learning problem to 60 seconds.

### 4.2 Learning problems

The learning problems of the 12 benchmarking datasets used in Experiment I are 
available from different sources. 10 out of the 12 learning problems are provided 
by [SML-Bench](https://github.com/SmartDataAnalytics/SML-Bench), while the remaining 
2 learning problems can be downloaded using the following command:

```shell
wget https://files.dice-research.org/projects/Ontolearn/LPs.zip -O ./LPs.zip && unzip LPs.zip
```

They are also included in this repository and can be found in the directory `lps/Exp I`.

### 4.3 Knowledge Bases

The knowledge bases of the 12 benchmarking datasets used for experiment I are 
available from different sources. 10 out of the 12 knowledge bases are provided 
by [SML-Bench](https://github.com/SmartDataAnalytics/SML-Bench), while the remaining 
2 knowledge bases can be downloaded using the following command:

```shell
wget https://files.dice-research.org/projects/Ontolearn/KGs.zip -O ./KGs.zip && unzip KGs.zip
```

### 4.4 Other Algorithms

We refer to the [examples](https://github.com/dice-group/Ontolearn/tree/develop/examples) of the Ontolearn project. 


#### Drill

Drill needs an embedding model for each knowledge base. 

The models we used can be found at [DOI: 10.5281/zenodo.21457284](https://zenodo.org/records/21457284). 

We also provide a pre-trained model for Drill [DOI: 10.5281/zenodo.21457432](https://zenodo.org/records/21457433)


#### NCES2

NCES2 needs a trained model for each knowledge base.

The trained models we used can be found at [DOI: 10.5281/zenodo.21457609](https://zenodo.org/records/21457609).



## 5, Run Experiment II 
### 5.1 Overview

We compare PruneCEL2 to PruneCEL, CELOE, Drill, Evolearner, NCES2, TDL, and ALCSAT on the 3 QALD-based benchmarking datasets QALD10, QALD9+DB and QALD9+WK with the learning problems provided by [PruneCEL].
We run all approaches with their default configuration and set their maximum runtime for a single learning problem to 600 seconds.

### 5.2 Learning problems

The learning problems of the three QALD-based datasets used in Experiment II are 
available at DOI: [10.5281/zenodo.16681824](https://doi.org/10.5281/zenodo.16681824). 
They are also included in this repository and can be found in the directory `lps/Exp II`.

### 5.3 Knowledge Bases

The knowledge bases of the three QALD-based benchmarking datasets used in Experiment II 
are available online at [10.5281/zenodo.14720669](https://zenodo.org/records/14720669).

For convenience, the authors of [PruneCEL] provide SPARQL endpoints for querying the 
corresponding knowledge graphs:

| Dataset | SPARQL Endpoint |
|---|---|
| QALD10 | http://expl-gerbil-qa.cs.uni-paderborn.de:9080/sparql |
| QALD9+DB | http://expl-gerbil-qa.cs.uni-paderborn.de:9050/sparql |
| QALD9+WK | http://expl-gerbil-qa.cs.uni-paderborn.de:9070/sparql |

### 5.4 Other Algorithms

Again, we refer to the [examples](https://github.com/dice-group/Ontolearn/tree/develop/examples) of the Ontolearn project. 

#### Drill
DRILL requires a knowledge graph embedding model for each knowledge base.
The embedding models used in our experiments are provided by the previous
paper [PruneCEL] and are available at:
[DOI: 10.5281/zenodo.14720609](https://doi.org/10.5281/zenodo.14720609).

In addition, the pre-trained embedding model for DRILL used in the previous
paper [PruneCEL] can be downloaded from:
[DOI: 10.5281/zenodo.14720524](https://doi.org/10.5281/zenodo.14720524).



## 6. References

[Ontolearn] Demir, Caglar, Alkid Baci, N'Dah Jean Kouagou, Leonie Nora Sieger, Stefan Heindorf, Simon Bin, Lukas Blübaum, Alexander Bigerl, and Axel-Cyrille Ngonga Ngomo.
*Ontolearn---A Framework for Large-scale OWL Class Expression Learning in Python*. Journal of Machine Learning Research 26, no. 63 (2025): 1-6.
Available at: https://www.jmlr.org/papers/v26/24-1113.html

[SMLBench] Westphal, Patrick, Lorenz Bühmann, Simon Bin, Hajira Jabeen, and Jens Lehmann.
*SML-Bench–A benchmarking framework for structured machine learning*. Semantic Web 10, no. 2 (2019): 231-245. 
Available at: https://journals.sagepub.com/doi/full/10.3233/SW-180308

[PruneCEL] Zhang, Quannian, Michael Röder, Nikit Srivastava, N'Dah Jean Kouagou, and Axel-Cyrille Ngonga Ngomo. 
*Explainable Benchmarking through the Lense of Concept Learning*. In Proceedings of the 13th Knowledge Capture Conference 2025, pp. 139-147. 2025. 
Available at: https://dl.acm.org/doi/full/10.1145/3731443.3771359

[NCES2] N’Dah Jean Kouagou, Stefan Heindorf, Caglar Demir, and Axel-Cyrille Ngonga Ngomo.  
*Neural Class Expression Synthesis in ALCHIQ(D) *. In Joint European Conference on Machine Learning and Knowledge Discovery in Databases. Cham: Springer Nature Switzerland, 2023. pp. 196–212.  
Available at: https://link.springer.com/chapter/10.1007/978-3-031-43421-1_12

[CELOE] Jens Lehmann, Sören Auer, Lorenz Bühmann, and Sebastian Tramp.  
*Class expression learning for ontology engineering*. Journal of Web Semantics, 9(1):71–81, 2011.  
https://doi.org/10.1016/j.websem.2011.01.001  
Available at: https://www.sciencedirect.com/science/article/pii/S1570826811000023

[Drill] Caglar Demir and Axel-Cyrille Ngonga Ngomo.  
*Neuro-symbolic class expression learning*.  
In *Proceedings of the Thirty-Second International Joint Conference on Artificial Intelligence (IJCAI '23)*, Macao, P.R. China, 2023.
Available at: https://doi.org/10.24963/ijcai.2023/403

[Evolearner] Stefan Heindorf, Lukas Blübaum, Nick Düsterhus, Till Werner, Varun Nandkumar Golani, Caglar Demir, and Axel-Cyrille Ngonga Ngomo.  
*EvoLearner: Learning Description Logics with Evolutionary Algorithms*.  
In *Proceedings of the ACM Web Conference 2022 (WWW '22)*, Virtual Event, Lyon, France, pp. 818–828. Association for Computing Machinery, New York, NY, USA, 2022.
Available at: https://doi.org/10.1145/3485447.3511925

[TDL] Demir, Caglar, Moshood Yekini, Michael Röder, Yasir Mahmood, and Axel-Cyrille Ngonga Ngomo
*Tree-Based OWL Class Expression Learner over Large Graphs*. In Joint European Conference on Machine Learning and Knowledge Discovery in Databases, pp. 495-511. Cham: Springer Nature Switzerland, 2025. 
Available at: https://link.springer.com/chapter/10.1007/978-3-032-06066-2_29

[ALCSAT] Maurice Funk, Jean Christoph Jung, Tom Voellmer.  
*SAT-Based Bounded Fitting for the Description Logic ALC*.  
In *TInternational Semantic Web Conference 2025*, Springer Nature Switzerland, 2025. pp. 42-60
Available at: https://link.springer.com/chapter/10.1007/978-3-032-09527-5_3

[DLFoil] Fanizzi, Nicola, Giuseppe Rizzo, Claudia d’Amato, and Floriana Esposito.
*DLFoil: Class expression learning revisited*. In European Knowledge Acquisition Workshop, pp. 98-113. Cham: Springer International Publishing, 2018.
Available at: https://link.springer.com/chapter/10.1007/978-3-030-03667-6_7
