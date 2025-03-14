# Tree-based OWL Class Expression Learner over Large Graphs

This repository contains the implementation of the paper "Tree-based OWL Class Expression Learner over Large Graphs"


## Installation

1. Download the repository to your machine using the 'Download Repository' button or [click here](https://anonymous.4open.science/r/TDL-AED8)
2. Navigage into the project directory: <br>
   - ``` Ontolearn-BE38 ```
3. Download the knowledge graphs and the learning problems:
   - ```wget https://files.dice-research.org/projects/Ontolearn/KGs.zip -O ./KGs.zip && unzip KGs.zip```
   - ```wget https://files.dice-research.org/projects/Ontolearn/LPs.zip -O ./LPs.zip && unzip LPs.zip```
4. Set up and activate a virtual environment (optional but recommended):
   - ``` conda create -n .venv python=3.10.13 --no-default-packages && conda activate .venv ```
5. Install the required dependencies:
   - ``` pip install -e . ```
6. To run the project 
   - ``` python3 main.py```

## Benchmark Results

### 10-Fold Cross Validation Family Benchmark Results
 
```shell
python3 examples/concept_learning_cv_evaluation.py --lps LPs/Family/lps.json --kb KGs/Family/family-benchmark_rich_background.owl --max_runtime 60 --report family_results.csv 
```

### 10-Fold Cross Validation Mutagenesis Benchmark Results
```shell
python3 examples/concept_learning_cv_evaluation.py --lps LPs/Mutagenesis/lps.json --kb KGs/Mutagenesis/mutagenesis.owl --max_runtime 60 --report mutagenesis_results.csv 
```


### 10-Fold Cross Validation Carcinogenesis Benchmark Results
```shell
python3 examples/concept_learning_cv_evaluation.py --lps LPs/Carcinogenesis/lps.json --kb KGs/Carcinogenesis/carcinogenesis.owl --max_runtime 60 --report carcinogenesis.csv 
```
