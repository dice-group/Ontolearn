
## Neural Description Logic Reasoning over incomplete Knowledge Base

This repository provides the implementation of our reasoner EBR. With this repository, once can perform instance retrieval even within an incomplete and inconsistent knowldege base. EBR leverages KGE to perform reasoning over incomplete and inconsistent knowledge bases (KBs). We employ a neural link predictor to facilitate the retrieval of missing data and handle inconsistencies.

## Installation


```shell
git clone https://github.com/dice-group/Ontolearn.git -b retrieval_eval_incomplete
# To create a virtual python env with conda 
conda create -n venv python=3.10.14 --no-default-packages && conda activate venv && pip install -e . && cd Ontolearn
# To download knowledge graphs
wget https://files.dice-research.org/projects/Ontolearn/KGs.zip -O ./KGs.zip && unzip KGs.zip
```
Other datasets used in the paper can be found [here](https://files.dice-research.org/projects/NCES/NCES/datasets.zip)

## Retrieval results from Table 2 

To reproduce our results, run the commands below

```shell
python examples/retrieval_eval.py --path_kg "KGs/Family/father.owl"
# Results of the Father dataset

python examples/retrieval_eval.py --path_kg "KGs/Family/family-benchmark_rich_background.owl"
# Results of the Family dataset
```

For larger datasets, we have to sample the number of entities and relations. For the experiments to run fast, we need to select the type of instance we are interested from line 136-140 of this [file](examples/retrieval_eval.py). Below we only present how to get results on semnatic Bible but for other datasets can be obtain similarly by adding the corect path to the argument ```--path_kg```.

```shell
# results on the semnatic bible data

python examples/retrieval_eval.py --path_kg "KGs/Semantic_bible/semantic_bible.owl" --seed 1 --ratio_sample_nc 1 --ratio_sample_object_prob 1 --path_report "ALCQI_semantic_seed_all_nc.csv"
# OWLClass expressions

python examples/retrieval_eval.py --path_kg "KGs/Semantic_bible/semantic_bible.owl" --seed 1 --ratio_sample_nc .5 --ratio_sample_object_prob .5 --path_report "ALCQI_semantic_seed_1_ratio_0.5_unions.csv"
# OWLObjectUnionOf

python examples/retrieval_eval.py --path_kg "KGs/Semantic_bible/semantic_bible.owl" --seed 1 --ratio_sample_nc 1 --ratio_sample_object_prob 1 --path_report "ALCQI_semantic_seed_1_interALCQI_semantic_seed_all_nc.csv"
# OWLObjectComplementOf

python examples/retrieval_eval.py --path_kg "KGs/Semantic_bible/semantic_bible.owl" --seed 1 --ratio_sample_nc .5 --ratio_sample_object_prob .5 --path_report "ALCQI_semantic_seed_1_ratio_0.5_inter.csv"
# OWLObjectIntersectionOf

python examples/retrieval_eval.py --path_kg "KGs/Semantic_bible/semantic_bible.owl" --seed 1 --ratio_sample_nc .2 --ratio_sample_object_prob .2 --path_report "ALCQI_semantic_seed_1_ratio_02_exits.csv"
# OWLObjectSomeValuesFrom

python examples/retrieval_eval.py --path_kg "KGs/Semantic_bible/semantic_bible.owl" --seed 1 --ratio_sample_nc .2 --ratio_sample_object_prob .2 --path_report "ALCQI_semantic_seed_1_ratio_02_forall.csv" 
# OWLObjectAllValuesFrom

python examples/retrieval_eval.py --path_kg "KGs/Semantic_bible/semantic_bible.owl" --seed 1 --ratio_sample_nc .1 --ratio_sample_object_prob .1 --path_report "ALCQI_semantic_seed_1_ratio_02_min_card.csv"
# minimum cardinality restrictions, n = {1,2,3} 

python examples/retrieval_eval.py --path_kg "KGs/Semantic_bible/semantic_bible.owl" --seed 1 --ratio_sample_nc .1 --ratio_sample_object_prob .1 --path_report "ALCQI_semantic_seed_1_ratio_02_max_card.csv"
# max cardinality restrictions, n = {1,2,3} 
```

## Results from Table 3

To obtain the results From Table 3, run the following commands:

```shell
python examples/retrieval_eval_under_incomplete.py --path_kg "KGs/Family/father.owl" --level_of_incompleteness 0.4 --operation "incomplete" --number_of_incomplete_graphs 5
# Results of the Father dataset

python examples/retrieval_eval_under_incomplete.py --path_kg "KGs/Family/family-benchmark_rich_background.owl" --level_of_incompleteness 0.4 --operation "incomplete" --number_of_incomplete_graphs 5
# Results of the Family dataset

python examples/retrieval_eval_under_incomplete.py --path_kg "KGs/Semantic_bible/semantic_bible.owl" --level_of_incompleteness 0.4 --operation "incomplete" --number_of_incomplete_graphs 5 --sample Yes
# Results of the Semantic Bible dataset

python examples/retrieval_eval_under_incomplete.py --path_kg "KGs/Mutagenesis/mutagenesis.owl" --level_of_incompleteness 0.4 --operation "incomplete" --number_of_incomplete_graphs 5 --sample Yes
# Results of the Mutagenesis dataset

python examples/retrieval_eval_under_incomplete.py --path_kg "KGs/Mutagenesis/mutagenesis.owl" --level_of_incompleteness 0.4 --operation "incomplete" --number_of_incomplete_graphs 5 --sample Yes
# Results of the Carcinogenesis dataset
```
To get the results with other ratio (0.1, 0.2, 0.6, 0.8, 0.9 etc...), just add it after the argument ```--level_of_incompleteness``` and run the same command. For results on inconcistencies, just change the argument ```--operation``` to "inconsistent" (this will not necessary make the KB inconsistent but will add noises in the data at the choosen level). See below for an example on the Father and Family datasets.

```shell
python examples/retrieval_eval_under_incomplete.py --path_kg "KGs/Family/father.owl" --level_of_incompleteness 0.4 --operation "inconsistent" --number_of_incomplete_graphs 5
# Results of the Father dataset

python examples/retrieval_eval_under_incomplete.py --path_kg "KGs/Family/family-benchmark_rich_background.owl" --level_of_incompleteness 0.4 --operation "inconsistent" --number_of_incomplete_graphs 5
# Results of the Family dataset
```

Or more simply, just create a bash file as shown [here](run_multiple_carcinogenesis.sh) for the carcinogenesis and execute it using 

```shell 
chmod +x run_multiple_carcinogenesis.sh
``` 
This will make the file executable then do 
```shell
./run_multiple_carcinogenesis.sh
```
This will run the carcinogenesis data with different level of inconsistencies.

## Example of Concepts retrieval results on Father dataset:

|   | Expression             | Type                     | Jaccard Similarity | F1  | Runtime Benefits      | Runtime EBR        | Symbolic Retrieval                                                                                                                                               | EBR Retrieval                                                                                                                                         |
|---|------------------------|--------------------------|--------------------|-----|-----------------------|-----------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 0 | female ⊓ male          | OWLObjectIntersectionOf   | 1.0                | 1.0 | 0.054    | 0.003    | set()                                                                                                                                                            | set()                                                                                                                                                            |
| 1 | ∃ hasChild.female       | OWLObjectSomeValuesFrom   | 1.0                | 1.0 | -0.001 | 0.001  | {'http://example.com/father#markus'}                                                                                                                             | {'http://example.com/father#markus'}                                                                                                                             |
| 2 | person ⊔ (¬person)      | OWLObjectUnionOf         | 1.0                | 1.0 | -0.003  | 0.003   | {'http://example.com/father#martin', 'http://example.com/father#stefan', 'http://example.com/father#markus', 'http://example.com/father#anna', 'http://example.com/father#michelle', 'http://example.com/father#heinz'} | {'http://example.com/father#martin', 'http://example.com/father#stefan', 'http://example.com/father#markus', 'http://example.com/father#anna', 'http://example.com/father#michelle', 'http://example.com/father#heinz'} |
| 3 | person ⊓ person         | OWLObjectIntersectionOf  | 1.0                | 1.0 | -0.002   | 0.002    | {'http://example.com/father#martin', 'http://example.com/father#stefan', 'http://example.com/father#markus', 'http://example.com/father#anna', 'http://example.com/father#michelle', 'http://example.com/father#heinz'} | {'http://example.com/father#martin', 'http://example.com/father#stefan', 'http://example.com/father#markus', 'http://example.com/father#anna', 'http://example.com/father#michelle', 'http://example.com/father#heinz'} |
| 4 | person ⊔ person         | OWLObjectUnionOf         | 1.0                | 1.0 | -0.002  | 0.002   | {'http://example.com/father#martin', 'http://example.com/father#stefan', 'http://example.com/father#markus', 'http://example.com/father#anna', 'http://example.com/father#michelle', 'http://example.com/father#heinz'} | {'http://example.com/father#martin', 'http://example.com/father#stefan', 'http://example.com/father#anna', 'http://example.com/father#markus', 'http://example.com/father#michelle', 'http://example.com/father#heinz'} |
