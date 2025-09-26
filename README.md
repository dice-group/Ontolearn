
# PACEL: Preference-Aware Class Expression Learning

This repository shows the implementatation of refinement based CEL algorithms with preference awareness. Our implementation is based
on the open source Framework Ontolearn.



## Installation

```shell
git clone (please add the link to this repo) && cd Ontolearn && unzip data.zip
# To create a virtual python env with conda 
conda create -n venv python=3.10.14 --no-default-packages && conda activate venv && pip install -e .
```
 

## Create a SPARQL endpoint to host the data

One way to obtain preference score of all individuals in the KBS, is to create a Sparql endpoint to host the datasets in a localhost.
The code is meant to get Sparql endpoint and get preference of individuals in both datasets.
To create a Sparql endpoint, make sure you have Apache Jena and Apache Jena Fuseki installed and replace the path below with your actual path.

```shell
Load the data: /home/dice/Downloads/Apache/apache-jena-5.4.0/bin/tdb2.tdbloader --loader=parallel --loc databases/imdb_10000 /home/dice/Downloads/IMDB/imdb_100.owl
Run the server: java -Xmx4g -jar fuseki-server.jar --tdb2 --loc=./databases/imdb_10000 --port=3030 /imdb_10000
```

## Model available

The current implementation of PACEL are only on refinement based class expression learner. In this repository, we have the following refinement based models available
CLIP, OCEL, CELOE and their preference aware extension CLIP_Pref, CELOE_Pref. 

```shell
CLIP, OCEL, CELOE, CLIP_Pref, CELOE_Pref, 
```

## How to run?

Once all the preliminary settings are done, the code is ready to run.

To run the CELOE algorithm with personas generated LPS on the Spotify dataset.

```shell
python concept_learning_evaluation_imdb.py --algorithm CELOE --max_runtime 300 --lps LPs/Music/lps_personas.json --url http://localhost:3030/music_10000/sparql --kb KGs/Music/music_10000.owl --report spotify_personas
```

Here,  
- `--max_runtime` is the maximum runtime of the algorithm
- `--algorithm` control the algorithm to be run, to choose among `[CLIP, OCEL, CELOE, CLIP_Pref, CELOE_Pref, OCEL_Pref]` for a different algorithm
- `--lps` is the path to the learning problems. Per dataset, two types of learning problems are available (Personas, and generated), check the [LPs](LPs) directory to choose.
- `--url` is the sparql endpoint of the data of interest. Replace with yours
- `--kb` is the path to the dataset knowledge base, check the [KGs](KGs) directory to choose either the Spotify or IMDB dataset.
- `--report` is the file name to be saved. e.g. for the code above the results will be save in this current directory with name spotify_personas_CELOE.csv
