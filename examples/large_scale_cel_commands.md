- With one command
1. For TDL
```bash
git clone https://github.com/dice-group/Ontolearn.git && cd Ontolearn && conda create -n venv python=3.10.14 --no-default-packages --y && conda activate venv && pip install -e . && wget https://files.dice-research.org/projects/Ontolearn/LPs.zip -O ./LPs.zip && unzip LPs.zip && python examples/dbpedia_concept_learning_with_ontolearn.py tdl
```
2. For Drill
```bash
git clone https://github.com/dice-group/Ontolearn.git && cd Ontolearn && conda create -n venv python=3.10.14 --no-default-packages --y && conda activate venv && pip install -e . && wget https://files.dice-research.org/projects/Ontolearn/LPs.zip -O ./LPs.zip && unzip LPs.zip && python examples/dbpedia_concept_learning_with_ontolearn.py drill
```
3. For DL-Learner
```bash
git clone https://github.com/dice-group/Ontolearn.git && cd Ontolearn && conda create -n venv python=3.10.14 --no-default-packages --y && conda activate venv && pip install -e . && wget https://github.com/SmartDataAnalytics/DL-Learner/releases/download/1.4.0/dllearner-1.4.0.zip -O ./dllearner-1.4.0.zip && unzip dllearner-1.4.0.zip && wget https://files.dice-research.org/projects/Ontolearn/LPs.zip -O ./LPs.zip && unzip LPs.zip && python examples/dbpedia_concept_learning_with_dllearner.py
```
