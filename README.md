# Ontolearn

*Ontolearn* is an open-source software library for explainable structured machine learning in Python.
It contains the following (ready-to-apply) algorithms that learn OWL class expressions from positive and negative examples:
- **NCES2** &rarr; (soon) [Neural Class Expression Synthesis in ALCHIQ(D)](https://papers.dice-research.org/2023/ECML_NCES2/NCES2_public.pdf)
- **Drill** &rarr; [Deep Reinforcement Learning for Refinement Operators in ALC](https://arxiv.org/pdf/2106.15373.pdf)
- **NCES** &rarr; [Neural Class Expression Synthesis](https://link.springer.com/chapter/10.1007/978-3-031-33455-9_13)
- **NERO** &rarr; (soon) [Learning Permutation-Invariant Embeddings for Description Logic Concepts](https://github.com/dice-group/Nero)
- **EvoLearner** &rarr; [An evolutionary approach to learn concepts in ALCQ(D)](https://dl.acm.org/doi/abs/10.1145/3485447.3511925)
- **CLIP** &rarr; (soon) [Learning Concept Lengths Accelerates Concept Learning in ALC](https://link.springer.com/chapter/10.1007/978-3-031-06981-9_14)
- **CELOE** &rarr; [Class Expression Learning for Ontology Engineering](https://www.sciencedirect.com/science/article/abs/pii/S1570826811000023)
- **OCEL** &rarr; A limited version of CELOE

You can find more details about *Ontolearn* and these algorithms and their variations in the [documentation](https://ontolearn-docs-dice-group.netlify.app/index.html).

Quick navigation: 
- [Installation](#installation)
- [Usage](#usage)
- [Relevant Papers](#relevant-papers)
## Installation
For detailed instructions please refer to the [installation guide](https://ontolearn-docs-dice-group.netlify.app/usage/installation.html) in the documentation.

### Installation from source

Make sure to set up a virtual python environment like [Anaconda](https://www.anaconda.com/) 
before continuing with the installation. 


To successfully pass all the tests you need to download some external resources in advance 
(see [_Download external files_](#download-external-files-link-files)). We recommend to
download them all. Also, install _java_ and _curl_ if you don't have them in your system:

```commandline
sudo apt install openjdk-11-jdk
sudo apt install curl
```

A quick start up will be as follows:

```shell
git clone https://github.com/dice-group/Ontolearn.git && conda create --name onto python=3.8 && conda activate onto 
# Incase needed
# conda env update --name onto
python -c 'from setuptools import setup; setup()' develop
python -c "import ontolearn"
python -m pytest tests # Partial test with pytest
tox  # full test with tox
```
#### Installation via pip

```shell
pip install ontolearn  # more on https://pypi.org/project/ontolearn/
```
## Usage

In the [examples](https://github.com/dice-group/Ontolearn/tree/develop/examples) folder, you can find examples on how to use
the learning algorithms. Also in the [tests](https://github.com/dice-group/Ontolearn/tree/develop/tests) folder we have added some test cases.

For more detailed instructions we suggest to follow the [guides](https://ontolearn-docs-dice-group.netlify.app/usage/03_algorithm.html) in the documentation.

Below we give a simple example on using CELOE to learn class expressions for a small dataset.
```python
from ontolearn.concept_learner import CELOE
from ontolearn.model_adapter import ModelAdapter
from owlapy.model import OWLNamedIndividual, IRI
from owlapy.namespaces import Namespaces
from owlapy.render import DLSyntaxObjectRenderer
from examples.experiments_standard import ClosedWorld_ReasonerFactory

NS = Namespaces('ex', 'http://example.com/father#')

positive_examples = {OWLNamedIndividual(IRI.create(NS, 'stefan')),
                     OWLNamedIndividual(IRI.create(NS, 'markus')),
                     OWLNamedIndividual(IRI.create(NS, 'martin'))}
negative_examples = {OWLNamedIndividual(IRI.create(NS, 'heinz')),
                     OWLNamedIndividual(IRI.create(NS, 'anna')),
                     OWLNamedIndividual(IRI.create(NS, 'michelle'))}

# Only the class of the learning algorithm is specified
model = ModelAdapter(learner_type=CELOE,
                     reasoner_factory=ClosedWorld_ReasonerFactory,
                     path="KGs/father.owl")

model.fit(pos=positive_examples,
          neg=negative_examples)

dlsr = DLSyntaxObjectRenderer()

for desc in model.best_hypotheses(1):
    print('The result:', dlsr.render(desc.concept), 'has quality', desc.quality)
```
The goal in this example is to learn a class expression for the concept "father". 
The output is as follows:
```
The result: (¬female) ⊓ (∃ hasChild.⊤) has quality 1.0
```

NCES can be used as follows (you can also download all datasets and pretrained models as described in the next section)
```shell
wget https://hobbitdata.informatik.uni-leipzig.de/NCES_Ontolearn_Data/NCESFamilyData.zip -O NCESFamilyData.zip
unzip -o NCESFamilyData.zip
rm -f NCESFamilyData.zip
```
```python
from ontolearn.concept_learner import NCES
from ontolearn.knowledge_base import KnowledgeBase
from owlapy.parser import DLSyntaxParser
from owlapy.render import DLSyntaxObjectRenderer
import sys
sys.path.append("examples/")
from quality_functions import quality
import time

nces = NCES(knowledge_base_path="NCESData/family/family.owl", learner_name="SetTransformer",\
            path_of_embeddings="NCESData/family/embeddings/ConEx_entity_embeddings.csv",load_pretrained=True,\
            max_length=48, proj_dim=128, rnn_n_layers=2, drop_prob=0.1, num_heads=4, num_seeds=1, num_inds=32,\
            pretrained_model_name=["SetTransformer", "LSTM", "GRU"])

KB = KnowledgeBase(path=nces.knowledge_base_path)
dl_syntax_renderer = DLSyntaxObjectRenderer()
dl_parser = DLSyntaxParser(nces.kb_namespace)
brother = dl_parser.parse('Brother')
daughter = dl_parser.parse('Daughter')

pos = set(KB.individuals(brother)).union(set(KB.individuals(daughter)))
neg = set(KB.individuals())-set(pos)

t0 = time.time()
concept = nces.fit(pos, neg)
# Use NCES to synthesize the solution class expression.
# Note that NCES is not given the concepts Brother and Daughter.
# Yet, it is able to compute the exact solution!
t1 = time.time()
print("Duration: ", t1-t0, " seconds")
print("\nPrediction: ", dl_syntax_renderer.render(concept))
quality(KB, concept, pos, neg)
```

```
Duration: 0.5029337406158447  seconds
```

```
Prediction: Brother ⊔ Daughter
```

```
Accuracy: 100.0%
Precision: 100.0%
Recall: 100.0%
F1: 100.0%
```

----------------------------------------------------------------------------

#### Download external files (.link files)

Some resources like pre-calculated embeddings or `pre_trained_agents`
are not included in the Git repository directly. Use the following
command to download them from our data server.

For Drill:
```shell
./big_gitext/download_big.sh examples/pre_trained_agents.zip.link
./big_gitext/download_big.sh -A  # to download them all into examples folder
```

For NCES:
```shell
./big_gitext/download_nces_data
```

To update or upload resource files, follow the instructions
[here](https://github.com/dice-group/Ontolearn-internal/wiki/Upload-big-data-to-hobbitdata)
and use the following command (only for Drill):

```shell
./big_gitext/upload_big.sh pre_trained_agents.zip
```
----------------------------------------------------------------------------
#### Building (sdist and bdist_wheel)
You can use <code>tox</code> to build sdist and bdist_wheel packages for Ontolearn.
- "sdist" is short for "source distribution" and is useful for distribution of packages that will be installed from source.
- "bdist_wheel" is short for "built distribution wheel" and is useful for distributing packages that include large amounts of compiled code, as well as for distributing packages that have complex dependencies.

To build and compile the necessary components of Ontolearn, use:
```shell
tox -e build
```

To automatically build and test the documentation of Ontolearn, use:
```shell
tox -e docs
```

----------------------------------------------------------------------------

#### Simple Linting

Using the following command will run the linting tool [flake8](https://flake8.pycqa.org/) on the source code.
```shell
tox -e lint --
```
----------------------------------------------------------------------------

#### Contribution
Feel free to create a pull request!


## Relevant papers

- [NCES2](https://papers.dice-research.org/2023/ECML_NCES2/NCES2_public.pdf): Neural Class Expression Synthesis in ALCHIQ(D)
- [NCES](https://link.springer.com/chapter/10.1007/978-3-031-33455-9_13): Neural Class Expression Synthesis
- [Evolearner](https://doi.org/10.1145/3485447.3511925): Learning description logics with evolutionary algorithms.
- [CLIP](https://link.springer.com/chapter/10.1007/978-3-031-06981-9_14): Learning Concept Lengths Accelerates Concept Learning in ALC.
### Citing
Currently, we are working on our manuscript describing our framework. 
If you find our work useful in your research, please consider citing the respective paper:
```

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

For any further questions, please contact:  ```onto-learn@lists.uni-paderborn.de```
