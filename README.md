# Ontolearn

*Ontolearn* is an open-source software library for explainable structured machine learning in Python.
It contains the following ready-to-apply algorithms that learn OWL class expressions from positive and negative examples:
- **EvoLearner** &rarr; [An evolutionary approach to learn concepts in ALCQ(D)](https://dl.acm.org/doi/abs/10.1145/3485447.3511925)
- **Drill** &rarr; [Deep Reinforcement Learning for Refinement Operators in ALC](https://arxiv.org/pdf/2106.15373.pdf)
- **CELOE** &rarr; Concept Learning for Refinement Operators in ALC
- **OCEL** &rarr; A limited version of CELOE
- **CLIP** &rarr; (soon) [Learning Concept Lengths Accelerates Concept Learning in ALC](https://link.springer.com/chapter/10.1007/978-3-031-06981-9_14)
- **NCES** &rarr; (soon) [Neural Class Expression Synthesis](https://arxiv.org/pdf/2111.08486.pdf)
- **NERO** &rarr; (soon) [Learning Permutation-Invariant Embeddings for Description Logic Concepts](https://github.com/dice-group/Nero)

You can find more details about *Ontolearn* and these algorithms and their variations in the [documentation](https://ontolearn-docs-dice-group.netlify.app/index.html).

Quick navigation: 
- [Installation](#installation)
- [Usage](#usage)
- [Relevant Papers](#relevant-papers)
## Installation


Note: Make sure to set up a virtual python environment like [Anaconda](https://www.anaconda.com/) before continuing with the installation.
```shell
git clone https://github.com/dice-group/Ontolearn.git
cd Ontolearn
conda create --name temp python=3.8
conda activate temp
conda env update --name temp
python -c 'from setuptools import setup; setup()' develop
python -c "import ontolearn"
python -m pytest tests # Partial test with pytest
tox  # full test with tox
```
For more detailed instructions please refer to the [installation guide](https://ontolearn-docs-dice-group.netlify.app/usage/installation.html) in the documentation.
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
                     path="/KGs/father.owl")

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
----------------------------------------------------------------------------

#### Download external files (.link files)

Some resources like pre-calculated embeddings or `pre_trained_agents`
are not included in the Git repository directly. Use the following
command to download them from our data server.


```shell
./big_gitext/download_big.sh examples/pre_trained_agents.zip.link
./big_gitext/download_big.sh -A  # to download them all into examples folder
```

To update or upload resource files, follow the instructions
[here](https://github.com/dice-group/Ontolearn-internal/wiki/Upload-big-data-to-hobbitdata)
and use the following command.

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

- [Evolearner](https://doi.org/10.1145/3485447.3511925): Learning description logics with evolutionary algorithms.
- [CLIP](https://arxiv.org/abs/2107.04911): Learning Concept Lengths Accelerates Concept Learning in ALC.
### Citing
Currently, we are working on our manuscript describing our framework. 
If you find our work useful in your research, please consider citing the respective paper:
```
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
  author={Kouagou, N and Heindorf, Stefan and Demir, Caglar and Ngonga Ngomo, Axel-Cyrille},
  booktitle={European Semantic Web Conference},
  pages={236--252},
  year={2022},
  organization={Springer}
}
```

For any further questions, please contact:  ```onto-learn@lists.uni-paderborn.de```
