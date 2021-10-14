# Ontolearn

Ontolearn is an open-source software library for explainable structured machine learning in Python.

- [Installation](#installation)

# Installation

### Installation from source

```shell
git clone https://github.com/dice-group/Ontolearn.git
cd Ontolearn
conda create --name temp python=3.8
conda activate temp
conda env update --name temp
python -c 'from setuptools import setup; setup()' develop
python -c "import ontolearn"
tox  # to test
```

### Installation via pip

```shell
pip install ontolearn  # see https://pypi.org/project/ontolearn/
```

### Download external files (.link files)

Some resources like pre-calculated embeddings or `pre_trained_agents`
are not included in the Git repository directly. Use the following
command to download them from our data server.

```shell
./big_gitext/download_big.sh pre_trained_agents.zip.link
./big_gitext/download_big.sh -A  # to download them all
```

To update or upload resource files, follow the instructions
[here](https://github.com/dice-group/Ontolearn-internal/wiki/Upload-big-data-to-hobbitdata)
and use the following command.

```shell
./big_gitext/upload_big.sh pre_trained_agents.zip
```

### Building (sdist and bdist_wheel)

```shell
tox -e build
```

#### Building the docs

```shell
tox -e docs
```

## Usage
See the [manual](https://ontolearn-docs-dice-group.netlify.app/),
tests and examples folder for details.

```python
from ontolearn.concept_learner import CELOE
from ontolearn.model_adapter import ModelAdapter
from owlapy.model import OWLNamedIndividual, IRI
from owlapy.namespaces import Namespaces
from owlapy.render import DLSyntaxObjectRenderer
from experiments_standard import ClosedWorld_ReasonerFactory

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


## Contribution
Feel free to create a pull request

### Simple Linting

Run
```shell
tox -e lint --
```

This will run [flake8](https://flake8.pycqa.org/) on the source code.

For any further questions, please contact:  ```onto-learn@lists.uni-paderborn.de```
