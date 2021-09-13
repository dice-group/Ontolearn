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
pip install ontolearn  # https://pypi.org/project/ontolearn/ only a place holder.
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
See the [manual](https://ontolearn-docs-dice-group.netlify.app/), tests and examples folder.

## Contribution
Feel free to create a pull request

### Simple Linting

Run
```shell
flake8
```

For any further questions, please contact:  ```onto-learn@lists.uni-paderborn.de```
