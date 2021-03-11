# Ontolearn

Ontolearn is an open-source software library for explainable structured machine learning in Python.

- [Installation](#installation)

# Installation

### Installation from source

```
git clone https://github.com/dice-group/OntoPy.git
cd OntoPy
conda create --name temp python=3.8
conda activate temp
conda env update --name temp
python -c 'from setuptools import setup; setup()' develop  # OR
# export PYTHONPATH=$PWD
python -c "import ontolearn"
conda install pytest
pytest
```

### Installation via pip

```
pip install ontolearn  # https://pypi.org/project/ontolearn/ only a place holder.
```

### Building (sdist and bdist_wheel)

```
pip install build
python -m build
```

## Usage
See tests and examples folder.

## Contribution
Feel free to create a pull request

### Simple Linting

Run
```shell script
flake8
```

For any further questions, please contact:  ```caglar.demir@upb.de```
