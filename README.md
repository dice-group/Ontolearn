# Ontolearn

Ontolearn is an open-source software library for explainable structured machine learning in Python.

- [Installation](#installation)

# Installation
### Installation from source
```
git clone https://github.com/dice-group/OntoPy.git
conda create -n temp python=3.7.1
conda activate temp
pip install -e .
python -c "import ontolearn"
python -m pytest tests
```
### Installation via pip

```
pip install ontolearn # https://pypi.org/project/ontolearn/ only a place holder.
```

## Usage
See examples folder.

## Contribution
Feel free to create a pull request

### Simple Linting and Testing

Run
```shell script
flake8
pylint
```

For any further questions, please contact:  ```caglar.demir@upb.de```