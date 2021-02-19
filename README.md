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
python -m pytest tests #
```
### Installation via pip

```
pip install ontolearn # https://pypi.org/project/ontolearn/ only a place holder.
```

## Usage
See examples folder.

## Contribution
Feel free to create a pull request. We will promptly review pull requests.

### Simple Linting and Testing

Run
```shell script
flake8
pylint
```

### Integration Testing with Docker

For testing we use [docker](https://docs.docker.com/engine/install/). 

We have a docker image [`CI/Dockerfile`](./CI/Dockerfile) which builds the package and runs all tests. 

Build it with:
```shell script
docker build -f CI/Dockerfile . --tag ontopy-test
```

Run it with:
```shell script
docker run ontopy-test
```

For any further questions, please contact:  ```caglar.demir@upb.de```