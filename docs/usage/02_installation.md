# Installation

Since Ontolearn is a Python library, you will need to have Python installed on
your system (currently supporting version 3.10.13 or higher). Since python comes in various 
versions and with different, sometimes conflicting dependencies, most guides will 
recommend to set up a "virtual environment" to work in and so do we.

To create a virtual python environments you can consider using the builtin python module
[venv](https://docs.python.org/3/library/venv.html) or [conda](https://conda.io/projects/conda/en/latest/index.html). 

## Installation via _pip_

Released versions of Ontolearn can be installed using `pip`, the
Package Installer for Python. `pip` comes as part of Python.

```shell
pip install ontolearn
```

This will download and install the latest release version of Ontolearn
and all its dependencies from <https://pypi.org/project/ontolearn/>.

## Installation From Source

To download the Ontolearn source code, you will also need to have a
copy of the [Git](https://git-scm.com/) version control system conda installed.

Install java and curl:
```shell
# for Unix systems (Linux and macOS)
sudo apt install openjdk-11-jdk
sudo apt install curl
# for Windows please check online for yourself :)
```

Once you have the done previous step, you can continue setting up a virtual
environment and installing the dependencies.

* -> First download (clone) the source code
  ```shell
  git clone https://github.com/dice-group/Ontolearn.git
  cd Ontolearn
  ```
  
* -> Create and activate a conda virtual environment.
  ```shell
  conda create -n venv python=3.10.14 --no-default-packages
  conda activate venv
  ```
* -> Install the dependencies
  ```shell
  pip install -e .
  ```
  
Now you are ready to develop on Ontolearn or use the library!

### Verify installation

To test if the installation was successful, you can try this command:
It will only try to load the main library file into Python:

```shell
python -c "import ontolearn"
```

### Tests

You can run the tests as follows but make sure you have installed 
the external files using the commands described [here](#download-external-files-link-files)
to successfully pass all the tests:
```shell
pytest
```
Note: The tests are designed to run successfully on Linux machines since we also use them in 
GitHub Action. Therefore, trying to run them on a Windows machine can lead to some issues.

## Download External Files

Some resources like pre-calculated embeddings or `pre_trained_agents` and datasets (ontologies)
are not included in the repository directly. Use the command `wget` to download them from our data server.

> **NOTE: Before you run the following commands in your terminal, make sure you are 
in the root directory of the project!**

To download the datasets:

```shell
wget https://files.dice-research.org/projects/Ontolearn/KGs.zip -O ./KGs.zip
```

Then depending on your operating system, use the appropriate command to unzip the files:

```shell
# Windows
tar -xf KGs.zip

# or

# macOS and Linux
unzip KGs.zip
```

Finally, remove the _.zip_ file:

```shell
rm KGs.zip
```

To download learning problems:

```shell
wget https://files.dice-research.org/projects/Ontolearn/LPs.zip
```

Follow the same steps to unzip as the in the KGs case.

--------------------------------------------------------

### Other Data
Below you will find the links to get the necesseray data for _NCES_, _NCES2_, _ROCES_ and _CLIP_.
The process to extract the data is the same as shown earlier with "KGs".

```

#NCES:
https://files.dice-research.org/projects/NCES/NCES_Ontolearn_Data/NCESData.zip

#NCES2:
https://files.dice-research.org/projects/NCES/NCES_Ontolearn_Data/NCES2Data.zip

#ROCES:
https://files.dice-research.org/projects/NCES/NCES_Ontolearn_Data/ROCESData.zip

#CLIP:
https://files.dice-research.org/projects/Ontolearn/CLIP/CLIPData.zip
```

## Building (sdist and bdist_wheel)

In order to create a *distribution* of the Ontolearn source code, typically when creating a new release, 
it is necessary to use the `build` tool. It can be invoked with:

```shell
python -m build

# or

python setup.py bdist_wheel sdist
```

Distribution packages that are created, can then
be published to the [Python Package Index (PyPI)](https://pypi.org/) using [twine](https://pypi.org/project/twine/).

```shell
py -m twine upload --repository pypi dist/*
```


### Building the docs

The documentation can be built with

```shell
sphinx-build -M html docs/ docs/_build/
```

It is also possible to create a PDF manual, but that requires LaTeX to
be installed:

```shell
sphinx-build -M latex docs/ docs/_build/
```

## Simple Linting

You can lint check using [flake8](https://flake8.pycqa.org/):
```shell
flake8
```

or ruff:
```shell
ruff check
```

Additionally, you can specify the path where you want to execute the linter.


----------------------------------------------------------------------

In the next guide we give some example on the main usage of Ontolearn. The
guides after that, goes into more details on the key concepts and functionalities 
used in Ontolearn.
