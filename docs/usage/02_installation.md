# Installation

Since Ontolearn is a Python library, you will need to have Python on
your system. Python comes in various versions and with different,
sometimes conflicting dependencies. Hence, most guides will recommend
to set up a "virtual environment" to work in.

One such system for virtual python environments is 
[conda](https://conda.io/projects/conda/en/latest/index.html).

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
Note: Since Unix and Windows reference files differently, the test are set to work on Linux 
but in Widows the filepaths throughout test cases should be changed which is something that
is not very convenient to do. If you really want to run the tests in Windows, you can
make use of the replace all functionality to change them.

## Download External Files

Some resources like pre-calculated embeddings or `pre_trained_agents` and datasets (ontologies)
are not included in the repository directly. Use the command line command `wget`
to download them from our data server.

> **NOTE: Before you run this commands in your terminal, make sure you are 
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

--------------------------------------------------------

### NCES data: 

```shell
wget https://files.dice-research.org/projects/NCES/NCES_Ontolearn_Data/NCESData.zip -O ./NCESData.zip
unzip NCESData.zip
rm NCESData.zip
```

If you are getting any error check if the following flags can help:

```shell
unzip -o NCESData.zip
rm -f NCESData.zip
```

-------------------------------------------------------

### CLIP data:

```commandline
wget https://files.dice-research.org/projects/Ontolearn/CLIP/CLIPData.zip
unzip CLIPData.zip
rm CLIPData.zip 
```

## Building (sdist and bdist_wheel)

In order to create a *distribution* of the Ontolearn source code, typically when creating a new release, 
it is necessary to use the `build` tool. It can be invoked with:

```shell
python -m build
```

from the main source code folder. Packages created by `build` can then
be uploaded as releases to the [Python Package Index (PyPI)](https://pypi.org/) using
[twine](https://pypi.org/project/twine/).


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

Using the following command will run the linting tool [flake8](https://flake8.pycqa.org/) on the source code.
```shell
flake8
```

Additionally, you can specify the path where you want to flake8 to run.


----------------------------------------------------------------------

In the next guide we give some example on the main usage of Ontolearn. The
guides after that, goes into more details on the key concepts and functionalities 
used in Ontolearn.
