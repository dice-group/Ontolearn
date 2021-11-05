# Installation

Since Ontolearn is a Python library, you will need to have Python on
your system. Python comes in various versions and with different,
sometimes conflicting dependencies. Hence, most guides will recommend
to set up a "virtual environment" to work in.

One such system for virtual python environments is
[Anaconda](https://www.anaconda.com/). You can download miniconda from
<https://docs.conda.io/en/latest/miniconda.html>.

We have good experience with it and make use of conda in the
[Installation from source](#installation-from-source) step.

## Installation from source

To download the Ontolearn source code, you will also need to have a
copy of the [Git](https://git-scm.com/) version control system.

Once you have `conda` and `git` installed, the following commands
should be typed in your shell in order to download the Ontolearn
development sources, install the dependencies listened in the
`environment.yml` into a conda environment and create the necessary
installation links to get started with the library.

* Download (clone) the source code
  ```shell
  git clone https://github.com/dice-group/Ontolearn.git
  cd Ontolearn
  ```
* Load the dependencies into a new conda environment called "temp" (you can name it however you like)
  ```shell
  conda create --name temp python=3.8
  conda activate temp
  conda env update --name temp
  ```
* Install the development links so that Python will find the library
  ```shell
  python -c 'from setuptools import setup; setup()' develop 
  ```
* Instead of the previous step there is also Possibility B, which is valid temporarily only in your current shell:
  ```shell
  export PYTHONPATH=$PWD
  ```

Now you are ready to develop on Ontolearn or use the library!

### Verify installation

To test if the installation was successful, you can try this command:
It will only try to load the main library file into Python:

```shell
python -c "import ontolearn"
```

### Tests

In order to run our test suite, type:

```shell
tox
```

You can also run the tests directly using:

```shell
pytest
```


## Installation via pip

Released versions of Ontolearn can also be installed using `pip`, the
Package Installer for Python. It comes as part of Python. Please
research externally (or use above `conda create` command) on how to
create virtual environments for Python programs.

```shell
pip install ontolearn
```

This will download and install the latest release version of Ontolearn
and all its dependencies from <https://pypi.org/project/ontolearn/>.

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

## Building (sdist and bdist_wheel)

In order to create a *distribution* of the Ontolearn source code, typically when creating a new release, it is necessary to use the `build` tool. It can be invoked with:

```shell
tox -e build
```

from the main source code folder. Packages created by `build` can then
be uploaded as releases to the [Python Package Index (PyPI)](https://pypi.org/) using
[twine](https://pypi.org/project/twine/).


### Building the docs

The documentation can be built with

```shell
tox -e docs
```

It is also possible to create a PDF manual, but that requires LaTeX to
be installed:

```shell
tox -e docs latexpdf
```

## Contribution

Feel free to create a pull request.

## Questions

For any further questions, please contact: ```onto-learn@lists.uni-paderborn.de```
or open an issue on our [GitHub issues
page](https://github.com/dice-group/Ontolearn/issues).
