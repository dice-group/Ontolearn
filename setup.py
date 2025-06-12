# -----------------------------------------------------------------------------
# MIT License
#
# Copyright (c) 2024 Ontolearn Team
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# -----------------------------------------------------------------------------

"""
# Min version                  : pip3 install -e .
# Full version (to be reduced) : pip3 install -e .["full"]
# Document version             : pip3 install -e .["doc"]
"""

from setuptools import setup, find_packages
import re

with open('README.md', 'r') as fh:
    long_description = fh.read()
_deps = [
    "matplotlib>=3.3.4",
    "scikit-learn>=1.4.1",
    "torch==2.2.0",
    "rdflib>=6.0.2",
    "ruff>=0.7.2",
    "pandas>=1.5.0",
    "sortedcontainers>=2.4.0",
    "deap>=1.3.1",
    "flask>=1.1.2",
    "httpx>=0.25.2",
    "tqdm>=4.64.0",
    "transformers>=4.38.1",
    "pytest>=7.2.2",
    "owlapy==1.5.1",
    "dicee==0.1.4",
    "ontosample>=0.2.2",
    "sphinx>=7.2.6",
    "sphinx-autoapi>=3.0.0",
    "sphinx_rtd_theme>=2.0.0",
    "sphinx-theme>=1.0",
    "sphinxcontrib-plantuml>=0.27",
    "plantuml-local-client>=1.2022.6",
    "myst-parser>=2.0.0",
    "flake8>=6.0.0",
    "fastapi>=0.110.1",
    "uvicorn>=0.29.0",
    "openai>=1.86.0"]

deps = {b: a for a, b in (re.findall(r"^(([^!=<>~ ]+)(?:[!=<>~ ].*)?$)", x)[0] for x in _deps)}


def deps_list(*pkgs):
    return [deps[pkg] for pkg in pkgs]


extras = dict()
extras["min"] = deps_list(
    "matplotlib",
    "torch",
    "rdflib",
    "pandas",
    "sortedcontainers",
    "owlapy",
    "flask",  # Drill, NCES
    "tqdm", "transformers",  # NCES
    "dicee",  # Drill
    "deap",  # Evolearner
    "fastapi",
    "uvicorn",
    "openai")

extras["doc"] = (deps_list("sphinx",
                           "sphinx-autoapi",
                           "sphinx-theme",
                           "sphinx_rtd_theme",
                           "sphinxcontrib-plantuml",
                           "plantuml-local-client", "myst-parser"))

extras["full"] = (extras["min"] + deps_list("httpx", "pytest", "ontosample", "ruff"))

setup(
    name="ontolearn",
    description="Ontolearn is an open-source software library for structured machine learning in Python. Ontolearn includes modules for processing knowledge bases, inductive logic programming and ontology engineering.",
    version="0.9.2",
    packages=find_packages(),
    install_requires=extras["min"],
    extras_require=extras,
    author='Caglar Demir',
    author_email='caglardemir8@gmail.com',
    url='https://github.com/dice-group/Ontolearn',
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"],
    python_requires='>=3.10.13',
    entry_points={"console_scripts": ["ontolearn-webservice=ontolearn.scripts.run:main"]},
    long_description=long_description,
    long_description_content_type="text/markdown",
)
