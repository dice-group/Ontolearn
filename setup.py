from setuptools import setup, find_packages
import re

with open('README.md', 'r') as fh:
    long_description = fh.read()

_deps = [
        "matplotlib>=3.3.4",
        "owlready2>=0.40",
        "torch>=1.7.1",
        "rdflib>=6.0.2",
        "pandas>=1.5.0",
        "sortedcontainers>=2.4.0",
        "flask>=1.1.2",
        "deap>=1.3.1",
        "httpx>=0.25.2",
        "tqdm>=4.64.0",
        "transformers>=4.38.1",
        "pytest>=7.2.2",
        "owlapy>=0.1.1",
        "dicee>=0.1.2",
        "ontosample>=0.2.2",
        "gradio>=4.11.0"]

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
    "owlready2",
    "owlapy",
    "flask",  # Drill, NCES
    "tqdm", "transformers",  # NCES
    "dicee",  # Drill
    "deap",  # Evolearner
)

extras["full"] = (extras["min"] + deps_list("httpx", "pytest", "gradio", "ontosample"))

setup(
    name="ontolearn",
    description="Ontolearn is an open-source software library for structured machine learning in Python. Ontolearn includes modules for processing knowledge bases, inductive logic programming and ontology engineering.",
    version="0.7.0",
    packages=find_packages(),
    install_requires=extras["min"],
    extras_require=extras,
    author='Caglar Demir',
    author_email='caglardemir8@gmail.com',
    url='https://github.com/dice-group/Ontolearn',
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"],
    python_requires='>=3.9.18',
    entry_points={"console_scripts": ["ontolearn = ontolearn.run:main"]},
    long_description=long_description,
    long_description_content_type="text/markdown",
)
