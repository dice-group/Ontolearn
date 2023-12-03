from setuptools import setup, find_packages

with open('README.md', 'r') as fh:
    long_description = fh.read()
setup(
    name="ontolearn",
    description="Ontolearn is an open-source software library for structured machine learning in Python. Ontolearn includes modules for processing knowledge bases, inductive logic programming and ontology engineering.",
    version="0.6.1",
    packages=find_packages(),
    install_requires=[
        "scikit-learn>=0.24.1",
        "matplotlib>=3.3.4",
        "owlready2>=0.40",
        "torch>=1.7.1",
        "rdflib>=6.0.2",
        "pandas>=1.5.0",
        "sortedcontainers>=2.4.0",
        "flask>=1.1.2",
        "deap>=1.3.1",
        "httpx>=0.21.1",
        "parsimonious>=0.8.1",
        "tqdm>=4.64.0",
        "tokenizers>=0.12.1",
        "transformers>=4.19.2",
        "pytest>=7.2.2",
        "owlapy>=0.1.0"],
    author='Caglar Demir',
    author_email='caglardemir8@gmail.com',
    url='https://github.com/dice-group/Ontolearn',
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"],
    python_requires='>=3.8',
    entry_points={"console_scripts": ["ontolearn = ontolearn.run:main"]},
    long_description=long_description,
    long_description_content_type="text/markdown",
)
