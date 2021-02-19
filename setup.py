from setuptools import setup, find_packages

with open('README.md', 'r') as fh:
    long_description = fh.read()

setup(
    name='ontolearn',
    description='Ontolearn is an open-source software library for structured machine learning in Python. Ontolearn '
                'includes modules for processing knowledge bases, inductive logic programming and ontology. '
                'engineering.',
    version='0.0.2',
    packages=find_packages(exclude=('tests', 'test.*', 'examples.*')),
    install_requires=['scikit-learn>=0.22.1',
                      'matplotlib',
                      'owlready2>=0.23',
                      'torch',
                      'rdflib',
                      'pandas',
                      'pytest'
                      ],
    author='Caglar Demir',
    author_email='caglardemir8@gmail.com',
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License", ],
    python_requires='==3.7.1',
    long_description=long_description,
    long_description_content_type="text/markdown",
)
