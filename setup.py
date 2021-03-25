from setuptools import setup, find_packages

with open('README.md', 'r') as fh:
    long_description = fh.read()

setup(
    name='ontolearn',
    description='Ontolearn is an open-source software library for structured machine learning in Python. Ontolearn '
                'includes modules for processing knowledge bases, inductive logic programming and ontology. '
                'engineering.',
    version='0.1.4',
    packages=find_packages(exclude=('tests', 'test.*', 'examples.*')),
    install_requires=['scikit-learn==0.24.1',
                      'matplotlib==3.3.4',
                      'pytest==6.2.2',
                      'owlready2==0.26',
                      'torch==1.7.1',
                      'rdflib==5.0.0',
                      'pandas==1.2.2',
                      'flask==1.1.2',
                      ],
    author='Caglar Demir',
    author_email='caglardemir8@gmail.com',
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License", ],
    python_requires='==3.7.1',
    scripts=['ontolearn/endpoint/simple_drill_endpoint'],
    long_description=long_description,
    long_description_content_type="text/markdown",
)
