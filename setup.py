import os

from setuptools import setup, find_packages

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name='OntoPy',
    version='0.0.1',
    packages=find_packages(exclude=('tests', 'test.*','examples.*')),
    description='OntoPy',
    install_requires=['scikit-learn==0.22.1',
                      'owlready2==0.23',
                      'torch'],
    author='Caglar Demir',
    author_email='caglardemir8@gmail.com',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License", ],
    python_requires='>=3.8',
    long_description=read('README.md'),
    long_description_content_type="test/markdown",
)
