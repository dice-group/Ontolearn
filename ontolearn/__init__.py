"""
Structured Machine learning modules for Python
==================================


    name='ontolearn',
    description='Ontolearn is an open-source software library for structured machine learning in Python.

    Ontolearn '
                'includes modules for processing knowledge bases, inductive logic programming and ontology. '
                'engineering.',

Ontolearn is an open-source software library for structured machine learning in Python
The goal of ontolearn os to provide efficent solutions for concept learning on RDF knowledge bases
"""
__version__ = '0.0.1'


import warnings
warnings.filterwarnings("ignore")

from .base import KnowledgeBase
from .refinement_operators import Refinement
from .data_struct import Data
from .concept import Concept
from .ilp import SampleConceptLearner