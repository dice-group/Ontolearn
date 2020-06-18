"""
Structured Machine learning modules for Python
==================================
Ontolearn is an open-source software library for structured machine learning in Python
The goal of ontolearn os to provide efficient solutions for concept learning on RDF knowledge bases
"""
__version__ = '0.0.2'


import warnings
warnings.filterwarnings("ignore")

from .base import KnowledgeBase
from .refinement_operators import Refinement
from .data_struct import Data
from .concept import Concept
from .concept_learner import SampleConceptLearner