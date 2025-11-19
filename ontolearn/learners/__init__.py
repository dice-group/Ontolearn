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
Concept Learning Algorithms Module
===================================

This module provides various concept learning algorithms for ontology engineering and OWL class expression learning.

Available Learners:
    
    Refinement-Based Learners:
    - CELOE: Class Expression Learning for Ontology Engineering
    - OCEL: A limited version of CELOE
    
    Neural/Hybrid Learners:
    - Drill: Neuro-Symbolic Class Expression Learning
    - TDL: Tree-based Description Logic Learner
    
    Query-Based Learners:
    - SPARQLQueryLearner: Learning SPARQL queries from DL concepts
    
    Experimental:
    - NERO: Neural Evolutionary Reinforcement Ontology learner (experimental)

Example:
    >>> from ontolearn.learners import CELOE, Drill
    >>> from ontolearn.knowledge_base import KnowledgeBase
    >>> 
    >>> kb = KnowledgeBase(path="example.owl")
    >>> model = CELOE(knowledge_base=kb)
    >>> model.fit(pos_examples, neg_examples)
"""

from .base import BaseConceptLearner, RefinementBasedConceptLearner
from .celoe import CELOE
from .clip import CLIP
from .drill import Drill
from .evolearner import EvoLearner
from .nces import NCES
from .nces2 import NCES2
from .nero import NERO
from .ocel import OCEL
from .roces import ROCES
from .sparql_query_learner import SPARQLQueryLearner
from .tree_learner import TDL

__all__ = [
    'BaseConceptLearner',
    'RefinementBasedConceptLearner',
    'CELOE',
    'CLIP',
    'Drill',
    'EvoLearner',
    'NCES',
    'NCES2',
    'NERO',
    'OCEL',
    'ROCES',
    'SPARQLQueryLearner',
    'TDL',
]