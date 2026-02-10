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
    - CELOE: A refinement-operator based learner (originating from DL-Learner).
      It performs heuristic-guided search over class expression refinements to
      find compact OWL class expressions that fit positive/negative examples.
      Suitable when symbolic search with ontological reasoning is required.
    - OCEL: A lightweight / constrained variant of CELOE. It uses a smaller set
      of refinements or simplified search heuristics to trade expressivity for
      speed and lower computational cost.

    SAT-Based Learners:
    - ALCSAT: A SAT-based learner that encodes the ALC concept learning problem
      as a SAT problem. It uses incremental SAT solving to find concepts of
      increasing size that maximize accuracy on positive/negative examples.
      Particularly effective for finding compact, exact solutions.
    - SPELL: A SAT-based learner using the general SPELL fitting framework.
      Supports different search modes (exact, neg_approx, full_approx) and can
      find separating queries of bounded size using SAT encoding.

    Neural / Hybrid Learners:
    - Drill: A neuro-symbolic learner that combines neural scoring or guidance
      with symbolic refinement/search. Typically, uses learned models to rank
      candidates while keeping final outputs in an interpretable DL form.
    - CLIP: A hybrid approach that leverages pretrained embeddings to assist
      candidate generation or scoring (e.g., using semantic similarity signals).
      Useful when distributional signals complement logical reasoning.
    - NCES, NCES2: Neural concept-expression search variants. These rely on
      neural encoders or learned scorers to propose and rank candidate
      class expressions; NCES2 represents an improved/iterated version.
    - NERO: A neural embedding model that learns permutation-invariant
      embeddings for sets of examples tailored towards predicting F1
      scores of pre-selected description logic concepts.
    - ROCES: A hybrid/refinement-based approach that combines ranking,
      coverage estimation, and refinement operators to discover candidate
      expressions efficiently. Extension of NCES2.

    -Evolutionary:
    - EvoLearner: Evolutionary search-based learner that evolves candidate
      descriptions (e.g., via genetic operators) using fitness functions
      derived from coverage and other objectives.

    Query-Based Learners:
    - SPARQLQueryLearner: Learns query patterns expressed as SPARQL queries
      that capture the target concept. Useful when working directly with
      SPARQL endpoints or large RDF datasets where query-based retrieval is
      preferable to reasoning-heavy symbolic search.

    Tree / Rule-Based Learners:
    - TDL: Tree-based Description Logic Learner. Adapts decision-tree style
      induction to construct DL class expressions from attribute-like splits
      or tests, producing interpretable, rule-like descriptions.

Example:
    >>> from ontolearn.learners import CELOE, Drill
    >>> from ontolearn.knowledge_base import KnowledgeBase
    >>>
    >>> kb = KnowledgeBase(path="example.owl")
    >>> model = CELOE(knowledge_base=kb)
    >>> model.fit(pos_examples, neg_examples)
"""

from .base import BaseConceptLearner, RefinementBasedConceptLearner
from .alcsat import ALCSAT
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
from .spell import SPELL
from .tree_learner import TDL

__all__ = [
    'BaseConceptLearner',
    'RefinementBasedConceptLearner',
    'ALCSAT',
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
    'SPELL',
    'TDL',
]