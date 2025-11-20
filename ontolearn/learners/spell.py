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

"""SPELL Learner - SAT-based concept learning using SPELL fitting."""

from typing import Optional

from owlapy.class_expression import OWLClassExpression
from owlapy.abstracts import AbstractOWLReasoner

from ontolearn.abstracts import AbstractKnowledgeBase
from ontolearn.learning_problem import PosNegLPStandard
from ontolearn.learners.sat_base import SATBaseLearner

from ontolearn.learners.spell_kit import fitting
from ontolearn.learners.spell_kit.structures import Structure


class SPELL(SATBaseLearner):
    """
    SPELL: SAT-based concept learner using general SPELL fitting.

    This learner uses SAT solvers to find concept expressions that fit positive and negative examples.
    Unlike ALCSAT which is specialized for ALC, SPELL uses the more general fitting.py module
    which supports different modes of operation.

    The algorithm incrementally searches for queries of increasing size that maximize
    the coverage on the given examples.

    Attributes:
        kb (AbstractKnowledgeBase): The knowledge base that the concept learner is using.
        max_query_size (int): Maximum size of queries to search for.
        search_mode: Search mode - exact, neg_approx, or full_approx.
        _best_hypothesis (OWLClassExpression): Best found hypothesis.
        _best_hypothesis_accuracy (float): Accuracy of the best hypothesis.
        _structure (Structure): Internal structure representation of the knowledge base.
        _ind_to_owl (dict): Mapping from internal individual indices to OWL individuals.
        _owl_to_ind (dict): Mapping from OWL individuals to internal indices.
    """

    __slots__ = ('max_query_size', 'starting_query_size', 'search_mode')

    name = 'spell'

    def __init__(self,
                 knowledge_base: AbstractKnowledgeBase,
                 reasoner: Optional[AbstractOWLReasoner] = None,
                 max_runtime: Optional[int] = 60,
                 max_query_size: int = 19,
                 starting_query_size: int = 1,
                 search_mode: str = 'full_approx'):
        """
        Initialize SPELL learner.

        Args:
            knowledge_base: The knowledge base to use for learning.
            reasoner: Optional reasoner (if None, uses the KB's reasoner).
            max_runtime: Maximum allowed runtime in seconds.
            max_query_size: Maximum query size to search. Defaults to 19.
            starting_query_size: Starting query size for incremental search. Defaults to 1.
            search_mode: Search mode - 'exact', 'neg_approx', or 'full_approx'. Defaults to 'full_approx'.
                  - exact: Search for queries that cover all positive examples
                  - neg_approx: Allow approximation on negative examples
                  - full_approx: Allow approximation on both positive and negative examples
        """
        super().__init__(knowledge_base, reasoner, max_runtime)

        self.max_query_size = max_query_size
        self.starting_query_size = starting_query_size

        # Convert string mode to enum
        mode_map = {
            'exact': fitting.mode.exact,
            'neg_approx': fitting.mode.neg_approx,
            'full_approx': fitting.mode.full_approx
        }

        if search_mode not in mode_map:
            raise ValueError(f"Invalid mode: {search_mode}. Must be one of {list(mode_map.keys())}")

        self.search_mode = mode_map[search_mode]

    def fit(self, lp: PosNegLPStandard):
        """
        Find concept expressions that explain positive and negative examples.

        Args:
            lp: Learning problem with positive and negative examples.

        Returns:
            self
        """
        import time

        self.clean()
        self.start_time = time.time()

        # Construct learning problem
        assert isinstance(lp, PosNegLPStandard)
        self._learning_problem = lp

        pos = set(self._learning_problem.pos)
        neg = set(self._learning_problem.neg)

        # Convert positive and negative examples to indices
        P = [self._owl_to_ind[ind] for ind in pos]
        N = [self._owl_to_ind[ind] for ind in neg]

        # Run incremental search using SPELL fitting
        timeout = self.max_runtime if self.max_runtime else -1
        best_coverage, best_query = fitting.solve_incr(
            A=self._structure,
            P=P,
            N=N,
            m=self.search_mode,
            timeout=timeout,
            starting_size=self.starting_query_size,
            max_size=self.max_query_size
        )

        # Calculate accuracy
        total_examples = len(P) + len(N)
        best_acc = best_coverage / total_examples if total_examples > 0 else 0.0

        # Convert solution to OWL expression
        if best_query is not None and best_query.max_ind > 0:
            owl_expr = self._structure_to_owl_expression(best_query)
            self._best_hypothesis = owl_expr
            self._best_hypothesis_accuracy = best_acc

        return self
