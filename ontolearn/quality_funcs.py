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

from typing import Set
from owlapy.class_expression import OWLClassExpression
from ontolearn.abstracts import EncodedLearningProblem, AbstractScorer, AbstractKnowledgeBase
from ontolearn.metrics import F1 as _F1, Accuracy as _Accuracy
from ontolearn.search import EvaluatedConcept


def _tp_tn_fp_fn(individuals: Set, pos: Set, neg: Set):
    assert isinstance(individuals, set)
    assert isinstance(pos, set)
    assert isinstance(neg, set)

    tp = len(pos.intersection(individuals))
    tn = len(neg.difference(individuals))
    fp = len(neg.intersection(individuals))
    fn = len(pos.difference(individuals))
    return tp, tn, fp, fn


def f1(*, individuals: Set, pos: Set, neg: Set):
    tp, tn, fp, fn = _tp_tn_fp_fn(individuals, pos, neg)
    applicable, score = _F1().score2(tp=tp, fn=fn, fp=fp, tn=tn)
    return score if applicable else 0


def acc(*, individuals: Set, pos: Set, neg: Set):
    tp, tn, fp, fn = _tp_tn_fp_fn(individuals, pos, neg)
    _, score = _Accuracy().score2(tp=tp, fn=fn, fp=fp, tn=tn)
    return score


def evaluate_concept(kb: AbstractKnowledgeBase, concept: OWLClassExpression, quality_func: AbstractScorer,
                     encoded_learning_problem: EncodedLearningProblem) -> EvaluatedConcept:
    """Evaluates a concept by using the encoded learning problem examples, in terms of Accuracy or F1-score.

    Note:
        This method is useful to tell the quality (e.q) of a generated concept by the concept learners, to get
        the set of individuals (e.inds) that are classified by this concept and the amount of them (e.ic).
    Args:
        kb: The knowledge base where to evaluate the concept.
        concept: The concept to be evaluated.
        quality_func: Quality measurement in terms of Accuracy or F1-score.
        encoded_learning_problem: The encoded learning problem.
    Return:
        The evaluated concept.
    """

    e = EvaluatedConcept()
    e.inds = kb.individuals_set(concept)
    e.ic = len(e.inds)
    _, e.q = quality_func.score_elp(e.inds, encoded_learning_problem)
    return e
