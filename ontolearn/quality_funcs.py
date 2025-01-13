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
from ontolearn.search import EvaluatedConcept


def f1(*, individuals: Set, pos: Set, neg: Set):
    assert isinstance(individuals, set)
    assert isinstance(pos, set)
    assert isinstance(neg, set)

    tp = len(pos.intersection(individuals))
    tn = len(neg.difference(individuals))

    fp = len(neg.intersection(individuals))
    fn = len(pos.difference(individuals))

    try:
        recall = tp / (tp + fn)
    except ZeroDivisionError:
        return 0

    try:
        precision = tp / (tp + fp)
    except ZeroDivisionError:
        return 0

    if precision == 0 or recall == 0:
        return 0

    f_1 = 2 * ((precision * recall) / (precision + recall))
    return f_1


def acc(*, individuals: Set, pos: Set, neg: Set):
    assert isinstance(individuals, set)
    assert isinstance(pos, set)
    assert isinstance(neg, set)

    tp = len(pos.intersection(individuals))
    tn = len(neg.difference(individuals))

    fp = len(neg.intersection(individuals))
    fn = len(pos.difference(individuals))
    return (tp + tn) / (tp + tn + fp + fn)


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
