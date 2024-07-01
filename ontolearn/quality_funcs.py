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
