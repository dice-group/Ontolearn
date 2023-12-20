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
