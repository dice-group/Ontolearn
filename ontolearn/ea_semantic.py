"""Semantic-aware parent selection for EvoLearner.

EvoLearner's default loop scores an individual by one scalar (F1 / accuracy minus a length penalty). Here the
per-example outcome of each individual, its *error vector* (``ind.errors[i]`` is True iff example ``i`` is
misclassified, available at no extra reasoner cost), is used for lexicase selection.
"""

import random
from typing import List

import numpy as np

from ontolearn.ea_utils import Tree


def lexicase_select(population: List[Tree], k: int) -> List[Tree]:
    """Lexicase selection on binary example errors.

    For each selection the examples are visited in random order and the candidate pool is reduced to the
    individuals that are correct on the current example (if any is). The individual left once the pool is a
    singleton or the examples are exhausted is selected; among several survivors the shortest tree wins (random
    tie-break), which doubles as parsimony pressure.

    Args:
        population: Evaluated individuals, each with a boolean numpy attribute ``errors``.
        k: Number of individuals to select.

    Returns:
        The selected individuals (references, not copies, like :func:`deap.tools.selTournament`).
    """
    errors = np.stack([ind.errors for ind in population])  # (pop, examples), True = misclassified
    lengths = np.fromiter((len(ind) for ind in population), dtype=np.int64, count=len(population))
    n_pop, n_cases = errors.shape
    selected = []
    for _ in range(k):
        pool = np.arange(n_pop)
        for case in random.sample(range(n_cases), n_cases):
            col = errors[pool, case]
            if col.all():
                continue  # nobody is right on this example: it cannot discriminate
            pool = pool[~col]
            if len(pool) == 1:
                break
        if len(pool) > 1:
            shortest = pool[lengths[pool] == lengths[pool].min()]
            chosen = int(random.choice(list(shortest)))
        else:
            chosen = int(pool[0])
        selected.append(population[chosen])
    return selected
