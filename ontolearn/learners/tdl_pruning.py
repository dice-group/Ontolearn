"""Reduced-error pruning of the class expression that :class:`ontolearn.learners.tree_learner.TDL` reads off its tree.

.. warning::
   **This is not part of the published TDL algorithm.** TDL (Tree-based Description Logic learner, the peer-reviewed
   method) fits a decision tree on binary class-expression features and reads the hypothesis off the tree. The
   pruning here was added afterwards, is *not* from a peer-reviewed publication, and has only been evaluated in the
   Ontolearn repository (issue #672). It is opt-in (``TDL(hypothesis_extraction="pruned_leaves")``); the default
   behaviour of TDL is unchanged. Do not cite it as part of TDL.

A *rule* is a tuple of ``(column, holds)`` literals: the conjunction of the column expressions that must hold
(``holds=True``) or must not hold (``holds=False``, the complement). The hypothesis is the union of the rules. Everything
works on plain numpy arrays (no reasoner).
"""
from typing import List, Sequence, Tuple

import numpy as np

Rule = Tuple[Tuple[int, bool], ...]


def rule_mask(X: np.ndarray, rule: Rule) -> np.ndarray:
    """Boolean vector: which rows of `X` satisfy the conjunction `rule` (an empty rule covers every row)."""
    mask = np.ones(X.shape[0], dtype=bool)
    for column, holds in rule:
        mask &= (X[:, column] > 0.5) == holds
    return mask


def prune_rules(rules: Sequence[Rule], X: np.ndarray, y: np.ndarray, tolerance: float = 0.0) -> List[Rule]:
    """Derefine a union of rules on the rows `(X, y)` (reduced-error pruning of a class expression).

    Two kinds of edit make the hypothesis simpler and more general: dropping one literal from a conjunction (a
    conjunction keeps at least one literal) and dropping a whole conjunction (at least one stays). Each round applies
    the edit with the best F1 on `(X, y)` (the first of equally good edits, so the result is deterministic). It is
    accepted if the F1 stays within `tolerance` of the F1 of the *unpruned* rules, so the F1 cannot drift down over
    many rounds. With ``tolerance=0`` ties are resolved in favour of the simpler hypothesis, which removes
    redundancy without lowering the F1 on `(X, y)`. Pruning stops when no edit is acceptable.

    If `y` has no positive row there is nothing to measure and the rules are returned unchanged.
    """
    rules = [tuple(r) for r in rules]
    if not rules or int((y == 1).sum()) == 0:
        return rules
    positive = y == 1
    n_positive = int(positive.sum())

    def f1_of(covered):
        tp = int((covered & positive).sum())
        return 0.0 if tp == 0 else 2 * tp / (int(covered.sum()) + n_positive)

    masks = [rule_mask(X, r) for r in rules]
    count = np.sum(masks, axis=0)  # how many rules cover each row: the union without rule i is count - masks[i] > 0
    floor = f1_of(count > 0) - tolerance - 1e-12
    while True:
        best_score, best = -1.0, None
        for i, rule in enumerate(rules):
            rest = count - masks[i]
            if len(rules) > 1:  # drop the whole conjunction
                score = f1_of(rest > 0)
                if score > best_score + 1e-12:
                    best_score, best = score, (i, None, None)
            if len(rule) > 1:  # drop one literal
                for j in range(len(rule)):
                    shorter = rule[:j] + rule[j + 1:]
                    mask = rule_mask(X, shorter)
                    score = f1_of((rest + mask) > 0)
                    if score > best_score + 1e-12:
                        best_score, best = score, (i, shorter, mask)
        if best is None or best_score < floor:
            return rules
        i, shorter, mask = best
        if shorter is None:
            count = count - masks[i]
            del rules[i], masks[i]
        else:
            count = count - masks[i] + mask
            rules[i], masks[i] = shorter, mask
