"""Helpers for :class:`ontolearn.learners.tree_learner_forest.TDLForest`: bagged decision trees on the binary
class-expression feature matrix.

TDL turns every example into a row of binary features (column = one OWL class expression, 1 iff the example is an
instance of it) and fits one decision tree. These helpers fit *many* bagged trees on the same matrix and turn them
into something TDL can return. Everything here works on plain numpy arrays (no reasoner), so it is cheap to test.

A *rule* is a tuple of ``(column, holds)`` literals: the conjunction of the column expressions that must hold
(``holds=True``) or must not hold (``holds=False``, i.e. the complement). A tree's rules are its root-to-leaf paths
that end in a leaf predicting the positive class.
"""
import itertools
from typing import List, Sequence, Tuple

import numpy as np
from sklearn.tree import DecisionTreeClassifier

#: Ways to use a forest in ``TDLForest(mode=...)``; see :class:`ontolearn.learners.tree_learner_forest.TDLForest`.
FOREST_MODES = ("tree", "feature_selection", "rules", "vote", "forest_predictor", "best_tree", "pruned", "full_rules", "pruned_full")

Rule = Tuple[Tuple[int, bool], ...]


def fit_bagged_trees(X: np.ndarray, y: np.ndarray, n_trees: int, max_features, tree_kwargs: dict,
                     random_state: int = 0) -> List[Tuple[DecisionTreeClassifier, np.ndarray]]:
    """Fit `n_trees` decision trees on bootstrap samples of the rows.

    Args:
        X: Binary feature matrix (examples x features).
        y: Labels, 1 for positive and 0 for negative examples.
        n_trees: Number of trees.
        max_features: Features considered per split (``"sqrt"``, ``None``, an int or a float, as in sklearn).
        tree_kwargs: Extra keyword arguments for every :class:`DecisionTreeClassifier`.
        random_state: Seed; tree ``t`` draws its bootstrap sample and its split randomness from ``random_state + t``.

    Returns:
        A list of ``(tree, oob_rows)`` where ``oob_rows`` are the row indices that are not in the tree's bootstrap
        sample ("out-of-bag"), i.e. rows the tree never saw during training.
    """
    n = len(y)
    trees = []
    for t in range(n_trees):
        rng = np.random.RandomState(random_state + t)
        bag = rng.randint(0, n, size=n)
        # A bag must contain both classes, otherwise the tree is a constant; redraw (bounded) in that rare case.
        for _ in range(20):
            if 0 < y[bag].sum() < n:
                break
            bag = rng.randint(0, n, size=n)
        in_bag = np.zeros(n, dtype=bool)
        in_bag[bag] = True
        kwargs = dict(tree_kwargs)
        kwargs.setdefault("max_features", max_features)
        kwargs["random_state"] = random_state + t
        clf = DecisionTreeClassifier(**kwargs).fit(X[bag], y[bag])
        trees.append((clf, np.flatnonzero(~in_bag)))
    return trees


def top_features(trees: Sequence[Tuple[DecisionTreeClassifier, np.ndarray]], n_features: int, top_k: int) -> List[int]:
    """Indices of the `top_k` features with the highest mean impurity-based importance over the trees.

    Features that no tree uses are never returned. Ties are broken by column index, so the result is deterministic.
    The indices are returned in ascending order, which keeps the original column order of the feature matrix.
    """
    importance = np.zeros(n_features)
    for clf, _ in trees:
        importance += clf.feature_importances_
    order = sorted(range(n_features), key=lambda j: (-importance[j], j))
    return sorted(j for j in order[:top_k] if importance[j] > 0)


def positive_leaf_rules(clf: DecisionTreeClassifier) -> List[Rule]:
    """The root-to-leaf paths of `clf` that end in a leaf that predicts the positive class (depth first, left first)."""
    tree_ = clf.tree_
    rules: List[Rule] = []

    def visit(node: int, path: Tuple[Tuple[int, bool], ...]):
        if tree_.children_left[node] == -1:
            if clf.classes_[int(np.argmax(tree_.value[node, 0]))] == 1:
                rules.append(path)
            return
        feature = int(tree_.feature[node])
        visit(tree_.children_left[node], path + ((feature, False),))
        visit(tree_.children_right[node], path + ((feature, True),))

    visit(0, ())
    return rules


def rule_mask(X: np.ndarray, rule: Rule) -> np.ndarray:
    """Boolean vector: which rows of `X` satisfy the conjunction `rule` (an empty rule covers every row)."""
    mask = np.ones(X.shape[0], dtype=bool)
    for column, holds in rule:
        mask &= (X[:, column] > 0.5) == holds
    return mask


def candidate_rules(trees: Sequence[Tuple[DecisionTreeClassifier, np.ndarray]], X: np.ndarray, y: np.ndarray
                    ) -> List[Tuple[Rule, float]]:
    """All distinct positive-leaf rules of the trees with an honest precision estimate.

    The precision of a rule is measured on the out-of-bag rows of the tree it came from, with a Laplace correction
    ``(tp + 1) / (covered + 2)``: those rows did not shape the rule, so a rule that merely memorised its bootstrap
    sample does not look good. If a rule covers no out-of-bag row its precision on all rows is used instead.
    A rule found by several trees keeps its best estimate. Output order: first occurrence, deterministic.
    """
    best = {}
    for clf, oob in trees:
        for rule in positive_leaf_rules(clf):
            if len(rule) == 0:
                continue  # the all-covering rule says nothing; the greedy selection would always reject it
            covered = rule_mask(X[oob], rule)
            if covered.sum() > 0:
                precision = (y[oob][covered].sum() + 1) / (covered.sum() + 2)
            else:
                covered = rule_mask(X, rule)
                precision = (y[covered].sum() + 1) / (covered.sum() + 2)
            if rule not in best or precision > best[rule]:
                best[rule] = precision
    return list(best.items())


def select_rules(candidates: Sequence[Tuple[Rule, float]], X: np.ndarray, y: np.ndarray, min_precision: float
                 ) -> List[Rule]:
    """Greedy set cover over the positive examples.

    Only rules with an estimated precision of at least `min_precision` are eligible. Repeatedly add the rule that
    gains most: (newly covered positives) - (negatives it covers that no chosen rule covers yet). Stop when no rule
    has a positive gain. Ties go to the shorter rule, then to the earlier candidate (deterministic).
    """
    eligible = [(rule, rule_mask(X, rule)) for rule, precision in candidates if precision >= min_precision]
    positive, negative = y == 1, y == 0
    covered = np.zeros(len(y), dtype=bool)
    chosen: List[Rule] = []
    while True:
        best, best_key = None, None
        for index, (rule, mask) in enumerate(eligible):
            gain = int((mask & positive & ~covered).sum()) - int((mask & negative & ~covered).sum())
            key = (-gain, len(rule), index)
            if gain > 0 and (best is None or key < best_key):
                best, best_key = (rule, mask), key
        if best is None:
            return chosen
        chosen.append(best[0])
        covered |= best[1]


def majority_subsets(n_members: int) -> List[Tuple[int, ...]]:
    """All index subsets of size ``n_members // 2 + 1``: the majority of an odd number of voters is "any such subset
    agrees", i.e. a union over these subsets of the intersections of their members."""
    return list(itertools.combinations(range(n_members), n_members // 2 + 1))


def positive_region_predictor(clf: DecisionTreeClassifier, X_train: np.ndarray, y_train: np.ndarray,
                              extraction: str = "positive_paths"):
    """Row-level predictions that agree with the class expression TDL extracts from `clf`.

    A row is predicted positive iff it lands in a leaf that the extracted hypothesis keeps: with ``positive_paths``
    every leaf that holds at least one positive *training* row, otherwise (``predicted_leaves`` and
    ``factored_tree``, which make the same predictions) every leaf whose majority class is positive. This lets a
    hypothesis be scored on feature rows without a reasoner, e.g. to tune hyperparameters by cross-validation.

    Args:
        clf: The fitted tree.
        X_train: Feature rows that `clf` was trained on (only used by ``positive_paths``).
        y_train: Their labels (1 positive, 0 negative).
        extraction: The ``hypothesis_extraction`` mode.

    Returns:
        A function mapping a feature matrix to a 0/1 numpy vector.
    """
    if extraction == "positive_paths":
        leaves = np.unique(clf.apply(X_train[y_train == 1]))
    else:
        leaves = np.array([node for node in range(clf.tree_.node_count)
                           if clf.tree_.children_left[node] == -1
                           and clf.classes_[int(np.argmax(clf.tree_.value[node, 0]))] == 1])
    return lambda rows: np.isin(clf.apply(rows), leaves).astype(int)


def rules_f1(X: np.ndarray, y: np.ndarray, rules: Sequence[Rule]) -> float:
    """F1 of the union of `rules` (positive class) on the rows `X` with labels `y`; 0 if nothing is predicted."""
    covered = np.zeros(len(y), dtype=bool)
    for rule in rules:
        covered |= rule_mask(X, rule)
    tp = int((covered & (y == 1)).sum())
    return 0.0 if tp == 0 else 2 * tp / (int(covered.sum()) + int((y == 1).sum()))


def prune_rules(rules: Sequence[Rule], X: np.ndarray, y: np.ndarray, tolerance: float = 0.0) -> List[Rule]:
    """Derefine a union of rules on the rows `(X, y)` (reduced-error pruning of a class expression).

    The hypothesis is the union of the conjunctions `rules`. Two kinds of edit make it simpler and more general:
    dropping one literal from a conjunction (a conjunction keeps at least one literal) and dropping a whole
    conjunction (at least one stays). Each round applies the edit with the best F1 on `(X, y)` (the first of equally
    good edits, so the result is deterministic). It is accepted if the F1 stays within `tolerance` of the F1 of the
    *unpruned* rules; with ``tolerance=0`` that means ties are resolved in favour of the simpler hypothesis, which
    removes redundancy without changing the F1. Pruning stops when no edit is acceptable.

    The rows decide what is pruned. On rows the rules were fitted on (training rows) only a positive `tolerance`
    simplifies beyond redundancy, and it has to be tuned; on rows the rules did not see (e.g. out-of-bag rows) edits
    also undo overfitting.

    If `y` has no positive row there is nothing to measure and the rules are returned unchanged.
    """
    rules = [tuple(r) for r in rules]
    if not rules or int((y == 1).sum()) == 0:
        return rules
    positive = y == 1
    n_positive = int(positive.sum())

    def mask_of(literals):
        mask = np.ones(len(y), dtype=bool)
        for column, holds in literals:
            mask &= (X[:, column] > 0.5) == holds
        return mask

    def f1_of(covered):
        tp = int((covered & positive).sum())
        return 0.0 if tp == 0 else 2 * tp / (int(covered.sum()) + n_positive)

    masks = [mask_of(r) for r in rules]
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
                    mask = mask_of(shorter)
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
