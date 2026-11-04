"""TDLForest: TDL with a forest of bagged decision trees instead of a single tree."""
import logging
from typing import Callable, List, Optional, Set

import numpy as np
import pandas as pd
import sklearn
from sklearn.tree import DecisionTreeClassifier
from owlapy.class_expression import OWLClassExpression, OWLObjectIntersectionOf, OWLObjectUnionOf
from owlapy.owl_individual import OWLNamedIndividual

from ontolearn.learning_problem import PosNegLPStandard
from ontolearn.verbalizer import verbalize_learner_prediction

from .tdl_forest_utils import (FOREST_MODES, candidate_rules, fit_bagged_trees, majority_subsets,
                               positive_leaf_rules, positive_region_predictor, prune_rules, rule_mask,
                               select_rules, top_features)
from .tree_learner import TDL, concepts_reducer

logger = logging.getLogger(__name__)


class TDLForest(TDL):
    """TDL with a forest of bagged decision trees on the binary class-expression features.

    Every example becomes a row of binary features (each feature is one OWL class expression the example is an
    instance of), exactly as in :class:`TDL`; this class changes what is learned from those rows. A forest's majority
    vote has no compact class expression, so the modes differ in how they get back to one:

    ``"tree"``
        A single tree on all features, exactly :class:`TDL` (same hypothesis for the same ``kwargs_classifier``), but
        with a fixed seed and the same prediction/tuning interface as the forest modes. It is the reference to compare
        the forest modes against.
    ``"feature_selection"``
        ``n_trees`` bagged trees rank the features (mean impurity importance); one tree is then fitted on the
        ``top_k`` best and read back as in :class:`TDL`. Readable, the hypothesis of a single tree.
    ``"rules"``
        Candidate rules are the root-to-positive-leaf paths of all trees. Each rule's precision is estimated on the
        examples its tree did not see (out-of-bag). Rules with at least ``min_precision`` are picked by greedy set
        cover over the positives and the hypothesis is their union. Readable.
    ``"best_tree"``
        Of ``n_trees`` bagged trees, the one with the best out-of-bag F1, read as the union of its positive-leaf
        conjunctions. The control for ``"pruned"``.
    ``"pruned"``
        ``"best_tree"`` derefined on its own out-of-bag rows: literals and whole conjunctions are dropped while the
        out-of-bag F1 does not fall by more than ``prune_tolerance`` (:func:`prune_rules`).
    ``"full_rules"``
        One tree on all rows (no bagging), read as the union of its positive-leaf conjunctions. The control for
        ``"pruned_full"``.
    ``"pruned_full"``
        ``"full_rules"`` derefined on the training rows themselves: literals and conjunctions are dropped while the
        training F1 does not fall by more than ``prune_tolerance`` (:func:`prune_rules`). ``prune_tolerance=0`` only
        removes redundancy; larger values trade training F1 for shorter expressions and should be tuned by
        cross-validation.
    ``"vote"``
        The exact majority vote of ``vote_trees`` (3, 5 or 7) trees as a class expression: the union over all
        majority subsets of the intersection of the members' expressions. Exact, but its size is combinatorial in
        the number and depth of the trees, so :meth:`predict` is the practical interface.
    ``"forest_predictor"``
        The forest is the model (:meth:`predict`). For the symbolic API, :meth:`best_hypotheses` is the single tree
        with the best out-of-bag F1, which is not the forest's decision.

    The grid search of :class:`TDL` is not used; ``kwargs_classifier`` is passed to every tree. Hyperparameters can
    be changed between fits (they are plain attributes) which is what the cross-validated tuning in
    ``examples/benchmark_tdl_forest.py`` does.

    Args:
        knowledge_base: The knowledge base.
        mode: One of ``FOREST_MODES``.
        n_trees: Number of bagged trees (``feature_selection``, ``rules``, ``forest_predictor``).
        vote_trees: Number of voters for ``vote``; odd, between 3 and 7.
        top_k: Number of features kept by ``feature_selection``.
        prune_tolerance: Largest F1 drop (against the unpruned rules) that pruning may cause in ``pruned`` (out-of-bag
            rows) and ``pruned_full`` (training rows).
        min_precision: Minimum out-of-bag precision (Laplace-corrected) of a rule in ``rules``.
        max_features: Features tried per split (sklearn semantics, ``"sqrt"`` by default).
        random_state: Seed of the bootstrap samples and of the trees; fits are deterministic given the seed.
        **kwargs: Everything else is passed to :class:`TDL` (feature extractors, ``hypothesis_extraction``, ...).
    """

    def __init__(self, knowledge_base, mode: str = "feature_selection", n_trees: int = 25, vote_trees: int = 5,
                 top_k: int = 30, min_precision: float = 0.8, max_features="sqrt", random_state: int = 0,
                 prune_tolerance: float = 0.0, **kwargs):
        if mode not in FOREST_MODES:
            raise ValueError(f"mode must be one of {FOREST_MODES}. Currently: {mode!r}")
        if mode == "vote" and (vote_trees % 2 == 0 or not 3 <= vote_trees <= 7):
            raise ValueError(f"mode='vote' needs an odd number of voters between 3 and 7, got {vote_trees}")
        # Own copy: the seed of the final single tree is written into it (see `_single_tree`)
        kwargs["kwargs_classifier"] = dict(kwargs.get("kwargs_classifier") or {})
        super().__init__(knowledge_base, **kwargs)
        self.mode = mode
        self.n_trees, self.vote_trees, self.top_k = n_trees, vote_trees, top_k
        self.min_precision, self.max_features, self.random_state = min_precision, max_features, random_state
        self.prune_tolerance = prune_tolerance
        self.forest: List = []  # fitted bagged trees ('vote', 'forest_predictor')
        self.selected_features: Optional[list] = None  # columns kept by 'feature_selection'
        self.fell_back = False
        self._row_predictor: Optional[Callable[[np.ndarray], np.ndarray]] = None
        self._used_columns: List[int] = []
        # Instances of a feature expression do not depend on the fit, so they are kept across fits (fast tuning)
        self._instances_cache = dict()

    # ------------------------------------------------------------------------------------------------------------
    def fit(self, learning_problem: PosNegLPStandard = None, max_runtime: int = None):
        """Extract the features of the examples (as :meth:`TDL.fit`) and fit the forest according to ``mode``."""
        assert isinstance(learning_problem, PosNegLPStandard), \
            f"Learning problem must be PosNegLPStandard. Currently:{learning_problem}."
        if max_runtime is not None:
            self.max_runtime = max_runtime
        X, y = self.training_data(learning_problem)
        self.X, self.y = X, y
        self._maybe_plot_embeddings(X, y)
        self.fit_features(X, y)
        return self

    def training_data(self, learning_problem: PosNegLPStandard):
        """The feature matrix and labels of :meth:`TDL.create_training_data` with the rows in a canonical order.

        :class:`TDL` lists the examples in the iteration order of a ``set``, which changes from process to process
        (string hashing). That is harmless for one tree, but bagging draws rows by position, so a fixed
        ``random_state`` would not give the same forest. Here positives come first, each class sorted by IRI.
        """
        X, y = self.create_training_data(learning_problem=learning_problem, start_time=None)
        order = sorted(range(len(y)), key=lambda i: (-y.iloc[i, 0], str(y.index[i])))
        return X.iloc[order], y.iloc[order]

    def fit_features(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        """Fit on an already extracted feature matrix (columns: class expressions, rows: examples).

        Separated from :meth:`fit` so that the same matrix can be reused to fit several models, e.g. in
        cross-validation, where the (slow) feature extraction would otherwise be repeated for every model.
        """
        self.X, self.y = X, y
        self.forest, self.selected_features, self._row_predictor, self._used_columns = [], None, None, []
        self.fell_back = False  # mode='rules' found no rule precise enough and returned a single tree instead
        if X.shape[1] == 0:  # nothing to learn from: same fallback as TDL (hypothesis Thing)
            self._fit_decision_tree(X, y)
            self._report_and_plot(X, y)
            self._finalize_predictions(X, y)
            self._row_predictor = lambda rows: np.ones(len(rows), dtype=int)
            return
        Xv, yv = X.values.astype(float), y.values.ravel().astype(int)
        if self.mode == "tree":
            self._single_tree(X, y, list(range(X.shape[1])), Xv, yv)
            return
        n = {"vote": self.vote_trees, "full_rules": 0, "pruned_full": 0}.get(self.mode, self.n_trees)  # no bagging
        trees = fit_bagged_trees(Xv, yv, n, self.max_features, self.kwargs_classifier, self.random_state)
        getattr(self, f"_fit_{self.mode}")(X, y, Xv, yv, trees)

    # ----------------------------------------------------------------------------------------------- the modes
    def _single_tree(self, X: pd.DataFrame, y: pd.DataFrame, columns: List[int], Xv: np.ndarray, yv: np.ndarray):
        """Fit one tree on `columns` of the matrix and extract its hypothesis as TDL does."""
        X_sub = X.iloc[:, columns]
        self.kwargs_classifier["random_state"] = self.random_state  # a reproducible tie-breaking among equal splits
        self._fit_decision_tree(X_sub, y)
        self._report_and_plot(X_sub, y)
        self._finalize_predictions(X_sub, y)
        predictor = positive_region_predictor(self.clf, Xv[:, columns], yv, self.hypothesis_extraction)
        self._row_predictor = lambda rows: predictor(rows[:, columns])
        self._used_columns = [columns[f] for f in sorted({int(f) for f in self.clf.tree_.feature if f >= 0})]

    def _fit_feature_selection(self, X, y, Xv, yv, trees):
        keep = top_features(trees, X.shape[1], self.top_k) or list(range(X.shape[1]))
        self.selected_features = X.columns[keep].tolist()
        self.features = self.selected_features
        self._single_tree(X, y, keep, Xv, yv)

    def _fit_rules(self, X, y, Xv, yv, trees):
        chosen = select_rules(candidate_rules(trees, Xv, yv), Xv, yv, self.min_precision)
        if not chosen:
            logger.warning("mode='rules': no rule reached min_precision; falling back to a single tree.")
            self.fell_back = True
            self._single_tree(X, y, list(range(X.shape[1])), Xv, yv)
            return
        self._use_rules(X, y, Xv, yv, chosen)

    def _fit_best_tree(self, X, y, Xv, yv, trees):
        self._use_rules(X, y, Xv, yv, positive_leaf_rules(self._best_oob_tree(trees, yv, Xv)[0]))

    def _fit_pruned(self, X, y, Xv, yv, trees):
        """The tree of ``best_tree``, derefined on its out-of-bag rows (see :func:`prune_rules`)."""
        clf, oob = self._best_oob_tree(trees, yv, Xv)
        self._use_rules(X, y, Xv, yv, prune_rules(positive_leaf_rules(clf), Xv[oob], yv[oob], self.prune_tolerance))

    def _fit_full_rules(self, X, y, Xv, yv, trees):
        self._use_rules(X, y, Xv, yv, positive_leaf_rules(self._full_tree(Xv, yv)))

    def _fit_pruned_full(self, X, y, Xv, yv, trees):
        """The tree on all rows of ``full_rules``, derefined on the training rows (see :func:`prune_rules`)."""
        rules = positive_leaf_rules(self._full_tree(Xv, yv))
        self._use_rules(X, y, Xv, yv, prune_rules(rules, Xv, yv, self.prune_tolerance))

    def _full_tree(self, Xv, yv):
        kwargs = dict(self.kwargs_classifier)
        kwargs["random_state"] = self.random_state
        return DecisionTreeClassifier(**kwargs).fit(Xv, yv)

    @staticmethod
    def _best_oob_tree(trees, yv, Xv):
        """The (tree, out-of-bag rows) with the best out-of-bag F1 (the first maximum: deterministic)."""
        def oob_f1(tree):
            rows = tree[1] if len(tree[1]) > 0 else np.arange(len(yv))
            return sklearn.metrics.f1_score(yv[rows], tree[0].predict(Xv[rows]), zero_division=0)
        return max(trees, key=oob_f1)

    def _use_rules(self, X, y, Xv, yv, rules):
        """Make the union of the conjunctions `rules` (lists of ``(column, holds)``) the hypothesis and the model."""
        if not rules:
            logger.warning("No positive rule was found; falling back to a single tree.")
            self.fell_back = True
            self._single_tree(X, y, list(range(X.shape[1])), Xv, yv)
            return
        thing = self.knowledge_base.generator.thing  # the empty conjunction: a tree whose root already predicts positive
        conjunctions = [concepts_reducer([X.columns[c] if holds else X.columns[c].get_object_complement_of()
                                          for c, holds in rule], OWLObjectIntersectionOf) if rule else thing
                        for rule in rules]
        covered = np.zeros(len(yv), dtype=bool)
        for rule in rules:
            covered |= rule_mask(Xv, rule)
        self._set_hypotheses(conjunctions)
        self._set_report(y, covered.astype(int))
        self._row_predictor = lambda rows: np.any([rule_mask(rows, rule) for rule in rules], axis=0).astype(int)
        self._used_columns = sorted({c for rule in rules for c, _ in rule})

    def _fit_vote(self, X, y, Xv, yv, trees):
        thing, nothing = self.knowledge_base.generator.thing, self.knowledge_base.generator.nothing
        members = [self._tree_expression(X, clf) for clf, _ in trees]
        conjunctions, seen = [], set()
        for subset in majority_subsets(len(trees)):
            parts = [members[i] for i in subset]
            if any(p == nothing for p in parts):
                continue  # these voters can never agree
            parts = [p for p in parts if p != thing]  # a voter that accepts everything adds nothing
            conjunction = concepts_reducer(parts, OWLObjectIntersectionOf) if parts else thing
            if conjunction not in seen:
                seen.add(conjunction)
                conjunctions.append(conjunction)
        self.forest = [clf for clf, _ in trees]
        self._set_hypotheses(conjunctions or [nothing])
        self._set_forest_model(y, Xv)

    def _fit_forest_predictor(self, X, y, Xv, yv, trees):
        def oob_f1(clf, oob):
            rows = oob if len(oob) > 0 else np.arange(len(yv))
            return sklearn.metrics.f1_score(yv[rows], clf.predict(Xv[rows]), zero_division=0)
        best = max(trees, key=lambda t: oob_f1(*t))[0]  # first maximum: deterministic
        self.forest = [clf for clf, _ in trees]
        self.clf = best
        self._report_and_plot(X, y)
        self._finalize_predictions(X, y)
        self._set_forest_model(y, Xv)

    def _set_forest_model(self, y, Xv):
        """Make the vote of `self.forest` the model behind :meth:`predict`."""
        self._row_predictor = lambda rows: (self._votes(rows) > 0.5).astype(int)
        self._used_columns = sorted({int(f) for clf in self.forest for f in clf.tree_.feature if f >= 0})
        self._set_report(y, self._row_predictor(Xv))

    # --------------------------------------------------------------------------------------------------- helpers
    def _votes(self, rows: np.ndarray) -> np.ndarray:
        return np.mean([clf.predict(rows) for clf in self.forest], axis=0)

    def _tree_expression(self, X: pd.DataFrame, clf) -> OWLClassExpression:
        """The class expression a single tree predicts positive with: the union of its positive-leaf conjunctions."""
        previous, self.clf = self.clf, clf
        try:
            return concepts_reducer(self._expressions_from_predicted_leaves(X), OWLObjectUnionOf)
        finally:
            self.clf = previous

    def _set_report(self, y: pd.DataFrame, y_pred: np.ndarray) -> None:
        if self.report_classification:
            self._TDL__classification_report = "Classification Report: Negatives: -1 and Positives 1 \n" + \
                sklearn.metrics.classification_report(y.values.ravel(), y_pred, target_names=["Negative", "Positive"],
                                                      zero_division=0)
            if self.verbose > 0:
                logger.info(self._TDL__classification_report)

    def _set_hypotheses(self, conjunctions: List[OWLClassExpression]) -> None:
        """Store `conjunctions` as the hypothesis (their union) when no single tree stands behind it."""
        self.clf = None
        self.owl_class_expressions.clear()
        self.owl_class_expressions.update(conjunctions)
        self.conjunctive_concepts = list(conjunctions)
        self.disjunction_of_conjunctive_concepts = concepts_reducer(conjunctions, OWLObjectUnionOf)
        if self.verbalize:
            verbalize_learner_prediction(self.disjunction_of_conjunctive_concepts)

    def _instances(self, expression: OWLClassExpression) -> Set[OWLNamedIndividual]:
        if expression not in self._instances_cache:
            self._instances_cache[expression] = set(self.knowledge_base.individuals(expression))
        return self._instances_cache[expression]

    # ---------------------------------------------------------------------------------------------------- predict
    def predict_rows(self, rows: np.ndarray, proba: bool = False) -> np.ndarray:
        """Predictions for feature rows (same columns as the training matrix), without a reasoner.

        ``proba=True`` returns the fraction of positive votes for the forest modes (``vote``, ``forest_predictor``);
        the other modes are a single hypothesis, so their output is always 0/1.
        """
        if self._row_predictor is None:
            raise RuntimeError("fit first")
        if proba and self.mode in ("vote", "forest_predictor"):
            return self._votes(rows)
        return self._row_predictor(rows)

    def predict(self, X: List[OWLNamedIndividual], proba=True) -> np.ndarray:
        """Classify individuals with the fitted model.

        For ``vote`` and ``forest_predictor`` this is the forest's vote (the fraction of positive votes if `proba`),
        for ``feature_selection`` and ``rules`` the 0/1 decision of the hypothesis. Only the feature expressions the
        model actually uses are tested (one retrieval each, cached), so the individuals need not have been in the
        training data.
        """
        rows = np.zeros((len(X), self.X.shape[1]))
        for column in self._used_columns:
            instances = self._instances(self.X.columns[column])
            rows[:, column] = [1.0 if individual in instances else 0.0 for individual in X]
        return self.predict_rows(rows, proba=proba)
