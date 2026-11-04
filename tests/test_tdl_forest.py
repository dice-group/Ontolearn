import json

import numpy as np
import pytest
from owlapy.owl_individual import OWLNamedIndividual

from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.learners import TDL, TDLForest
from ontolearn.learners.tdl_forest_utils import (candidate_rules, fit_bagged_trees, majority_subsets,
                                                 positive_leaf_rules, positive_region_predictor, prune_rules,
                                                 rule_mask, rules_f1, select_rules, top_features)


def _toy(n=120, seed=0):
    """Column 0 and 1 decide the label (y = c0 AND NOT c1), the other 8 columns are noise."""
    rng = np.random.RandomState(seed)
    X = rng.randint(0, 2, size=(n, 10)).astype(float)
    y = ((X[:, 0] == 1) & (X[:, 1] == 0)).astype(int)
    return X, y


def test_bagged_trees_are_deterministic_and_have_disjoint_oob_rows():
    X, y = _toy()
    a = fit_bagged_trees(X, y, 5, None, {}, random_state=3)
    b = fit_bagged_trees(X, y, 5, None, {}, random_state=3)
    for (ca, oa), (cb, ob) in zip(a, b):
        assert np.array_equal(oa, ob)
        assert np.array_equal(ca.predict(X), cb.predict(X))
    assert all(len(oob) > 0 for _, oob in a)  # a bootstrap sample leaves ~37% of the rows out


def test_top_features_finds_the_informative_columns():
    X, y = _toy()
    trees = fit_bagged_trees(X, y, 15, None, {}, random_state=0)
    assert top_features(trees, X.shape[1], 2) == [0, 1]


def test_positive_leaf_rules_reproduce_the_tree_predictions():
    X, y = _toy()
    clf = fit_bagged_trees(X, y, 1, None, {}, random_state=0)[0][0]
    covered = np.zeros(len(y), dtype=bool)
    for rule in positive_leaf_rules(clf):
        covered |= rule_mask(X, rule)
    assert np.array_equal(covered.astype(int), clf.predict(X))


def test_select_rules_covers_positives_and_rejects_imprecise_rules():
    X, y = _toy()
    trees = fit_bagged_trees(X, y, 15, None, {}, random_state=0)
    chosen = select_rules(candidate_rules(trees, X, y), X, y, min_precision=0.8)
    assert chosen
    covered = np.zeros(len(y), dtype=bool)
    for rule in chosen:
        covered |= rule_mask(X, rule)
    assert (covered & (y == 1)).sum() >= 0.9 * y.sum()  # most positives covered
    assert (covered & (y == 0)).sum() <= 2  # hardly any negative
    # Nothing is precise enough at an unreachable threshold
    assert select_rules(candidate_rules(trees, X, y), X, y, min_precision=1.01) == []


def test_prune_rules_drops_what_held_out_rows_do_not_support():
    # column 0 is the signal; column 1 is noise that the first rule memorised; the second rule is spurious.
    X = np.array([[1, 1], [1, 0], [1, 0], [0, 1], [0, 0], [0, 0], [0, 1], [0, 0]], dtype=float)
    y = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    rules = [((0, True), (1, True)), ((1, True), (0, False))]
    # on the rows the rules were not fitted on: [1,0] positives need column 1 dropped, the second rule covers negatives
    pruned = prune_rules(rules, X, y)
    assert pruned == [((0, True),)]
    assert rules_f1(X, y, pruned) == 1.0 > rules_f1(X, y, rules)


def test_prune_rules_never_gets_worse_and_is_a_noop_without_positives():
    rng = np.random.RandomState(0)
    X = (rng.rand(60, 6) > 0.5).astype(float)
    y = (X[:, 0] * X[:, 1] + (rng.rand(60) > 0.9) > 0).astype(int)
    rules = [((0, True), (1, True), (2, True)), ((3, False), (4, True)), ((5, True),)]
    pruned = prune_rules(rules, X, y)
    assert rules_f1(X, y, pruned) >= rules_f1(X, y, rules)
    assert sum(map(len, pruned)) <= sum(map(len, rules)) and len(pruned) <= len(rules)
    assert prune_rules(rules, X, np.zeros(60, dtype=int)) == rules
    assert prune_rules([], X, y) == []


def test_pruned_hypothesis_is_not_longer_than_best_tree():
    from owlapy import owl_expression_to_dl
    kb, lp, pos, neg = _uncle()
    kwargs = dict(knowledge_base=kb, verbose=0, report_classification=False, n_trees=15)
    best = TDLForest(mode='best_tree', **kwargs).fit(lp)
    pruned = TDLForest(mode='pruned', **kwargs).fit(lp)
    assert not pruned.fell_back
    assert len(owl_expression_to_dl(pruned.best_hypotheses(1))) <= len(owl_expression_to_dl(best.best_hypotheses(1)))


def test_a_tree_that_predicts_positive_everywhere_gives_thing():
    # The root-only tree has the empty rule; its conjunction is Thing (concepts_reducer would return None).
    kb, lp, pos, neg = _uncle()
    model = TDLForest(knowledge_base=kb, mode='pruned', verbose=0, report_classification=False)
    X, y = model.training_data(lp)
    model.X, model.y = X, y
    model._use_rules(X, y, X.values.astype(float), y.values.ravel().astype(int), [()])
    assert model.best_hypotheses(1) == kb.generator.thing
    assert model.predict_rows(X.values.astype(float)).tolist() == [1] * len(y)


def test_majority_subsets_counts():
    assert len(majority_subsets(3)) == 3  # C(3, 2)
    assert len(majority_subsets(5)) == 10  # C(5, 3)
    assert all(len(s) == 3 for s in majority_subsets(5))


def test_invalid_arguments_are_rejected():
    kb = KnowledgeBase(path='KGs/Family/family-benchmark_rich_background.owl')
    with pytest.raises(ValueError):
        TDLForest(knowledge_base=kb, mode='bagging')
    with pytest.raises(ValueError):
        TDLForest(knowledge_base=kb, mode='vote', vote_trees=4)  # even: ties
    with pytest.raises(ValueError):
        TDLForest(knowledge_base=kb, mode='vote', vote_trees=9)  # expression would explode


def test_tdl_itself_has_no_ensemble_options():
    # The forest lives in its own class; TDL keeps its original signature.
    import inspect
    assert 'ensemble' not in inspect.signature(TDL.__init__).parameters


def _uncle():
    with open('examples/synthetic_problems.json') as f:
        settings = json.load(f)
    ex = settings['problems']['Uncle']
    pos = {OWLNamedIndividual(i) for i in ex['positive_examples']}
    neg = {OWLNamedIndividual(i) for i in ex['negative_examples']}
    from ontolearn.learning_problem import PosNegLPStandard
    return KnowledgeBase(path=settings['data_path'][3:]), PosNegLPStandard(pos=pos, neg=neg), pos, neg


def _f1(covered, pos, neg):
    tp, fp = len(covered & pos), len(covered & neg)
    return 2 * tp / (2 * tp + fp + (len(pos) - tp))


@pytest.mark.parametrize('mode', ['tree', 'feature_selection', 'rules', 'vote', 'forest_predictor', 'best_tree',
                                  'pruned'])
def test_every_mode_learns_a_good_hypothesis(mode):
    kb, lp, pos, neg = _uncle()
    model = TDLForest(knowledge_base=kb, mode=mode, verbose=0, report_classification=False, n_trees=15, vote_trees=3,
                      top_k=20, kwargs_classifier={'max_depth': 3} if mode == 'vote' else None).fit(lp)
    f1 = _f1(set(kb.individuals(model.best_hypotheses(1))), pos, neg)
    # Sanity floor, not a quality claim: the trivial hypothesis (Thing) has F1 = 2/3 on this balanced problem.
    assert f1 >= 0.72


@pytest.mark.parametrize('mode, kwargs', [
    ('tree', {}),
    ('feature_selection', {}),
    ('feature_selection', {'hypothesis_extraction': 'predicted_leaves'}),
    ('rules', {}),
    ('best_tree', {}),
    ('pruned', {}),
    ('vote', {'vote_trees': 3, 'kwargs_classifier': {'max_depth': 3}}),
])
def test_row_predictions_equal_retrieving_the_hypothesis(mode, kwargs):
    # The cross-validated tuning scores models on feature rows (no reasoner). That is only valid if those predictions
    # are exactly what retrieving the extracted class expression gives.
    kb, lp, pos, neg = _uncle()
    model = TDLForest(knowledge_base=kb, mode=mode, verbose=0, report_classification=False, n_trees=15, top_k=20,
                      **kwargs).fit(lp)
    individuals = sorted(pos | neg, key=str)
    covered = set(kb.individuals(model.best_hypotheses(1)))
    assert model.predict(individuals, proba=False).tolist() == [1 if i in covered else 0 for i in individuals]
    # The row predictor itself, on rows built from the reasoner (not from the training matrix: a few of its cells,
    # e.g. for "∀ r.⊤", disagree with the reasoner, see test_training_matrix_semantics below).
    rows = np.array([[1.0 if i in set(kb.individuals(c)) else 0.0 for c in model.X.columns] for i in individuals])
    assert model.predict_rows(rows).tolist() == [1 if i in covered else 0 for i in individuals]


@pytest.mark.parametrize('extraction', ['positive_paths', 'predicted_leaves', 'factored_tree'])
def test_positive_region_predictor_matches_tdl_for_a_single_tree(extraction):
    kb, lp, pos, neg = _uncle()
    tdl = TDL(knowledge_base=kb, verbose=0, report_classification=False, hypothesis_extraction=extraction,
              kwargs_classifier={'max_depth': 4}).fit(lp)
    covered = set(kb.individuals(tdl.best_hypotheses(1)))
    individuals = sorted(pos | neg, key=str)
    predictor = positive_region_predictor(tdl.clf, tdl.X.values.astype(float), tdl.y.values.ravel().astype(int),
                                          extraction)
    assert predictor(tdl.X.loc[individuals].values.astype(float)).tolist() == \
        [1 if i in covered else 0 for i in individuals]


def test_fits_are_deterministic_across_processes():
    # Bagging draws rows by position, so the row order must not depend on set iteration order, which changes with
    # PYTHONHASHSEED. Two processes with different hash seeds have to learn the very same hypothesis.
    import os
    import subprocess
    import sys
    script = ('import json; from owlapy import owl_expression_to_dl as dl; '
              'from owlapy.owl_individual import OWLNamedIndividual as I; '
              'from ontolearn.knowledge_base import KnowledgeBase; from ontolearn.learners import TDLForest; '
              'from ontolearn.learning_problem import PosNegLPStandard; '
              "s = json.load(open('examples/synthetic_problems.json')); e = s['problems']['Uncle']; "
              "lp = PosNegLPStandard(pos={I(i) for i in e['positive_examples']}, "
              "neg={I(i) for i in e['negative_examples']}); "
              "kb = KnowledgeBase(path=s['data_path'][3:]); "
              "m = TDLForest(knowledge_base=kb, mode='MODE', verbose=0, report_classification=False, "
              "n_trees=9, vote_trees=3, kwargs_classifier={'max_depth': 3}).fit(lp); "
              'print(sorted(dl(c) for c in m.conjunctive_concepts))')  # sorted: the union order is a set order
    for mode in ('rules', 'feature_selection', 'forest_predictor'):
        outputs = set()
        for hash_seed in ('0', '1'):
            result = subprocess.run([sys.executable, '-c', script.replace('MODE', mode)],
                                    env={**os.environ, 'PYTHONHASHSEED': hash_seed}, capture_output=True, text=True,
                                    check=True)
            outputs.add(result.stdout.strip().splitlines()[-1])
        assert len(outputs) == 1, mode


def test_training_matrix_semantics():
    # Documents a property of TDL's feature extraction that the evaluation has to respect: a handful of cells of the
    # training matrix (universal restrictions such as "∀ hasParent⁻.⊤", vacuously true for individuals without such
    # a successor) say False where the reasoner says True. Everything else agrees. Hypotheses are therefore always
    # scored by retrieving them from the reasoner, never from the matrix.
    kb, lp, pos, neg = _uncle()
    X, _ = TDLForest(knowledge_base=kb, verbose=0).training_data(lp)
    disagreeing = [c for c in X.columns
                   if any((i in set(kb.individuals(c))) != bool(X.loc[i, c]) for i in X.index)]
    assert 0 < len(disagreeing) <= 0.02 * X.shape[1]
    assert all(type(c).__name__ == 'OWLObjectAllValuesFrom' for c in disagreeing)


def test_tree_mode_is_exactly_tdl():
    kb, lp, _, _ = _uncle()
    reference = TDL(knowledge_base=kb, verbose=0, report_classification=False,
                    kwargs_classifier={'random_state': 0}).fit(lp)
    forest = TDLForest(knowledge_base=kb, mode='tree', random_state=0, verbose=0, report_classification=False).fit(lp)
    # Same features and (up to the row order, which does not matter for one tree without bagging) the same
    # hypothesis semantics
    individuals = sorted(lp.pos | lp.neg, key=str)
    covered = lambda m: {i for i in individuals if i in set(kb.individuals(m.best_hypotheses(1)))}  # noqa: E731
    assert covered(reference) == covered(forest)


def test_proba_is_the_vote_fraction():
    kb, lp, pos, neg = _uncle()
    model = TDLForest(knowledge_base=kb, mode='forest_predictor', n_trees=10, verbose=0,
                      report_classification=False).fit(lp)
    proba = model.predict(sorted(pos | neg, key=str), proba=True)
    assert ((proba >= 0) & (proba <= 1)).all() and len(set(np.round(proba * 10, 6) % 1)) == 1  # multiples of 1/10


def test_tdl_default_is_unchanged():
    kb, lp, _, _ = _uncle()
    model = TDL(knowledge_base=kb, verbose=0, report_classification=False).fit(lp)
    assert model.clf is not None
    with pytest.raises(NotImplementedError):
        model.predict([])
