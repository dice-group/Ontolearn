"""Does a forest generalise better than a single tree in TDL? A benchmark of ``TDLForest`` against the tree.

Protocol (chosen to make the comparison about generalisation and nothing else):

* **Repeated stratified k-fold CV** (default 5 folds x 4 repeats) on every learning problem. Each example is tested
  ``repeats`` times and a single unlucky split cannot decide the outcome (the Family problems have only 8-14 test
  examples per fold, so a single 80/20 split moves the F1 by whole examples).
* **Everything is fitted on the training fold only**: the features are extracted from the training examples, and the
  hyperparameters are chosen by an **inner** 3-fold CV inside the training fold ("nested CV"). The test fold is only
  used to score the final model. Tuned and untuned variants of the single tree and of the forests are both reported
  (the forests are not handicapped by untuned defaults, nor is the tree).
* **Scored under the reasoner's semantics**: a symbolic hypothesis is scored on the test fold by retrieving it from
  the knowledge base (the training matrix of TDL disagrees with the reasoner in a few cells, so matrix predictions
  would not measure the class expression). The majority vote of ``vote``/``forest_vote`` is scored through
  ``predict``.
* **Statistics at the level of the problem.** The 20 problems are the independent units, the folds of a problem are
  not (they share training data). Across problems: paired Wilcoxon signed-rank on the per-problem mean difference
  with a Holm correction, a Friedman test over all configurations, and a bootstrap CI over problems. Within a
  problem: the Nadeau-Bengio corrected resampled t-test, which accounts for the overlap of the training folds.

    python examples/benchmark_tdl_forest.py run --repeats 4 --workers 16 --out results.jsonl
    python examples/benchmark_tdl_forest.py report --out results.jsonl
"""
import argparse
import itertools
import json
import logging
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

FOLDS = 5
INNER_FOLDS = 3

# name -> (mode, parameters of TDLForest, 'hypothesis' = score the class expression | 'predict' = score the vote)
UNTUNED = {
    'tree': ('tree', {}, 'hypothesis'),
    'fs': ('feature_selection', {}, 'hypothesis'),
    'rules': ('rules', {}, 'hypothesis'),
    'vote5': ('vote', {'vote_trees': 5}, 'predict'),
    'forest_tree': ('forest_predictor', {}, 'hypothesis'),
    'forest_vote': ('forest_predictor', {}, 'predict'),
    'best_tree': ('best_tree', {}, 'hypothesis'),  # the control of 'pruned': same tree, not derefined
    'pruned': ('pruned', {}, 'hypothesis'),
}
# name -> (mode, grid of parameter dicts (simplest first, ties keep the first), scoring)
TUNED = {
    'tree_t': ('tree', [{'kwargs_classifier': {'max_depth': d, 'min_samples_leaf': leaf}}
                        for d, leaf in itertools.product((None, 3, 5), (1, 2, 5))], 'hypothesis'),
    'fs_t': ('feature_selection', [{'top_k': k} for k in (10, 30, 100, 300)], 'hypothesis'),
    'rules_t': ('rules', [{'min_precision': p} for p in (0.7, 0.8, 0.9, 0.95)], 'hypothesis'),
    'forest_vote_t': ('forest_predictor', [{'n_trees': n, 'max_features': f, 'kwargs_classifier': {'min_samples_leaf': leaf}}
                                           for n, f, leaf in itertools.product((25, 100), ('sqrt', None), (1, 3))],
                      'predict'),
}
_TREE_GRID = [{'kwargs_classifier': {'max_depth': d, 'min_samples_leaf': leaf}}
              for d, leaf in itertools.product((None, 3, 5), (1, 2, 5))]
_TOLERANCES = [{'prune_tolerance': t} for t in (0.0, 0.01, 0.02, 0.05, 0.1)]
LAMBDA = 0.01  # F1 points per doubling of the expression length, used as a tuning objective by the '*_l' configs
UNTUNED['full_rules'] = ('full_rules', {}, 'hypothesis')  # the control of 'pruned_full': same tree, not derefined
UNTUNED['pruned_full0'] = ('pruned_full', {'prune_tolerance': 0.0}, 'hypothesis')  # lossless: only removes redundancy
TUNED['pruned_full_t'] = ('pruned_full', _TOLERANCES, 'hypothesis')  # tolerance tuned for F1
# the same, but tuned for F1 - LAMBDA * log2(length): the 4th element is the weight of the length
TUNED['tree_l'] = ('tree', _TREE_GRID, 'hypothesis', LAMBDA)
TUNED['pruned_full_l'] = ('pruned_full', _TOLERANCES, 'hypothesis', LAMBDA)
TUNED['best_tree_t'] = ('best_tree', _TREE_GRID, 'hypothesis')
TUNED['pruned_t'] = ('pruned', _TREE_GRID, 'hypothesis')
CONFIGS = list(UNTUNED) + list(TUNED)
BASELINES = {'untuned': 'tree', 'tuned': 'tree_t', 'derefinement, untuned': 'best_tree', 'derefinement, tuned': 'best_tree_t',
             'full-tree derefinement': 'full_rules'}  # what each family of configurations is compared to
FAMILIES = {'untuned': [c for c in UNTUNED if c != 'tree'], 'tuned': [c for c in TUNED if c != 'tree_t'],
            'derefinement, untuned': ['pruned'], 'derefinement, tuned': ['pruned_t'], 'full-tree derefinement': ['pruned_full0']}  # pruned vs. its unpruned control

with open('LPs/Family/lps.json') as _f:
    _FAMILY = list(json.load(_f)['problems'])
# name -> (kb path, lps file, problem key)
PROBLEMS = {f'Family-{k}': ('KGs/Family/family-benchmark_rich_background.owl', 'LPs/Family/lps.json', k)
            for k in _FAMILY}
PROBLEMS['Mutagenesis'] = ('KGs/Mutagenesis/mutagenesis.owl', 'LPs/Mutagenesis/lps.json', 'NotKnown')
PROBLEMS['Carcinogenesis'] = ('KGs/Carcinogenesis/carcinogenesis.owl', 'LPs/Carcinogenesis/lps.json', 'NOTKNOWN')

_KB, _INSTANCES = {}, {}


def _scores(predicted_positive, pos, neg):
    """F1 of the positive class and accuracy, given the set of individuals predicted positive."""
    tp, fp = len(predicted_positive & pos), len(predicted_positive & neg)
    fn, tn = len(pos) - tp, len(neg) - fp
    f1 = 0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)
    return f1, (tp + tn) / (len(pos) + len(neg))


_ONLY = None  # restrict run_fold to these configurations (set by --configs; inherited by forked workers)


def run_fold(task):
    """Fit and score every configuration on one outer fold of one problem; returns one row per configuration."""
    problem, repeat, fold = task
    from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
    from owlapy import owl_expression_to_dl
    from owlapy.owl_individual import OWLNamedIndividual
    from ontolearn.knowledge_base import KnowledgeBase
    from ontolearn.learners import TDLForest
    from ontolearn.learning_problem import PosNegLPStandard

    logging.getLogger('ontolearn.learners.tree_learner_forest').setLevel(logging.ERROR)  # 'rules' fallbacks: see report
    kb_path, lps_path, key = PROBLEMS[problem]
    if kb_path not in _KB:
        _KB[kb_path] = KnowledgeBase(path=kb_path)
    kb = _KB[kb_path]
    with open(lps_path) as f:
        ex = json.load(f)['problems'][key]
    examples = sorted(ex['positive_examples']) + sorted(ex['negative_examples'])
    labels = np.array([1] * len(ex['positive_examples']) + [0] * len(ex['negative_examples']))
    splits = list(RepeatedStratifiedKFold(n_splits=FOLDS, n_repeats=repeat + 1, random_state=0)
                  .split(np.zeros(len(labels)), labels))
    train_idx, test_idx = splits[repeat * FOLDS + fold]
    mk_inds = lambda idx: [OWLNamedIndividual(examples[i]) for i in idx]  # noqa: E731
    train_inds, test_inds = mk_inds(train_idx), mk_inds(test_idx)
    p_te = {i for i, label in zip(test_inds, labels[test_idx]) if label == 1}
    n_te = set(test_inds) - p_te
    lp_train = PosNegLPStandard(pos={i for i, label in zip(train_inds, labels[train_idx]) if label == 1},
                                neg={i for i, label in zip(train_inds, labels[train_idx]) if label == 0})

    # Instances of a feature expression restricted to this problem's examples, shared by all fits of the problem
    universe = frozenset(OWLNamedIndividual(e) for e in examples)
    cache = _INSTANCES.setdefault(problem, {})

    def instances(expression):
        if expression not in cache:
            cache[expression] = frozenset(set(kb.individuals(expression)) & universe)
        return cache[expression]

    seed = repeat

    def make(mode, params):
        learner = TDLForest(knowledge_base=kb, mode=mode, random_state=seed, verbose=0, report_classification=False,
                            **params)
        learner._instances = instances  # restricted + shared retrieval (identical results, much cheaper)
        return learner

    # (1) Features are extracted from the training fold only; the rows are shared by all configurations
    X, y = make('tree', {}).training_data(lp_train)
    y_train = y.values.ravel().astype(int)

    def inner_f1(pred_positive, truth):
        tp = int(((pred_positive == 1) & (truth == 1)).sum())
        fp = int(((pred_positive == 1) & (truth == 0)).sum())
        fn = int(((pred_positive == 0) & (truth == 1)).sum())
        return 0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)

    def tune(mode, grid, length_weight=0.0):
        """The parameters with the best mean F1 (minus `length_weight` per doubling of the hypothesis length) over an
        inner CV of the training fold; the test fold is never seen."""
        inner = list(StratifiedKFold(INNER_FOLDS, shuffle=True, random_state=seed).split(np.zeros(len(y_train)),
                                                                                         y_train))
        best, best_score = None, -1.0
        for params in grid:
            score = 0.0
            for tr, va in inner:
                model = make(mode, params)
                model.fit_features(X.iloc[tr], y.iloc[tr])
                fold_score = inner_f1(model.predict(list(X.index[va]), proba=False), y_train[va])
                if length_weight:
                    fold_score -= length_weight * np.log2(len(owl_expression_to_dl(model.best_hypotheses(1))))
                score += fold_score / len(inner)
            if score > best_score + 1e-12:  # ties keep the earlier (simpler) setting
                best, best_score = params, score
        return best

    rows = []
    jobs = [(name, mode, [params], scoring) for name, (mode, params, scoring) in UNTUNED.items()]  # (+ length weight)
    jobs += [(name, mode, grid, scoring, *rest) for name, (mode, grid, scoring, *rest) in TUNED.items()]
    jobs = [j for j in jobs if _ONLY is None or j[0] in _ONLY]
    for name, mode, grid, scoring, *weight in jobs:
        start = time.time()
        params = grid[0] if len(grid) == 1 else tune(mode, grid, weight[0] if weight else 0.0)
        model = make(mode, params)
        model.fit_features(X, y)
        fit_time = time.time() - start
        if scoring == 'predict':
            predicted = {i for i, p in zip(test_inds, model.predict(test_inds, proba=False)) if p == 1}
            retrieval = [None]
            length = None  # the equivalent expression has C(n, n//2+1) conjunctions
        else:
            hypothesis = model.best_hypotheses(1)
            # Time to retrieve all instances of the hypothesis from the KB: one warm-up (fills the reasoner's lazy
            # per-class/per-property caches), then the minimum of three runs. Cold-start costs are not what we compare.
            kb.individuals(hypothesis)
            retrieval = []
            for _ in range(3):
                t0 = time.perf_counter()
                kb.individuals(hypothesis)
                retrieval.append(time.perf_counter() - t0)
            predicted = set(kb.individuals(hypothesis)) & set(test_inds)
            length = len(owl_expression_to_dl(hypothesis))
        f1, accuracy = _scores(predicted, p_te, n_te)
        rows.append(dict(problem=problem, repeat=repeat, fold=fold, config=name, f1=f1, accuracy=accuracy,
                         length=length, fit_time=fit_time, retrieval_time=None if retrieval[0] is None else min(retrieval), params=params if len(grid) > 1 else None,
                         n_test=len(test_inds), fell_back=model.fell_back))
    return rows


def cmd_run(args):
    tasks = [(p, r, k) for r in range(args.repeats) for k in range(FOLDS) for p in PROBLEMS]
    done = set()
    try:
        with open(args.out) as f:
            done = {(r['problem'], r['repeat'], r['fold']) for r in map(json.loads, f)}  # per file: new configs go to a new file
    except FileNotFoundError:
        pass
    tasks = [t for t in tasks if t not in done]
    global _ONLY
    _ONLY = set(args.configs.split(',')) if args.configs else None
    print(f'{len(tasks)} outer folds to do ({len(done)} done in {args.out})', flush=True)
    # ProcessPoolExecutor (non-daemonic workers): owlready2 spawns a child process while parsing an ontology
    with ProcessPoolExecutor(args.workers) as pool, open(args.out, 'a') as out:
        futures = [pool.submit(run_fold, t) for t in tasks]
        for i, fut in enumerate(as_completed(futures), 1):
            for row in fut.result():
                out.write(json.dumps(row) + '\n')
            out.flush()
            if i % 20 == 0:
                print(f'{i}/{len(tasks)}', flush=True)


def holm(pvalues):
    """Holm step-down adjusted p-values (same order as the input)."""
    order = np.argsort(pvalues)
    adjusted, running = np.empty(len(pvalues)), 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(pvalues) - rank) * pvalues[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def corrected_resampled_t(diffs, test_fraction):
    """Nadeau-Bengio corrected resampled t-test on the per-fold differences of one problem; returns the p-value."""
    from scipy.stats import t as student
    d = np.asarray(diffs, dtype=float)
    if len(d) < 2 or d.std(ddof=1) == 0:
        return 1.0 if d.mean() == 0 else 0.0
    variance = (1 / len(d) + test_fraction / (1 - test_fraction)) * d.var(ddof=1)
    return float(2 * student.sf(abs(d.mean() / np.sqrt(variance)), len(d) - 1))


LAMBDAS = (0.0, 0.005, 0.01, 0.02, 0.05)  # F1 points charged per doubling of the length


def length_aware_report(per, problems, rng):
    """Judge the configurations by F1 *and* length (characters of the DL rendering of the hypothesis).

    The length of a hypothesis is a quality in its own right (readability, less overfitting). Three views:
    the Pareto front of (F1, length); the average rank under S = F1 - lambda * log2(length) for several lambdas
    (a doubling of the length costs lambda F1); and the break-even lambda against the tuned tree, the exchange rate at
    which a configuration that is shorter but less accurate becomes the better choice.
    """
    configs = [c for c in CONFIGS if all(per['length'][c].get(p) for p in problems)]
    f1 = {c: np.array([np.mean(list(per['f1'][c][p].values())) for p in problems]) for c in configs}
    log_length = {c: np.array([np.mean(np.log2(list(per['length'][c][p].values()))) for p in problems])
                  for c in configs}
    print('\n== length-aware evaluation (length = characters of the DL expression; log2 per doubling) ==')
    means = {c: (f1[c].mean(), log_length[c].mean()) for c in configs}
    front = [c for c in configs if not any(o != c and means[o][0] >= means[c][0] and means[o][1] <= means[c][1] and
                                           means[o] != means[c] for o in configs)]
    print(f'{"config":<15}{"F1":>8}{"log2 len":>10}{"geo len":>9}  Pareto' + ''.join(f'{"rank@" + str(x):>10}' for x in LAMBDAS))
    ranks = {}
    for lam in LAMBDAS:
        matrix = np.array([[f1[c][i] - lam * log_length[c][i] for c in configs] for i in range(len(problems))])
        ranks[lam] = np.mean([(-row).argsort().argsort() + 1 for row in matrix], axis=0)
    for k, c in enumerate(sorted(configs, key=lambda c: -means[c][0])):
        index = configs.index(c)
        print(f'{c:<15}{means[c][0]:>8.4f}{means[c][1]:>10.2f}{2 ** means[c][1]:>9.0f}  {"*" if c in front else " ":<6}' +
              ''.join(f'{ranks[lam][index]:>11.2f}' for lam in LAMBDAS))
    base = BASELINES['tuned']
    print(f'\nbreak-even lambda vs. {base}: delta F1 / delta log2(length). A configuration that is shorter and less '
          f'accurate is the better choice once a doubling of the length is worth more than this many F1 points.')
    for c in configs:
        if c == base:
            continue
        d_f1, d_len = f1[c] - f1[base], log_length[c] - log_length[base]
        boot = []
        for _ in range(5000):
            idx = rng.randint(0, len(d_f1), len(d_f1))
            if abs(d_len[idx].mean()) > 1e-9:
                boot.append(d_f1[idx].mean() / d_len[idx].mean())
        ratio = d_f1.mean() / d_len.mean() if abs(d_len.mean()) > 1e-9 else float('nan')
        verdict = ('dominates (F1 up, length down)' if d_f1.mean() >= 0 and d_len.mean() <= 0 else
                   'dominated (F1 down, length up)' if d_f1.mean() <= 0 and d_len.mean() >= 0 else
                   f'shorter, preferred if lambda > {ratio:.4f}' if d_len.mean() < 0 else
                   f'longer, preferred if lambda < {ratio:.4f}')
        print(f'{c:<15}dF1={d_f1.mean():+.4f}  dlog2len={d_len.mean():+.2f}  '
              f'ratio 95% CI [{np.percentile(boot, 2.5):+.4f}, {np.percentile(boot, 97.5):+.4f}]  {verdict}')
    print()


def cmd_report(args):
    from scipy.stats import friedmanchisquare, wilcoxon
    rows = [json.loads(line) for out in args.out for line in open(out)]
    global CONFIGS
    CONFIGS = [c for c in CONFIGS if any(r['config'] == c for r in rows)]
    problems = [p for p in PROBLEMS if any(r['problem'] == p for r in rows)]
    # per[metric][config][problem] -> {(repeat, fold): value}, so differences are paired by fold
    per = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for r in rows:
        for metric in ('f1', 'accuracy', 'length', 'fit_time'):
            if r[metric] is not None:
                per[metric][r['config']][r['problem']][(r['repeat'], r['fold'])] = r[metric]
    mean = lambda metric, config, problem: float(np.mean(list(per[metric][config][problem].values())))  # noqa: E731
    n_folds = len({(r['repeat'], r['fold']) for r in rows})
    print(f'{len(problems)} problems, {n_folds} outer folds per problem '
          f'({len({r["repeat"] for r in rows})} repeats of {FOLDS}-fold CV), inner CV {INNER_FOLDS}-fold\n')

    groups = {'Family (mean of 18)': [p for p in problems if p.startswith('Family')],
              'Mutagenesis': ['Mutagenesis'], 'Carcinogenesis': ['Carcinogenesis'], 'ALL (mean of problems)': problems}
    for metric in ('f1', 'accuracy', 'length', 'fit_time'):
        print(f'== {metric}: mean over the problems of the per-problem mean over folds ==')
        print(f'{"":<24}' + ''.join(f'{c:>14}' for c in CONFIGS))
        for group, members in groups.items():
            members = [p for p in members if p in problems]
            cells = []
            for c in CONFIGS:
                values = [mean(metric, c, p) for p in members if p in per[metric][c]]
                cells.append('-' if not values else f'{np.mean(values):.3f}' if metric in ('f1', 'accuracy')
                             else f'{np.mean(values):.1f}')
            print(f'{group:<24}' + ''.join(f'{x:>14}' for x in cells))
        print()

    print('== Friedman test over problems (F1; the per-problem mean F1 of every configuration) ==')
    matrix = np.array([[mean('f1', c, p) for c in CONFIGS] for p in problems])
    ranks = np.mean([(-row).argsort().argsort() + 1 for row in matrix], axis=0)  # 1 = best
    statistic, p = friedmanchisquare(*matrix.T)
    print(f'chi2={statistic:.2f}  p={p:.4f}   average rank (1 = best): ' +
          '  '.join(f'{c}={r:.2f}' for c, r in zip(CONFIGS, ranks)) + '\n')

    rng = np.random.RandomState(0)
    for family, configs in FAMILIES.items():
        base = BASELINES[family]
        configs = [c for c in configs if c in CONFIGS]
        if base not in CONFIGS or not configs:
            continue
        print(f'== {family} variants vs. {base}: problem-level paired comparison (F1) ==')
        print(f'{"config":<14}{"mean diff":>10}{"95% CI (bootstrap over problems)":>36}{"w/t/l":>10}{"Wilcoxon p":>12}'
              f'{"Holm p":>9}   problems significantly better/worse (corrected t, p<0.05)')
        diffs = {c: np.array([mean('f1', c, p) - mean('f1', base, p) for p in problems]) for c in configs}
        raw = [wilcoxon(d).pvalue if np.any(d != 0) else 1.0 for d in diffs.values()]
        adjusted = holm(np.array(raw))
        for (c, d), p_raw, p_adj in zip(diffs.items(), raw, adjusted):
            boot = [rng.choice(d, len(d)).mean() for _ in range(10000)]
            better = worse = 0
            for problem in problems:
                folds = sorted(set(per['f1'][c][problem]) & set(per['f1'][base][problem]))
                fold_diffs = [per['f1'][c][problem][f] - per['f1'][base][problem][f] for f in folds]
                if corrected_resampled_t(fold_diffs, 1 / FOLDS) < 0.05:
                    better += np.mean(fold_diffs) > 0
                    worse += np.mean(fold_diffs) < 0
            print(f'{c:<14}{d.mean():>+10.4f}{f"[{np.percentile(boot, 2.5):+.4f}, {np.percentile(boot, 97.5):+.4f}]":>36}'
                  f'{f"{(d > 0).sum()}/{(d == 0).sum()}/{(d < 0).sum()}":>10}{p_raw:>12.4f}{p_adj:>9.4f}'
                  f'   {better}/{worse}')
        print()

    length_aware_report(per, problems, rng)

    for name in ('rules', 'rules_t', 'best_tree', 'best_tree_t', 'pruned', 'pruned_t'):
        if not any(r['config'] == name for r in rows):
            continue
        share = np.mean([r['fell_back'] for r in rows if r['config'] == name])
        print(f'{name}: no rule reached min_precision and a single tree was returned in {share:.0%} of the folds')
    print()
    print('== chosen hyperparameters of the tuned configurations (how often each setting won) ==')
    for name in TUNED:
        counts = defaultdict(int)
        for r in rows:
            if r['config'] == name:
                counts[json.dumps(r['params'], sort_keys=True)] += 1
        total = sum(counts.values())
        print(f'{name}: ' + '; '.join(f'{k} x{v / total:.0%}' for k, v in sorted(counts.items(), key=lambda kv: -kv[1])[:4]))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['run', 'report'])
    ap.add_argument('--repeats', type=int, default=4)
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--out', action='append', help='results file; `report` accepts several and merges them')
    ap.add_argument('--configs', help='run: comma separated configurations to run (default all)')
    a = ap.parse_args()
    a.out = a.out or ['tdl_forest_results.jsonl']
    if a.cmd == 'run':
        assert len(a.out) == 1, 'run writes to one file'
        a.out = a.out[0]
    {'run': cmd_run, 'report': cmd_report}[a.cmd](a)
