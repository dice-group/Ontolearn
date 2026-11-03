"""Benchmark EvoLearner tournament vs. lexicase parent selection.

Each (problem, seed) uses one stratified 80/20 train/held-out split shared by all configurations (paired design).
Every run gets the same wall-clock budget, so cheaper-per-generation variants can use it for more generations.

    python examples/benchmark_evolearner_semantic.py run --seeds 10 --workers 16 --out results.jsonl
    python examples/benchmark_evolearner_semantic.py report --out results.jsonl
"""
import argparse
import json
import random
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

# name -> (selection, unused)
CONFIGS = {
    'baseline': ('tournament', None),
    'lexicase': ('lexicase', None),
}

# name -> (kb path, lps file, problem key, runtime budget in seconds)
PROBLEMS = {f'Family-{k}': ('KGs/Family/family-benchmark_rich_background.owl', 'examples/synthetic_problems.json', k, 20)
            for k in ('Aunt', 'Brother', 'Cousin', 'Granddaughter', 'Uncle', 'Grandgrandfather')}
PROBLEMS['Mutagenesis'] = ('KGs/Mutagenesis/mutagenesis.owl', 'LPs/Mutagenesis/lps.json', 'NotKnown', 60)
PROBLEMS['Carcinogenesis'] = ('KGs/Carcinogenesis/carcinogenesis.owl', 'LPs/Carcinogenesis/lps.json', 'NOTKNOWN', 60)

_KB = {}


def _f1(covered, pos, neg):
    tp = len(covered & pos)
    fp = len(covered & neg)
    fn = len(pos) - tp
    return 0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)


def run_one(task):
    problem, config, seed = task
    from owlapy.owl_individual import OWLNamedIndividual
    from ontolearn.knowledge_base import KnowledgeBase
    from ontolearn.learners import EvoLearner
    from ontolearn.learning_problem import PosNegLPStandard

    kb_path, lps_path, key, budget = PROBLEMS[problem]
    if kb_path not in _KB:
        _KB[kb_path] = KnowledgeBase(path=kb_path)
    kb = _KB[kb_path]
    with open(lps_path) as f:
        ex = json.load(f)['problems'][key]
    pos_all, neg_all = sorted(ex['positive_examples']), sorted(ex['negative_examples'])

    split_rng = random.Random(seed)  # the split depends on (seed) only, so it is shared across configurations
    def split(items):
        items = items[:]
        split_rng.shuffle(items)
        cut = max(1, int(round(0.8 * len(items))))
        return items[:cut], items[cut:]
    pos_tr, pos_te = split(pos_all)
    neg_tr, neg_te = split(neg_all)
    mk = lambda xs: {OWLNamedIndividual(x) for x in xs}  # noqa: E731
    lp = PosNegLPStandard(pos=mk(pos_tr), neg=mk(neg_tr))

    selection, algo_kwargs = CONFIGS[config]
    random.seed(seed * 1000 + 17)
    np.random.seed(seed)
    model = EvoLearner(knowledge_base=kb, selection=selection, max_runtime=budget,
                       num_generations=100000, terminate_on_goal=True)
    start = time.time()
    model.fit(lp)
    runtime = time.time() - start
    best = model.best_hypotheses(n=1)
    covered = set(kb.individuals_set(best))
    p_tr, n_tr = mk(pos_tr), mk(neg_tr)
    p_te, n_te = mk(pos_te), mk(neg_te)
    return dict(problem=problem, config=config, seed=seed, train_f1=_f1(covered, p_tr, n_tr),
                test_f1=_f1(covered, p_te, n_te), tested=model.number_of_tested_concepts, runtime=runtime,
                length=len(str(best)))


def cmd_run(args):
    tasks = [(p, c, s) for s in range(args.seeds) for p in PROBLEMS for c in CONFIGS]
    done = set()
    try:
        with open(args.out) as f:
            done = {(r['problem'], r['config'], r['seed']) for r in map(json.loads, f)}
    except FileNotFoundError:
        pass
    tasks = [t for t in tasks if t not in done]
    print(f'{len(tasks)} runs to do ({len(done)} already in {args.out})', flush=True)
    # ProcessPoolExecutor (non-daemonic workers): owlready2 spawns a child process while parsing an ontology
    with ProcessPoolExecutor(args.workers) as pool, open(args.out, 'a') as out:
        futures = [pool.submit(run_one, t) for t in tasks]
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            out.write(json.dumps(r) + '\n')
            out.flush()
            if i % 20 == 0:
                print(f'{i}/{len(tasks)}', flush=True)


def cmd_report(args):
    from scipy.stats import wilcoxon
    rows = [json.loads(line) for line in open(args.out)]
    by = defaultdict(lambda: defaultdict(dict))  # metric -> (problem, config) -> {seed: value}
    for r in rows:
        for m in ('train_f1', 'test_f1', 'tested', 'runtime'):
            by[m][(r['problem'], r['config'])][r['seed']] = r[m]
    problems = list(PROBLEMS)
    configs = list(CONFIGS)
    for metric in ('train_f1', 'test_f1', 'tested'):
        print(f'\n== {metric}: mean (sem) over seeds ==')
        print(f'{"problem":<24}' + ''.join(f'{c:>18}' for c in configs))
        for p in problems:
            cells = []
            for c in configs:
                v = list(by[metric][(p, c)].values())
                cells.append(f'{np.mean(v):.3f} ({np.std(v) / np.sqrt(len(v)):.3f})' if v else '-')
            print(f'{p:<24}' + ''.join(f'{x:>18}' for x in cells))
        means = {c: np.mean([np.mean(list(by[metric][(p, c)].values())) for p in problems]) for c in configs}
        print(f'{"MEAN":<24}' + ''.join(f'{means[c]:>18.3f}' for c in configs))
    print('\n== paired comparison vs baseline (all problems pooled over paired seeds) ==')
    for metric in ('train_f1', 'test_f1'):
        for c in configs[1:]:
            diffs = []
            for p in problems:
                a, b = by[metric][(p, c)], by[metric][(p, 'baseline')]
                diffs += [a[s] - b[s] for s in a if s in b]
            d = np.array(diffs)
            if len(d) and np.any(d != 0):
                pval = wilcoxon(d).pvalue
            else:
                pval = float('nan')
            print(f'{metric:<9} {c:<10} n={len(d):<4} mean diff={d.mean():+.4f}  win/tie/loss='
                  f'{(d > 0).sum()}/{(d == 0).sum()}/{(d < 0).sum()}  wilcoxon p={pval:.4f}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['run', 'report'])
    ap.add_argument('--seeds', type=int, default=10)
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--out', default='evolearner_semantic_results.jsonl')
    a = ap.parse_args()
    {'run': cmd_run, 'report': cmd_report}[a.cmd](a)
