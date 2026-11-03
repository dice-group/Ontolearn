"""EvoLearner(random_state=...): seeded runs repeat within a process, across processes and under threads."""
import json
import os
import random
import subprocess
import sys
import threading
from pathlib import Path

from deap import gp
from owlapy import owl_expression_to_dl
from owlapy.class_expression import OWLClass, OWLClassExpression
from owlapy.owl_individual import OWLNamedIndividual

from ontolearn.ea_initialization import EARandomInitialization, EARandomWalkInitialization
from ontolearn.ea_utils import cx_one_point_ordered
from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.learners import EvoLearner
from ontolearn.learning_problem import PosNegLPStandard

ROOT = Path(__file__).resolve().parents[1]
FAMILY = ROOT / 'KGs' / 'Family' / 'family-benchmark_rich_background.owl'
MUTAGENESIS = ROOT / 'KGs' / 'Mutagenesis' / 'mutagenesis.owl'
NS = 'http://www.benchmark.org/family#'

_KBS = {}


def _kb(path: Path) -> KnowledgeBase:
    if path not in _KBS:
        _KBS[path] = KnowledgeBase(path=str(path))
    return _KBS[path]


def _family_problem(name='Uncle'):
    with open(ROOT / 'examples' / 'synthetic_problems.json') as stream:
        examples = json.load(stream)['problems'][name]
    return PosNegLPStandard(pos={OWLNamedIndividual(i) for i in examples['positive_examples']},
                            neg={OWLNamedIndividual(i) for i in examples['negative_examples']})


def _mutagenesis_problem():
    namespace = 'http://dl-learner.org/mutagenesis#'
    pos = ['d190', 'd191', 'd194', 'd197', 'e1', 'e2', 'e27', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6']
    neg = ['d189', 'd192', 'd193', 'd195', 'd196', 'e10', 'e11', 'e12', 'e13', 'e14', 'e15', 'e16', 'e17', 'e18',
           'e19', 'e20', 'e21', 'e22', 'e23', 'e24', 'e25', 'e26', 'e3', 'e4', 'e5', 'e6', 'e7', 'e8', 'e9']
    return PosNegLPStandard(pos={OWLNamedIndividual(namespace + i) for i in pos},
                            neg={OWLNamedIndividual(namespace + i) for i in neg})


def _seeded_fit(random_state=7, dataset='family', init='random_walk', generations=4, problem='Uncle'):
    if dataset == 'family':
        path, lp = FAMILY, _family_problem(problem)
    else:
        path, lp = MUTAGENESIS, _mutagenesis_problem()
    init_method = EARandomWalkInitialization() if init == 'random_walk' else EARandomInitialization()
    model = EvoLearner(knowledge_base=_kb(path), random_state=random_state, population_size=60,
                       num_generations=generations, max_runtime=600, terminate_on_goal=False, init_method=init_method)
    return model.fit(lp)


def _fingerprint(model):
    nodes = model.best_hypotheses(n=3, return_node=True)
    return model.number_of_tested_concepts, [(owl_expression_to_dl(n.concept), n.quality) for n in nodes]


def _cross_process_fingerprints():
    """Fingerprints compared across processes (run in a subprocess, see the test below)."""
    return [
        _fingerprint(_seeded_fit(dataset='family', init='random_walk', generations=20, problem='Cousin')),
        _fingerprint(_seeded_fit(dataset='family', init='random', generations=20, problem='Cousin')),
        _fingerprint(_seeded_fit(dataset='mutagenesis', init='random_walk', generations=4)),
    ]


class TestRandomState:
    def test_same_seed_repeats_within_a_process(self):
        for init in ('random_walk', 'random'):
            assert _fingerprint(_seeded_fit(init=init)) == _fingerprint(_seeded_fit(init=init))

    def test_fit_restores_the_global_random_state(self):
        random.seed(123)
        expected = random.random()
        random.seed(123)
        _seeded_fit()
        assert random.random() == expected

    def test_same_seed_repeats_across_processes(self):
        # Separate processes with different hash seeds change set iteration order and object addresses, which
        # previously changed the random choices made during initialization and crossover. The Family runs use both
        # initialization methods over 20 generations; Mutagenesis adds data properties and their splits, whose
        # trees mix several node types (classes, numbers, data splits).
        script = ('import sys; sys.path.insert(0, sys.argv[1]); '
                  'from test_evolearner_reproducibility import _cross_process_fingerprints; '
                  'print(repr(_cross_process_fingerprints()))')
        outputs = set()
        for hash_seed in ('0', '1'):
            env = {**os.environ, 'PYTHONHASHSEED': hash_seed}
            result = subprocess.run([sys.executable, '-c', script, str(ROOT / 'tests')], env=env, cwd=ROOT,
                                    capture_output=True, text=True)
            assert result.returncode == 0, f'PYTHONHASHSEED={hash_seed} failed:\n{result.stderr}'
            outputs.add(result.stdout.strip().splitlines()[-1])
        assert len(outputs) == 1, outputs

    def test_concurrent_seeded_fits_match_sequential_fits(self):
        # Seeded fits seed the global `random` module; concurrent ones must not re-seed each other mid-run or
        # restore each other's state.
        seeds = (3, 4, 5)
        expected = {seed: _fingerprint(_seeded_fit(random_state=seed)) for seed in seeds}

        random.seed(99)
        untouched = random.getstate()
        results, errors = {}, []

        def run(seed):
            try:
                results[seed] = _fingerprint(_seeded_fit(random_state=seed))
            except Exception as exc:  # reported below
                errors.append((seed, exc))

        threads = [threading.Thread(target=run, args=(seed,)) for seed in seeds]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, errors
        assert results == expected
        assert random.getstate() == untouched


class TestOrderedCrossover:
    @staticmethod
    def _pset():
        pset = gp.PrimitiveSetTyped('concept', [], OWLClassExpression)
        pset.addPrimitive(lambda a, b: a, [OWLClassExpression, OWLClassExpression], OWLClassExpression, name='AND')
        pset.addPrimitive(lambda n, a: a, [int, OWLClassExpression], OWLClassExpression, name='MIN')
        for name in ('A', 'B', 'C'):
            pset.addTerminal(OWLClass(NS + name), OWLClassExpression, name=name)
        for i in (1, 2):
            pset.addTerminal(i, int, name=f'n{i}')
        return pset

    def test_matches_types_and_repeats_under_a_seed(self):
        pset = self._pset()
        make = lambda text: gp.PrimitiveTree.from_string(text, pset)  # noqa: E731
        results = []
        for _ in range(2):
            random.seed(5)
            pairs = [cx_one_point_ordered(make('AND(MIN(n1, A), B)'), make('MIN(n2, AND(B, C))')) for _ in range(20)]
            results.append([(str(a), str(b)) for a, b in pairs])
        assert results[0] == results[1]
        # Every child is still a well-typed tree: swaps only happen between subtrees of the same type.
        for a, b in results[0]:
            for text in (a, b):
                assert make(text).root.ret is OWLClassExpression

    def test_single_node_trees_are_unchanged(self):
        pset = self._pset()
        a, b = gp.PrimitiveTree.from_string('A', pset), gp.PrimitiveTree.from_string('B', pset)
        assert cx_one_point_ordered(a, b) == (a, b)
