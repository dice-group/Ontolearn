import sys
from queue import PriorityQueue
from typing import Literal, Optional, Iterable, Callable, Set, Tuple, Dict, List

import numpy as np

from owlapy.model import OWLClassExpression
from .abstracts import BaseRefinement
from .knowledge_base import KnowledgeBase
from .refinement_operators import LengthBasedRefinement
from .search import Node, LengthOrderedNode
from .utils import balanced_sets

SearchAlgos = Literal['dfs', 'strict-dfs']


class LearningProblemGenerator:
    """ Learning problem generator. """
    __slots__ = 'kb', 'operator', 'search_algo', 'min_num_instances', 'max_num_instances', 'min_length', 'max_length', \
                'depth', 'num_diff_runs', 'num_problems'

    kb: KnowledgeBase
    operator: BaseRefinement
    search_algo: SearchAlgos
    min_num_instances: Optional[int]
    max_num_instances: int
    min_length: int
    max_length: int
    depth: int
    num_diff_runs: int
    num_problems: int

    def __init__(self, knowledge_base: KnowledgeBase, refinement_operator: Optional[BaseRefinement] = None,
                 num_problems: int = 10_000, num_diff_runs: int = 100,
                 min_num_instances: Optional[int] = None, max_num_instances: int = sys.maxsize,
                 min_length: int = 3, max_length: int = 5, depth: int = 10,
                 search_algo: SearchAlgos = 'strict-dfs'):
        """
        Generate concepts via search algorithm to satisfy constraints.
         strict-dfs considers (min_length, max_length, min_num_ind, num_problems) as hard constraints.
         dfs- considers (min_length, max_length, min_num_ind) as hard constraints and soft (>=num_problems).

         Trade-off between num_diff_runs and num_problems.
        """
        if refinement_operator is None:
            refinement_operator = LengthBasedRefinement(knowledge_base=knowledge_base)

        if max_num_instances and min_num_instances:
            assert max_num_instances >= min_num_instances

        if max_length and min_length:
            assert max_length >= min_length
        self.kb = knowledge_base
        self.operator = refinement_operator
        self.search_algo = search_algo
        self.min_num_instances = min_num_instances
        self.max_num_instances = max_num_instances
        self.min_length = min_length
        self.max_length = max_length
        self.depth = depth
        self.num_diff_runs = num_diff_runs
        self.num_problems = num_problems // self.num_diff_runs

    def concept_individuals_to_string_balanced_examples(self, concept: OWLClassExpression) -> Dict[str, Set]:

        string_all_pos = self.kb.individuals_set(concept)

        string_all_neg = self.kb.all_individuals_set().difference(string_all_pos)
        assert len(string_all_pos) >= self.min_num_instances

        string_balanced_pos, string_balanced_neg = balanced_sets(set(string_all_pos), set(string_all_neg))
        assert len(string_balanced_neg) >= self.min_num_instances

        return {'string_balanced_pos': string_balanced_pos, 'string_balanced_neg': string_balanced_neg}

    def concept_individuals_to_string_balanced_n_samples(self, n: int, concept: OWLClassExpression) \
            -> Iterable[Dict[str, Set]]:
        """
        Generate n number of balanced negative and positive examples.
        To balance examples, randomly sample positive or negative examples.

        Args:
            n: in
            concept: an OWL Class Expression

        Returns:
            n dictionaries with positive and negative sets
        """

        string_all_pos = self.kb.individuals_set(concept)

        string_all_neg = self.kb.all_individuals_set().difference(string_all_pos)
        for i in range(n):
            string_balanced_pos, string_balanced_neg = balanced_sets(set(string_all_pos), set(string_all_neg))
            assert len(string_balanced_pos) >= self.min_num_instances
            assert len(string_balanced_neg) >= self.min_num_instances

            yield {'string_balanced_pos': string_balanced_pos, 'string_balanced_neg': string_balanced_neg}

    def get_balanced_n_samples_per_examples(self, *, n=5, min_num_problems=None, max_length=None, min_length=None,
                                            num_diff_runs=None, min_num_instances=None,
                                            search_algo: SearchAlgos = 'strict-dfs') \
            -> List[Tuple[OWLClassExpression, Set, Set]]:
        """
        1. We generate min_num_problems number of concepts
        2. For each concept, we generate n number of positive and negative examples
        3. Each example contains

        Args:
            n: ???
            min_num_problems:
            max_length:
            min_length:
            num_diff_runs:
            min_num_instances:
            search_algo:

        Returns:
            ???
        """

        def concept_sanity_check(x):
            try:
                assert self.kb.individuals_count(x.concept)
            except AssertionError:
                print(f'{x}\nDoes not contain any instances. No instance to balance. Exiting.')
                raise

        assert self.min_num_instances or min_num_instances

        res = []
        gen_examples = []
        for example_node in self.generate_examples(num_problems=min_num_problems, max_length=max_length,
                                                   min_length=min_length, num_diff_runs=num_diff_runs,
                                                   min_num_instances=min_num_instances, search_algo=search_algo):
            concept_sanity_check(example_node)
            gen_examples.append(example_node)

            for d in self.concept_individuals_to_string_balanced_n_samples(n, example_node.concept):
                assert len(d['string_balanced_pos']) == len(d['string_balanced_neg'])
                res.append((example_node.concept, d['string_balanced_pos'], d['string_balanced_neg']))

        try:
            assert len(gen_examples) > 0
        except AssertionError:
            print('*****No examples are created. Please update the configurations for learning problem generator****')
            raise

        stats = np.array([[x.len, x.individuals_count] for x in gen_examples])

        print(f'\nNumber of generated concepts:{len(gen_examples)}')
        print(f'Number of generated learning problems via sampling: {len(res)}')
        print(
            f'Average length of generated concepts:{stats[:, 0].mean():.3f}\n'
            f'Average number of individuals belong to a generated example:{stats[:, 1].mean():.3f}\n')
        return res

    def get_balanced_examples(self, *, min_num_problems=None, max_length=None, min_length=None,
                              num_diff_runs=None, min_num_instances=None, search_algo: SearchAlgos = 'strict-dfs') \
            -> List[Tuple[OWLClassExpression, Set, Set]]:
        """
        (1) Generate valid examples with input search algorithm.
        (2) Balance valid examples.

        Args:
            min_num_problems:
            max_length:
            min_length:
            num_diff_runs:
            min_num_instances:
            search_algo: 'dfs' or 'strict-'dfs=> strict-dfs considers num_problems as a hard constrain.

        Returns:
            A list of tuples (s,p,n) where s denotes the string representation of a concept,
            p and n denote a set of URIs of individuals indicating positive and negative examples.
        """

        def output_sanity_check(y):
            try:
                assert len(y) >= min_num_problems
            except AssertionError:
                print('Not enough number of problems are generated')
                raise

        assert self.min_num_instances or min_num_instances

        res = []
        gen_examples = []
        for example_node in self.generate_examples(num_problems=min_num_problems, max_length=max_length,
                                                   min_length=min_length, num_diff_runs=num_diff_runs,
                                                   min_num_instances=min_num_instances, search_algo=search_algo):
            d = self.concept_individuals_to_string_balanced_examples(example_node.concept)
            res.append((example_node.concept, d['string_balanced_pos'], d['string_balanced_neg']))
            gen_examples.append(example_node)

        output_sanity_check(res)
        stats = np.array([[x.len, x.individuals_count] for x in gen_examples])
        print(
            f'Average length of generated examples {stats[:, 0].mean():.3f}\n'
            f'Average number of individuals belong to a generated example {stats[:, 1].mean():.3f}')
        return res

    def get_concepts(self, *, num_problems=None, max_length=None, min_length=None, max_num_instances=None,
                     num_diff_runs=None, min_num_instances=None, search_algo=None) -> Iterable[Node]:
        """

        Args:
            max_num_instances:
            num_problems:
            max_length:
            min_length:
            num_diff_runs:
            min_num_instances:
            search_algo: 'dfs' or 'strict-'dfs=> strict-dfs considers num_problems as hard constriant.

        Returns:
            A list of nodes with generated concepts inside
        """
        yield from self.generate_examples(num_problems=num_problems,
                                          max_length=max_length, min_length=min_length,
                                          num_diff_runs=num_diff_runs,
                                          max_num_instances=max_num_instances,
                                          min_num_instances=min_num_instances,
                                          search_algo=search_algo)

    def generate_examples(self, *, num_problems=None, max_length=None, min_length=None,
                          num_diff_runs=None, max_num_instances=None, min_num_instances=None,
                          search_algo=None) -> Iterable[Node]:
        """
        Generate examples via search algorithm that are valid examples w.r.t. given constraints

        Args:
            num_diff_runs:
            num_problems:
            max_length:
            min_length:
            min_num_instances:
            max_num_instances:
            search_algo:

        Returns:
            ???
        """
        if num_problems and num_diff_runs:
            assert isinstance(num_problems, int)
            assert isinstance(num_diff_runs, int)
            self.num_problems = num_problems
            self.num_diff_runs = num_diff_runs
            self.num_problems //= self.num_diff_runs
        elif num_diff_runs:
            assert isinstance(num_diff_runs, int)
            self.num_diff_runs = num_diff_runs
            self.num_problems //= self.num_diff_runs
        elif num_problems:
            assert isinstance(num_problems, int)
            self.num_problems = num_problems // self.num_diff_runs

        if max_length:
            assert isinstance(max_length, int)
            self.max_length = max_length
        if min_length:
            assert isinstance(min_length, int)
            self.min_length = min_length
        if min_num_instances:
            assert isinstance(min_num_instances, int)
            self.min_num_instances = min_num_instances
            # Do not generate concepts that do not define enough individuals.
            self.operator.min_num_instances = self.min_num_instances

        if max_num_instances:
            assert isinstance(max_num_instances, int)
            self.max_num_instances = max_num_instances
        if search_algo:
            self.search_algo = search_algo

        if self.min_num_instances and self.max_num_instances == sys.maxsize:
            assert isinstance(self.min_num_instances, int)
            self.max_num_instances = self.kb.individuals_count() - self.min_num_instances

        if self.search_algo == 'dfs':
            yield from self._apply_dfs()
        elif self.search_algo == 'strict-dfs':
            yield from self._apply_dfs(strict=True)

        else:
            print(f'Invalid input: search_algo:{search_algo} must be in [dfs,strict-dfs]')
            raise ValueError

    def _apply_dfs(self, strict=False) -> Iterable[Node]:
        """Apply depth first search with backtracking to generate concepts.
        """

        def define_constrain() -> Callable[[Node], bool]:
            if self.min_num_instances:
                def f1(x: Node) -> bool:
                    a = self.max_length >= x.len >= self.min_length
                    b = self.max_num_instances >= x.individuals_count >= self.min_num_instances
                    return a and b

                return f1
            else:
                def f2(x: Node) -> bool:
                    return self.max_length >= x.len >= self.min_length

                return f2

        refinements = iter(self.apply_refinement_operator(self.kb.thing, len_constant=3))

        constrain_func = define_constrain()

        valid_concepts_gate: Set[OWLClassExpression] = set()
        while True:
            try:
                concept: OWLClassExpression = next(refinements)
            except StopIteration:
                print('All top concepts are refined.')
                break

            state = Node(concept, self.kb.cl(concept))
            state.individuals_count = self.kb.individuals_count(concept)

            if constrain_func(state):
                valid_concepts_gate.add(concept)
                yield state
                if strict:
                    if len(valid_concepts_gate) >= self.num_problems * self.num_diff_runs:
                        break

            temp_gate: Set[OWLClassExpression] = set()
            for v in self._apply_dfs_on_state(state=state,
                                              kb=self.kb,
                                              refine_concept=self.apply_refinement_operator,
                                              constrain_func=constrain_func,
                                              depth=self.depth,
                                              patience_per_depth=(self.num_problems // 2)):
                if v.concept not in valid_concepts_gate:
                    valid_concepts_gate.add(v.concept)
                    temp_gate.add(v.concept)
                    yield v
                    if strict:
                        if len(temp_gate) >= self.num_problems or (
                                len(valid_concepts_gate) >= self.num_problems * self.num_diff_runs):
                            break
            if strict:
                if len(valid_concepts_gate) >= self.num_problems * self.num_diff_runs:
                    break

        # sanity checking after the search.
        try:
            assert len(valid_concepts_gate) >= self.num_diff_runs * self.num_problems
        except AssertionError:
            print(f'Number of valid concepts generated:{len(valid_concepts_gate)}.\n'
                  f'Required number of concepts: {self.num_diff_runs * self.num_problems}.\n'
                  f'Please update the given constraints:'
                  f'Increase the max length (Currently {self.max_length}).\n'
                  f'Increase the max number of instances for concepts (Currently {self.max_num_instances}).\n'
                  f'Decrease the min length (Currently {self.min_length}).\n'
                  f'Decrease the max number of instances for concepts (Currently {self.min_num_instances}).\n')

    @staticmethod
    # @performance_debugger('_apply_dfs_on_state')
    def _apply_dfs_on_state(state: Node,
                            depth: int,
                            kb: KnowledgeBase,
                            refine_concept: Callable[..., Iterable[OWLClassExpression]],
                            constrain_func: Callable[[Node], bool],
                            patience_per_depth: int) -> Iterable[Node]:
        """

        Args:
            state:
            depth:
            kb: the knowledge base
            refine_concept: Function that takes a concept and a len_constant and refines the concept to new concepts
            constrain_func: Function that includes a Node only if true
            patience_per_depth:

        Returns:
            ?
        """
        valid_examples = set()
        q: PriorityQueue[Tuple[int, LengthOrderedNode]] = PriorityQueue()
        for _ in range(depth):
            temp_patience = patience_per_depth  # patience for valid exam. per depth.
            temp_not_valid_patience = patience_per_depth  # patience for not valid exam. per depth.
            for c in refine_concept(state.concept, len_constant=2):
                i = Node(c, kb.cl(c))
                i.individuals_count = kb.individuals_count(c)
                if constrain_func(i):  # validity checking.
                    # q.put((len(i), i))  # lower the length, higher priority.
                    if i not in valid_examples:
                        valid_examples.add(i)
                        q.put((i.len, LengthOrderedNode(i, i.len)))  # lower the length, higher priority.
                        yield i
                        temp_patience -= 1
                        if temp_patience == 0:
                            break
                else:
                    # Heuristic if, too many of them not valid, do not continue.
                    temp_not_valid_patience -= 1
                    if temp_not_valid_patience == 0:
                        break

            if not q.empty():
                state = q.get()[1].node
            else:
                return None

    def apply_refinement_operator(self, concept: OWLClassExpression, len_constant=1) -> Iterable[OWLClassExpression]:
        for i in self.operator.refine(concept,
                                      max_length=self.operator.len(
                                          concept) + len_constant if self.operator.len(
                                          concept) < self.max_length else self.operator.len(
                                          concept)):
            yield i
