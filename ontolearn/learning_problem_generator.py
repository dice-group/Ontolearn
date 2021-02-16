import sys
from queue import PriorityQueue
from typing import Generator, Literal, Optional, Iterable, Callable, Set, Tuple

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
            refinement_operator = LengthBasedRefinement(kb=knowledge_base)

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

    def owlready_individuals_to_string_balanced_examples(self, instances) -> Dict[str, Set]:

        string_all_pos = set(self.kb.convert_owlready2_individuals_to_uri_from_iterable(instances))

        string_all_neg = set(
            self.kb.convert_owlready2_individuals_to_uri_from_iterable(self.kb.individuals.difference(instances)))
        assert len(string_all_pos) >= self.min_num_instances

        string_balanced_pos, string_balanced_neg = balanced_sets(string_all_pos, string_all_neg)
        assert len(string_balanced_neg) >= self.min_num_instances

        return {'string_balanced_pos': string_balanced_pos, 'string_balanced_neg': string_balanced_neg}

    def owlready_individuals_to_string_balanced_n_samples(self, n: int, instances: set) -> Generator:
        """
        Generate n number of balanced negative and positive examples.
        To balance examples, randomly sample positive or negative examples.

        @param n: in
        @param instances: a set of owlready2 instances
        @return:
        """

        string_all_pos = set(self.kb.convert_owlready2_individuals_to_uri_from_iterable(instances))

        string_all_neg = set(
            self.kb.convert_owlready2_individuals_to_uri_from_iterable(self.kb.individuals.difference(instances)))
        for i in range(n):
            string_balanced_pos, string_balanced_neg = balanced_sets(string_all_pos, string_all_neg)
            assert len(string_balanced_pos) >= self.min_num_instances
            assert len(string_balanced_neg) >= self.min_num_instances

            yield {'string_balanced_pos': string_balanced_pos, 'string_balanced_neg': string_balanced_neg}

    def get_balanced_n_samples_per_examples(self, *, n=5, min_num_problems=None, max_length=None, min_length=None,
                                            num_diff_runs=None, min_num_instances=None,
                                            search_algo='strict-dfs') -> list:
        """
        1. We generate min_num_problems number of concepts
        2. For each concept, we generate n number of positive and negative examples
        3. Each example contains
        @param n:
        @param min_num_problems:
        @param max_length:
        @param min_length:
        @param num_diff_runs:
        @param min_num_instances:
        @param search_algo:
        @return:
        """

        def concept_sanity_check(x):
            try:
                assert len(x.concept.instances)
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

            for d in self.owlready_individuals_to_string_balanced_n_samples(n, example_node.concept.instances):
                assert len(d['string_balanced_pos']) == len(d['string_balanced_neg'])
                res.append((example_node.concept.str, d['string_balanced_pos'], d['string_balanced_neg']))

        try:
            assert len(gen_examples) > 0
        except AssertionError:
            print('*****No examples are created. Please update the configurations for learning problem generator****')
            exit(1)

        stats = np.array([[len(x), len(x.concept.instances)] for x in gen_examples])

        print(f'\nNumber of generated concepts:{len(gen_examples)}')
        print(f'Number of generated learning problems via sampling: {len(res)}')
        print(
            f'Average length of generated concepts:{stats[:, 0].mean():.3f}\nAverage number of individuals belong to a generated example:{stats[:, 1].mean():.3f}\n')
        return res

    def get_balanced_examples(self, *, min_num_problems=None, max_length=None, min_length=None,
                              num_diff_runs=None, min_num_instances=None, search_algo: SearchAlgos = 'strict-dfs') \
            -> list:
        """
        (1) Generate valid examples with input search algorithm.
        (2) Balance valid examples.

        @param min_num_problems:
        @param max_length:
        @param min_length:
        @param num_diff_runs:
        @param min_num_instances:
        @param search_algo: 'dfs' or 'strict-'dfs=> strict-dfs considers num_problems as a hard constrain.
        @return: A list of tuples (s,p,n) where s denotes the string representation of a concept,
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
            d = self.owlready_individuals_to_string_balanced_examples(example_node.concept.instances)
            res.append((example_node.concept.str, d['string_balanced_pos'], d['string_balanced_neg']))
            gen_examples.append(example_node)

        output_sanity_check(res)
        stats = np.array([[len(x), len(x.concept.instances)] for x in gen_examples])
        print(
            f'Average length of generated examples {stats[:, 0].mean():.3f}\nAverage number of individuals belong to a generated example {stats[:, 1].mean():.3f}')
        return res

    def get_examples(self, *, num_problems=None, max_length=None, min_length=None,
                     num_diff_runs=None, min_num_ind=None, search_algo=None) -> list:
        """
        (1) Get valid examples with input search algorithm.

        @param num_problems:
        @param max_length:
        @param min_length:
        @param num_diff_runs:
        @param min_num_ind:
        @param search_algo: 'dfs' or 'strict-'dfs=> strict-dfs considers num_problems as hard constriant.
        @return: A list of tuples (s,p,n) where s denotes the string representation of a concept,
        p and n denote a set of URIs of individuals indicating positive and negative examples.

        """
        res = []
        for example_node in self.generate_examples(num_problems=num_problems,
                                                   max_length=max_length, min_length=min_length,
                                                   num_diff_runs=num_diff_runs, min_num_instances=min_num_ind,
                                                   search_algo=search_algo):
            try:
                assert len(example_node.concept.instances)
            except AssertionError as e:
                print(e)
                print(f'{example_node}\nDoes not contain any instances. No instance to balance. Exiting.')
                exit(1)
            string_all_pos = set(
                self.kb.convert_owlready2_individuals_to_uri_from_iterable(example_node.concept.instances))
            string_all_neg = set(self.kb.convert_owlready2_individuals_to_uri_from_iterable(
                self.kb.individuals.difference(example_node.concept.instances)))
            res.append((example_node.concept.str, string_all_pos, string_all_neg))
        return res

    def get_concepts(self, *, num_problems=None, max_length=None, min_length=None,
                     max_num_instances=None, num_diff_runs=None, min_num_instances=None, search_algo=None) -> Generator:
        """
        @param max_num_instances:
        @param num_problems:
        @param max_length:
        @param min_length:
        @param num_diff_runs:
        @param min_num_instances:
        @param search_algo: 'dfs' or 'strict-'dfs=> strict-dfs considers num_problems as hard constriant.
        @return: A list of tuples (s,p,n) where s denotes the string representation of a concept,
        p and n denote a set of URIs of individuals indicating positive and negative examples.
        """
        return self.generate_examples(num_problems=num_problems,
                                      max_length=max_length, min_length=min_length,
                                      num_diff_runs=num_diff_runs,
                                      max_num_instances=max_num_instances,
                                      min_num_instances=min_num_instances,
                                      search_algo=search_algo)

    def generate_examples(self, *, num_problems=None, max_length=None, min_length=None,
                          num_diff_runs=None, max_num_instances=None, min_num_instances=None,
                          search_algo=None) -> Generator:
        """
        Generate examples via search algorithm that are valid examples w.r.t. given constraints

        @param num_diff_runs:
        @param num_problems:
        @param max_length:
        @param min_length:
        @param min_num_instances:
        @param max_num_instances:
        @param search_algo:
        @return:
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
            return self._apply_dfs()
        elif self.search_algo == 'strict-dfs':
            return self._apply_dfs(strict=True)

        else:
            print(f'Invalid input: search_algo:{search_algo} must be in [dfs,strict-dfs]')
            raise ValueError

    def _apply_dfs(self, strict=False) -> Iterable[Node]:
        """Apply depth first search with backtracking to generate concepts.
        """

        def define_constrain():
            if self.min_num_instances:
                def f1(x):
                    a = self.max_length >= self.kb.cl(x.concept) >= self.min_length
                    b = self.max_num_instances >= self.kb.individuals_count(x.concept) >= self.min_num_instances
                    return a and b

                return f1
            else:
                def f2(x):
                    return self.max_length >= self.kb.cl(x.concept) >= self.min_length

                return f2

        refinements = iter(self.apply_rho(Node(self.kb.thing, root=True), len_constant=3))

        constrain_func = define_constrain()

        valid_states_gate: Set[Node] = set()
        while True:
            try:
                state: Node = next(refinements)
            except StopIteration:
                print('All top concepts are refined.')
                break

            if state.individuals_count is None:
                state.individuals_count = self.kb.individuals_count(state.concept)

            if constrain_func(state):
                valid_states_gate.add(state)
                yield state
                if strict:
                    if len(valid_states_gate) >= self.num_problems * self.num_diff_runs:
                        break

            temp_gate: Set[Node] = set()
            for v in self._apply_dfs_on_state(state=state,
                                              kb=self.kb,
                                              apply_rho=self.apply_rho,
                                              constrain_func=constrain_func,
                                              depth=self.depth,
                                              patience_per_depth=(self.num_problems // 2)):
                if v not in valid_states_gate:
                    valid_states_gate.add(v)
                    temp_gate.add(v)
                    yield v
                    if strict:
                        if len(temp_gate) >= self.num_problems or (
                                len(valid_states_gate) >= self.num_problems * self.num_diff_runs):
                            break
            if strict:
                if len(valid_states_gate) >= self.num_problems * self.num_diff_runs:
                    break

        # sanity checking after the search.
        try:
            assert len(valid_states_gate) >= self.num_diff_runs * self.num_problems
        except AssertionError:
            print(f'Number of valid concepts generated:{len(valid_states_gate)}.\n'
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
                            apply_rho: Callable[..., Iterable[Node]],
                            constrain_func: Callable[[Node], bool],
                            patience_per_depth: int) -> Iterable[Node]:
        """

        Args:
            state:
            depth:
            kb: the knowledge base
            apply_rho: Function that takes a Node and a len_constant and refines the Node to new nodes
            constrain_func: Function that includes a refinement only if true
            patience_per_depth:

        Returns:
            ?
        """
        valid_examples = set()
        q: PriorityQueue[Tuple[int, LengthOrderedNode]] = PriorityQueue()
        for _ in range(depth):
            temp_patience = patience_per_depth  # patience for valid exam. per depth.
            temp_not_valid_patience = patience_per_depth  # patience for not valid exam. per depth.
            for i in apply_rho(state, len_constant=2):
                if i.individuals_count is None:
                    i.individuals_count = kb.individuals_count(i.concept)
                if constrain_func(i):  # validity checking.
                    # q.put((len(i), i))  # lower the length, higher priority.
                    if i not in valid_examples:
                        valid_examples.add(i)
                        concept_len = kb.cl(i.concept)
                        q.put((concept_len, LengthOrderedNode(i, concept_len)))  # lower the length, higher priority.
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

    def apply_rho(self, node, len_constant=1) -> Iterable[Node]:
        for i in self.operator.refine(node.concept,
                                      max_length=self.operator.len(
                                     node.concept) + len_constant if self.operator.len(
                                     node.concept) < self.max_length else self.operator.len(
                                     node.concept)):
            yield Node(i, parent_node=node)
