import random
from typing import List, Any, Generator
import asyncio
from .refinement_operators import LengthBasedRefinement, ModifiedCELOERefinement
from queue import PriorityQueue
import itertools
import sys
from .util import balanced_sets, performance_debugger
from collections import deque


class LearningProblemGenerator:
    """ Learning problem generator. """

    def __init__(self, knowledge_base, refinement_operator=None, num_problems=10, num_diff_runs=None,
                 min_num_instances=None, max_num_instances=sys.maxsize, min_length=3, max_length=5, depth=None,
                 search_algo='strict-dfs'):
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
        self.rho = refinement_operator
        self.search_algo = search_algo
        self.min_num_instances = min_num_instances
        self.max_num_instances = max_num_instances
        self.min_length = min_length
        self.max_length = max_length
        self.valid_learning_problems = []
        if depth is None:
            self.depth = 1 + (max_length // 3)
        else:
            self.depth = depth
        if num_diff_runs is None:
            self.num_diff_runs = int((num_problems * .01)) + 1  # quite expensive.
        else:
            self.num_diff_runs = num_diff_runs
        self.num_problems = num_problems // self.num_diff_runs

    def get_balanced_examples(self, *, num_problems=None, max_length=None, min_length=None,
                              num_diff_runs=None, min_num_instances=None, search_algo='strict-dfs') -> list:
        """
        (1) Generate valid examples with input search algorithm.
        (2) Balance valid examples.

        @param num_problems:
        @param max_length:
        @param min_length:
        @param num_diff_runs:
        @param min_num_instances:
        @param search_algo: 'dfs' or 'strict-'dfs=> strict-dfs considers num_problems as hard constriant.
        @return: A list of tuples (s,p,n) where s denotes the string representation of a concept,
        p and n denote a set of URIs of individuals indicating positive and negative examples.

        """

        assert self.min_num_instances or min_num_instances

        res = []
        for example_node in self.generate_examples(num_problems=num_problems, max_length=max_length,
                                                   min_length=min_length, num_diff_runs=num_diff_runs,
                                                   min_num_instances=min_num_instances, search_algo=search_algo):
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
            assert len(string_all_pos) >= self.min_num_instances
            # create balanced setting
            string_balanced_pos, string_balanced_neg = balanced_sets(string_all_pos, string_all_neg)
            assert len(string_balanced_neg) >= self.min_num_instances
            res.append((example_node.concept.str, string_balanced_pos, string_balanced_neg))
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
                     num_diff_runs=None, min_num_ind=None, search_algo=None) -> set:
        """
        @param num_problems:
        @param max_length:
        @param min_length:
        @param num_diff_runs:
        @param min_num_ind:
        @param search_algo: 'dfs' or 'strict-'dfs=> strict-dfs considers num_problems as hard constriant.
        @return: A list of tuples (s,p,n) where s denotes the string representation of a concept,
        p and n denote a set of URIs of individuals indicating positive and negative examples.
        """
        return self.generate_examples(num_problems=num_problems,
                                      max_length=max_length, min_length=min_length,
                                      num_diff_runs=num_diff_runs, min_num_instances=min_num_ind,
                                      search_algo=search_algo)

    @performance_debugger('generate_examples')
    def generate_examples(self, *, num_problems=None, max_length=None, min_length=None,
                          num_diff_runs=None, min_num_instances=None, max_num_instances=None, search_algo=None):
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
            assert isinstance(num_problems, int)
            self.max_length = max_length
        if min_length:
            assert isinstance(num_problems, int)
            self.min_length = min_length
        if min_num_instances:
            assert isinstance(num_problems, int)
            self.min_num_instances = min_num_instances
        if max_num_instances:
            assert isinstance(num_problems, int)
            self.max_num_instances = max_num_instances
        if search_algo:
            self.search_algo = search_algo

        if self.min_num_instances and self.max_num_instances == sys.maxsize:
            assert isinstance(self.min_num_instances, int)
            self.max_num_instances = len(self.kb.thing.instances) - self.min_num_instances

        if self.search_algo == 'dfs':
            return self._apply_dfs()
        elif self.search_algo == 'strict-dfs':
            return self._apply_dfs(strict=True)
        else:
            print(f'Invalid input: search_algo:{search_algo} must be in [dfs,strict-dfs]')
            raise ValueError

    def _apply_dfs(self, strict=False):
        """
        Apply depth first search with backtracking to generate concepts.

        @return:
        """
        # Generate all length 1 concepts.
        refinements = self.apply_rho(self.rho.getNode(self.kb.thing, root=True), len_constant=0)
        random.shuffle(refinements)

        if self.min_num_instances:
            def constrain_func(x):
                return (self.max_length >= len(x) >= self.min_length) and (
                        self.max_num_instances >= len(x.concept.instances) >= self.min_num_instances)
        else:
            def constrain_func(x):
                return self.max_length >= len(x) >= self.min_length

        valid_states_gate = {_ for _ in refinements if constrain_func(_)}

        counter = self.num_diff_runs
        while True:
            state = refinements.pop()
            for v in self._apply_dfs_on_state(state=state,
                                              apply_rho=self.apply_rho,
                                              depth=self.depth,
                                              num_problems=self.num_problems,
                                              constrain_func=constrain_func,
                                              strict=strict):
                valid_states_gate.add(v)
                if len(valid_states_gate) == self.num_problems:
                    break
            counter -= 1
            if counter == 0:
                break
        # sanity checking after the search.
        try:
            assert len(valid_states_gate) >= self.num_diff_runs * self.num_problems
        except AssertionError:
            print(f'Number of valid concepts generated:{len(valid_states_gate)}.\n'
                  f'Required number of concepts: {self.num_diff_runs * self.num_problems}.\n'
                  f'Increase the max length (Currently {self.max_length}) in order to generate more valid concepts..')

        return valid_states_gate

    @staticmethod
    # @performance_debugger('_apply_dfs_on_state')
    def _apply_dfs_on_state(state, depth, apply_rho, num_problems, constrain_func=None, strict=None) -> set:
        """

        @param state:
        @param depth:
        @param apply_rho:
        @param num_problems:
        @param max_length:
        @param min_length:
        @param min_num_ind:
        @param max_num_ind:
        @return:
        """
        valid_examples = set()
        q = PriorityQueue(maxsize=1_000_000)
        for _ in range(depth):
            for i in apply_rho(state, len_constant=2):
                q.put((-len(i), i))  # higher length, higher priority.
                if constrain_func(i):
                    if i not in valid_examples:
                        valid_examples.add(i)
                else:
                    """ ignore concept"""
                if strict:
                    if len(valid_examples) >= num_problems:
                        return valid_examples

            if not q.empty():
                _, state = q.get()
            else:
                print('queue is empty. Break condition.')
                return valid_examples
            if strict:
                if len(valid_examples) >= num_problems:
                    return valid_examples

        return valid_examples

    def apply_rho(self, node, len_constant=1):
        return [self.rho.getNode(i, parent_node=node) for i in self.rho.refine(node,
                                                                               maxlength=len(
                                                                                   node) + len_constant if len(
                                                                                   node) < self.max_length else len(
                                                                                   node))]

    """
    CD:Refactored.
    @property
    def concepts(self) -> List:

        Return list of positive and negative examples from valid learning problems.

        if len(self.valid_learning_problems) == 0:
            self.apply()
        return self.valid_learning_problems

    @property
    def balanced_examples(self) -> List:
        if len(self.valid_learning_problems) == 0:
            self.apply()
        results = list()
        counter = 0
        for example_node in self.valid_learning_problems:
            if counter == self.num_problems:
                break
            string_all_pos = set(
                self.kb.convert_owlready2_individuals_to_uri_from_iterable(example_node.concept.instances))
            string_all_neg = set(self.kb.convert_owlready2_individuals_to_uri_from_iterable(
                self.kb.individuals.difference(example_node.concept.instances)))
            # create balanced setting
            string_balanced_pos, string_balanced_neg = balanced_sets(string_all_pos, string_all_neg)
            if len(string_balanced_pos) > 0 and len(string_balanced_neg) > 0:
                results.append((example_node.concept.str, string_balanced_pos, string_balanced_neg))
                counter += 1

        return results
    @property
    def examples(self) -> List:
        if len(self.valid_learning_problems) == 0:
            self.apply()
        return self.valid_learning_problems
    """

    """
        CD:Due to controversy is ignored.
    
    
     @performance_debugger('DFSGeneration')
    def dfs_concept_generation(self) -> None:
        valid_learning_problems = set()
        for _ in range(self.num_problems):
            # Generate all length 1 concepts.
            refinements = list(self.apply_rho(self.rho.getNode(self.kb.thing, root=True), len_constant=0))
            random.shuffle(refinements)
            state = refinements.pop()
            temp = self._apply_dfs_on_state(state=state, apply_rho=self.apply_rho,
                                            depth=self.depth, num_problems=self.num_problems,
                                            min_num_ind=self.min_num_ind,
                                            max_length=self.max_length, min_length=self.min_length)
            valid_learning_problems.update(temp)
            if len(valid_learning_problems) > self.num_problems:
                break
        self.valid_learning_problems += list(valid_learning_problems)  # [:self.num_problems]
        
    @staticmethod
    async def async_apply_dfs(*, state, apply_rho, depth, num_problems, max_length, min_length) -> list:
        valid_examples = set()
        explored_state = deque()
        for i in range(depth):
            refinements = apply_rho(state, len_constant=3)
            if len(refinements) > 0:
                # Only constraints.
                valid_refs = set(filter(lambda x: max_length >= len(x) >= min_length, refinements))
                # Early select valid examples.
                valid_examples.update(valid_refs)
                if len(valid_examples) >= num_problems:
                    break
                state = refinements.pop()
                explored_state.extend(refinements)  # Append explored state
            else:
                state = explored_state.pop()  # Backtrack.
        return list(valid_examples)


    def concurrent_dfs_concept_generation(self) -> None:
        async def async_dfs(X):
            c = [self.async_apply_dfs(state=x, apply_rho=self.apply_rho,
                                      depth=self.depth, num_problems=self.num_problems // self.num_of_concurrent_search,
                                      max_length=self.max_length,
                                      min_length=self.min_length) for x in X]
            return await asyncio.gather(*c)

        refinements = list(self.apply_rho(self.rho.getNode(self.kb.thing, root=True), len_constant=0))
        random.shuffle(refinements)
        loop = asyncio.get_event_loop()
        results = loop.run_until_complete(async_dfs(refinements[:self.num_of_concurrent_search]))
        valid_examples = set(list(itertools.chain.from_iterable(results)))
        self.valid_learning_problems += list(valid_examples)
 
    def apply(self) -> None:
        if self.num_of_concurrent_search:
            print('Async will not be used.')
            self.dfs_concept_generation()
        else:
            self.dfs_concept_generation()
    
    def __iter__(self):
        if len(self.valid_learning_problems) == 0:
            self.apply()
        for c in self.valid_learning_problems:
            yield c

    def __len__(self):
        return len(self.valid_learning_problems)
    """
