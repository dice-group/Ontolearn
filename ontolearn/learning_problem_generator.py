import random
from typing import List, Any
import asyncio
from .refinement_operators import LengthBasedRefinement, ModifiedCELOERefinement
import itertools
import sys
from .util import balanced_sets, performance_debugger
from collections import deque


class LearningProblemGenerator:
    """ Learning problem generator. """

    def __init__(self, knowledge_base, refinement_operator=None, num_problems=10,
                 min_num_ind=0, min_length=3, max_length=5, num_of_concurrent_search=None):
        """
        Generate at least num_problems of concepts that has satisfy length constraints.
        """
        if refinement_operator is None:
            refinement_operator = LengthBasedRefinement(kb=knowledge_base)

        if num_of_concurrent_search:
            assert isinstance(num_of_concurrent_search, int)
        self.kb = knowledge_base
        self.rho = refinement_operator
        self.num_problems = num_problems
        self.num_of_concurrent_search = num_of_concurrent_search
        self.min_num_ind = min_num_ind
        self.min_length = min_length
        self.max_length = max_length
        self.valid_learning_problems = []
        self.depth = 1 + (max_length // 3)

    @property
    def concepts(self) -> List:
        """
        Return list of positive and negative examples from valid learning problems.
        """
        if len(self.valid_learning_problems) == 0:
            self.apply()
        return self.valid_learning_problems

    @property
    def balanced_examples(self) -> List:
        """
        Return list of balanced positive and negative examples from valid learning problems.
        """
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
            else:
                continue
        return results

    @property
    def examples(self) -> List:
        if len(self.valid_learning_problems) == 0:
            self.apply()
        return self.valid_learning_problems

    def apply_rho(self, node, len_constant=1):
        return {self.rho.getNode(i, parent_node=node) for i in
                self.rho.refine(node, maxlength=len(node) + len_constant if len(node) < self.max_length else len(node),
                                apply_combinations=False)}

    @performance_debugger('DFSGeneration')
    def dfs_concept_generation(self) -> None:
        """

        Given the constraints (number of required problems/class expressions, min and max length),
        search valid concepts in depth-first-search with backtracking manner.
        """
        valid_examples = set()
        for _ in range(self.num_problems):
            path = []
            explored_state = list()
            # Generate all length 1 concepts.
            refinements = list(self.apply_rho(self.rho.getNode(self.kb.thing, root=True), len_constant=0))
            random.shuffle(refinements)
            state = refinements.pop()
            explored_state.extend(refinements)

            for i in range(self.depth):
                path.append(state)  # Append state.
                refinements = self.apply_rho(state, len_constant=3)
                if len(refinements) > 0:
                    # Only constraints.
                    valid_refs = set(filter(lambda x: self.max_length >= len(x) >= self.min_length, refinements))
                    # Early select valid examples.
                    valid_examples.update(valid_refs)
                    if len(self.valid_learning_problems) >= self.num_problems:
                        break
                    state = refinements.pop()
                    explored_state.extend(refinements)  # Append explored state
                else:
                    state = explored_state.pop()  # Backtrack.
            if len(valid_examples) >= self.num_problems:
                break

        self.valid_learning_problems = list(valid_examples)#[:self.num_problems]

    @staticmethod
    async def apply_dfs(*, state, apply_rho, depth, num_problems, max_length, min_length) -> set:
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
        return valid_examples

    @performance_debugger('DFSGeneration')
    def concurrent_dfs_concept_generation(self) -> None:
        """

        Given the constraints (number of required problems/class expressions, min and max length),
        search valid concepts in depth-first-search with backtracking manner.
        """
        async def async_dfs(X):
            c = [self.apply_dfs(state=x, apply_rho=self.apply_rho,
                                depth=self.depth, num_problems=self.num_problems // self.num_of_concurrent_search,
                                max_length=self.max_length,
                                min_length=self.min_length) for x in X]
            return await asyncio.gather(*c)

        refinements = list(self.apply_rho(self.rho.getNode(self.kb.thing, root=True), len_constant=0))
        random.shuffle(refinements)
        loop = asyncio.get_event_loop()
        results = loop.run_until_complete(async_dfs(refinements[:self.num_of_concurrent_search]))
        valid_examples = set(list(itertools.chain.from_iterable(results)))
        self.valid_learning_problems = list(valid_examples)[:self.num_problems]

    def apply(self) -> None:
        """
        Generate concepts that satisfy the given constraints.
        """
        if self.num_of_concurrent_search:
            print('Concurrent dfs is not production ready. We will use serial computation.')
            self.dfs_concept_generation()
            # self.concurrent_dfs_concept_generation()
        else:
            self.dfs_concept_generation()

    def __iter__(self):
        if len(self.valid_learning_problems) == 0:
            self.apply()
        for goal_node in self.valid_learning_problems[:self.num_problems]:
            yield goal_node

    def __len__(self):
        return len(self.valid_learning_problems)
