import random
from typing import List, Any

from .refinement_operators import LengthBasedRefinement

import sys
from .util import balanced_sets


class LearningProblemGenerator:
    """
    Learning problem generator.
    """

    def __init__(self, knowledge_base, refinement_operator=None, num_problems=sys.maxsize, depth=2,
                 min_num_ind=15, min_length=0, max_length=10):
        if refinement_operator is None:
            refinement_operator = LengthBasedRefinement(kb=knowledge_base)

        self.kb = knowledge_base
        self.rho = refinement_operator
        self.num_problems = num_problems

        self.min_num_ind = min_num_ind
        self.min_length = min_length
        self.max_length = max_length
        self.valid_learning_problems = set()
        self.seen_concepts = set()
        self.num_apply_rho = depth

    @property
    def concepts(self):
        if len(self.valid_learning_problems) == 0:
            self.apply()
        return self.valid_learning_problems[:self.num_problems]

    @property
    def balanced_examples(self):
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
    def examples(self):
        if len(self.valid_learning_problems) == 0:
            self.apply()
        return self.valid_learning_problems[:self.num_problems]

    def apply_rho(self, node):
        return {self.rho.getNode(i, parent_node=node) for i in
                self.rho.refine(node, maxlength=len(node) + 3 if len(node) < self.max_length else len(node))}

    def apply(self):
        """
        Generate concepts that satisfy the given constraints.
        @return:
        """
        current_state = self.rho.getNode(self.kb.thing, root=True)
        for _ in range(self.num_apply_rho):
            refs = self.apply_rho(current_state)
            for i in refs:
                if self.min_length <= len(i) <= self.max_length and \
                        (i not in self.valid_learning_problems) and \
                        self.min_num_ind < len(i.concept.instances) < (len(self.kb.thing.instances) - self.min_num_ind):
                    self.valid_learning_problems.add(i)
            if len(self.valid_learning_problems) > self.num_problems:
                break
            if refs:
                current_state = random.sample(list(refs), 1)[0]  # random sample.

        print('|Concepts generated|', len(self.valid_learning_problems))
        self.valid_learning_problems = list(self.valid_learning_problems)

        self.valid_learning_problems = sorted(self.valid_learning_problems, key=lambda x: len(x), reverse=True)

    def __iter__(self):
        if len(self.valid_learning_problems) == 0:
            self.apply()
        for goal_node in self.valid_learning_problems[:self.num_problems]:
            yield goal_node

    def __len__(self):
        return len(self.valid_learning_problems)
