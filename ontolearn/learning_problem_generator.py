import random
from .refinement_operators import LengthBasedRefinement

import sys

class LearningProblemGenerator:
    """
    Learning problem generator.
    """

    def __init__(self, knowledge_base, refinement_operator=None, num_problems=sys.maxsize, depth=3,
                 min_length=0, max_length=5):
        if refinement_operator is None:
            refinement_operator = LengthBasedRefinement(kb=knowledge_base)

        self.kb = knowledge_base
        self.rho = refinement_operator
        self.num_problems = num_problems
        self.min_length = min_length
        self.max_length = max_length
        self.learning_problems_generated = set()
        self.seen_concepts = set()
        self.depth = depth

    def apply_rho(self, node):
        return {self.rho.getNode(i, parent_node=node) for i in
                self.rho.refine(node, maxlength=len(node) + 3 if len(node) < self.max_length else len(node))}

    def apply(self):
        root = self.rho.getNode(self.kb.thing, root=True)
        current_state = root
        for _ in range(self.depth):
            refs = self.apply_rho(current_state)
            self.seen_concepts.update(refs)
            if refs:
                current_state = random.sample(list(refs), 1)[0]  # random sample.
        valid_concepts = []
        for i in self.seen_concepts:
            if self.min_length <= len(i) <= self.max_length and (i not in self.learning_problems_generated):
                self.learning_problems_generated.add(i)
                valid_concepts.append(i)
        valid_concepts.sort(key=lambda x: len(x), reverse=False)
        print('Number of concepts generated {0}'.format(len(valid_concepts)))
        return valid_concepts[:self.num_problems]

    def __iter__(self):
        for goal_node in self.apply():
            yield goal_node
