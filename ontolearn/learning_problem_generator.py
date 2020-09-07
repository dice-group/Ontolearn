import random


class LearningProblemGenerator:
    """
    Learning problem generator.
    """

    def __init__(self, knowledge_base, refinement_operator, num_problems=10, depth=2, min_length=2):
        self.kb = knowledge_base
        self.rho = refinement_operator
        self.num_problems = num_problems
        self.depth = depth
        self.min_length = min_length

    def apply_rho(self, node):
        refinements= [self.rho.getNode(i, parent_node=node) for i in self.rho.refine(node, maxlength=len(node) + self.min_length)]
        return random.sample(refinements, 1)[0]

    def apply(self):
        root = self.rho.getNode(self.kb.thing, root=True)
        current_state = root
        path = [current_state]
        for _ in range(self.depth):
            try:
                current_state = self.apply_rho(current_state)
            except ValueError:
                print('Dead End. No refinement found under the constraints provided by refinement operator.')
                return path
            path.append(current_state)
        return path

    def __iter__(self):
        for _ in range(self.num_problems):
            yield self.apply()
