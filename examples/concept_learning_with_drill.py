from ontolearn import KnowledgeBase, LengthBasedRefinement, SearchTreePriorityQueue, AbstractScorer
from ontolearn.rl import DQLTrainer
from ontolearn.metrics import F1
import json
import numpy as np


with open('synthetic_problems.json') as json_file:
    settings = json.load(json_file)

kb = KnowledgeBase(path=settings['data_path'])


class SampleReward(AbstractScorer):
    def __init__(self, pos=None, neg=None, unlabelled=None):
        super().__init__(pos, neg, unlabelled)
        self.name = 'F1'
        self.beta = 0
        self.noise = 0

        self.alpha = 2.0 # magnifies the importance of being better than previous state.
        self.reward_of_goal = 10.0

    def score(self, pos, neg, instances):
        self.pos = pos
        self.neg = neg

        tp = len(self.pos.intersection(instances))
        tn = len(self.neg.difference(instances))

        fp = len(self.neg.intersection(instances))
        fn = len(self.pos.difference(instances))
        try:
            recall = tp / (tp + fn)
            precision = tp / (tp + fp)
            f_1 = 2 * ((precision * recall) / (precision + recall))
        except ZeroDivisionError:
            f_1 = 0

        return round(f_1, 5)

    def apply(self, node):
        self.applied += 1

        instances = node.concept.instances
        if len(instances) == 0:
            node.quality = 0
            return False

        tp = len(self.pos.intersection(instances))
        # tn = len(self.neg.difference(instances))

        fp = len(self.neg.intersection(instances))
        fn = len(self.pos.difference(instances))

        try:
            recall = tp / (tp + fn)
        except ZeroDivisionError:
            node.quality = 0
            return False

        try:
            precision = tp / (tp + fp)
        except ZeroDivisionError:
            node.quality = 0
            return False

        if precision == 0 or recall == 0:
            node.quality = 0
            return False

        f_1 = 2 * ((precision * recall) / (precision + recall))
        node.quality = round(f_1, 5)

        assert node.quality

    def calculate(self, current_state, next_state=None):
        self.apply(current_state)
        self.apply(next_state)

        if next_state.quality == 1.0:
            return self.reward_of_goal
        discrepancy = next_state.quality - current_state.quality
        if discrepancy > 0.0:
            return discrepancy*self.alpha
        return 0


trainer = DQLTrainer(
    knowledge_base=kb,
    refinement_operator=LengthBasedRefinement(kb=kb),
    quality_func=F1(),
    reward_func=SampleReward(),
    search_tree=SearchTreePriorityQueue(),
    train_data=settings['problems'])
trainer.start()
