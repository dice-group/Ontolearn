from typing import Tuple
import numpy as np


class Data:
    def __init__(self, knowledge_base):
        self.kb = knowledge_base
        self.individuals = list(self.kb.T.instances)

        self.num_individuals = len(self.individuals)

        self.labels = None

        self.num_of_outputs = None

    def concepts_for_training(self, m):
        self.labels = np.array(m)

    def score(self, pos, neg):
        assert isinstance(pos, list)
        assert isinstance(neg, list)

        pos = set(pos)
        neg = set(neg)

        y = []
        for j in self.labels:  # training data corresponds to number of outputs.
            individuals = {self.individuals.index(j) for j in j.instances()}

            tp = len(pos.intersection(individuals))
            tn = len(neg.difference(individuals))

            fp = len(neg.intersection(individuals))
            fn = len(pos.difference(individuals))
            try:
                recall = tp / (tp + fn)
                precision = tp / (tp + fp)
                f_1 = 2 * ((precision * recall) / (precision + recall))
            except:
                f_1 = 0

            y.append(round(f_1, 5))
        return y

    def train(self, num):

        for i in self.labels:
            if len(i.instances()) < num or (len(i.instances()) >= len(self.kb.T.instances()) - num):
                continue

            all_positives = {self.individuals.index(_) for _ in i.instances()}

            all_negatives = {self.individuals.index(_) for _ in self.kb.T.instances() - i.instances()}

            return all_positives, all_negatives, i


class PriorityQueue:
    def __init__(self):
        from queue import PriorityQueue
        self.struct = PriorityQueue()
        self.items_in_queue = set()
        self.expressionTests = 0

    def __len__(self):
        return len(self.items_in_queue)

    def add_into_queue(self, t: Tuple):
        assert not isinstance(t[0], str)
        if not (t[1].ce.expression in self.items_in_queue):
            self.struct.put(t)
            self.items_in_queue.add(t[1].ce.expression)
            self.expressionTests += 1

    def get_from_queue(self):
        assert len(self.items_in_queue) > 0
        node_to_return = self.struct.get()[1]
        self.items_in_queue.remove(node_to_return.ce.expression)
        return node_to_return
