from abc import ABC, abstractmethod


class AbstractScorer(ABC):
    @abstractmethod
    def __init__(self, pos, neg):
        self.pos = pos
        self.neg = neg

    def apply(self, n):
        pass