from abc import ABC, abstractmethod


class AbstractRefinement(ABC):
    @abstractmethod
    def __init__(self, kb):
        self.kb = kb

    @abstractmethod
    def refine(self, concept):
        pass

    @abstractmethod
    def refine_atomic_concept(self, concept):
        pass

    @abstractmethod
    def refine_complement_of(self, concept):
        pass

    @abstractmethod
    def refine_object_some_values_from(self, concept):
        pass

    @abstractmethod
    def refine_object_all_values_from(self, concept):
        pass

    @abstractmethod
    def refine_object_union_of(self, concept):
        pass

    @abstractmethod
    def refine_object_intersection_of(self, concept):
        pass
