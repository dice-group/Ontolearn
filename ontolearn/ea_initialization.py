import random
from abc import ABCMeta, abstractmethod
from deap.gp import Primitive, PrimitiveSetTyped


class AbstractEAInitialization(metaclass=ABCMeta):
    """Abstract base class for initialization methods for evolutionary algorithms.

    """
    __slots__ = ()

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def get_individual(self, pset: PrimitiveSetTyped):
        pass


class EARandomInitialization(AbstractEAInitialization):
    """Rnndom initialization methods for evolutionary algorithms.

    """
    __slots__ = 'min_height', 'max_height', 'method'

    def __init__(self, min_height: int = 3, max_height: int = 6, method: str = "rhh"):
        """
        Args:
            min_height: minimum height of trees
            max_height: maximum height of trees
            method: initialization method possible values: rhh, grow, full
        """
        super().__init__()
        self.min_height = min_height
        self.max_height = max_height
        self.method = method

    def get_individual(self, pset: PrimitiveSetTyped):
        use_grow = (self.method == 'grow' or (self.method == 'rhh' and random.random() < 0.5))

        individual = []
        height = random.randint(self.min_height, self.max_height)
        self._build_tree(individual, pset, height, 0, pset.ret, use_grow)
        return individual

    def _build_tree(self, tree, 
                    pset: PrimitiveSetTyped, 
                    height: int, 
                    current_height: int, 
                    type_: type,
                    use_grow: bool):

        if current_height == height or len(pset.primitives[type_]) == 0: 
            tree.append(random.choice(pset.terminals[type_]))
        else:
            operators = []
            if use_grow and current_height >= self.min_height:
                operators = pset.primitives[type_] + pset.terminals[type_]
            else:
                operators = pset.primitives[type_]
            
            operator = random.choice(operators)
            tree.append(operator)

            if isinstance(operator, Primitive):
                for arg_type in operator.args:
                    self._build_tree(tree, pset, height, current_height+1, arg_type, use_grow)
