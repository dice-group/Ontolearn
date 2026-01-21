from queue import PriorityQueue
from abc import ABC
from typing import Set, List

class Role:
    """Represents an OWL object property/role."""
    def __init__(self, *, name: str):
        assert isinstance(name, str)
        self.name = name

    def __str__(self):
        return f'Role at {hex(id(self))} | {self.name}'

    def __repr__(self):
        return self.__str__()


class TargetClassExpression:
    """Represents a target class expression for neural training."""
    def __init__(self, *, label_id, name: str, idx_individuals: Set = None,
                 expression_chain: List = None, length: int = None,
                 str_individuals: Set = None, _type=None):
        self.label_id = label_id
        self.name = name
        self.idx_individuals = idx_individuals
        self.str_individuals = str_individuals
        self.type = _type
        self.expression_chain = expression_chain
        self.num_individuals = len(self.str_individuals) if self.str_individuals else 0
        self.length = length
        self.quality = None

    @property
    def size(self):
        return self.num_individuals

    def __lt__(self, other):
        return self.quality < other.quality

    def __str__(self):
        return f'TargetClassExpression | {self.name} | Indv:{self.num_individuals} | Quality:{self.quality}'

    def __repr__(self):
        return self.__str__()


class ClassExpression(ABC):
    """Base class for class expressions."""
    def __init__(self, *, name: str, str_individuals: Set, expression_chain: List,
                 owl_class=None, quality=None, length=None):
        assert isinstance(name, str)
        assert isinstance(str_individuals, set)
        assert isinstance(expression_chain, (list, tuple))
        self.name = name
        self.str_individuals = str_individuals
        self.expression_chain = expression_chain
        self.num_individuals = len(self.str_individuals)
        self.quality = quality if quality is not None else -1.0
        self.owl_class = owl_class
        self.length = length if length is not None else len(self.name.split())
        self.type = 'class_expression'  # Default type, overridden by subclasses

    def __str__(self):
        return f'{self.type} | {self.name} | Indv:{self.num_individuals} | Quality:{self.quality:.3f}'

    def __repr__(self):
        return self.__str__()

    @property
    def size(self):
        return self.num_individuals

    def __lt__(self, other):
        return self.quality < other.quality

    def __mul__(self, other):
        """Create intersection of two class expressions (A ⊓ B)"""
        if self.length <= 2 and other.length <= 2:
            name = f'{self.name} ⊓ {other.name}'
        elif self.length <= 2 < other.length:
            name = f'{self.name} ⊓ ({other.name})'
        elif self.length > other.length <= 2:
            name = f'({self.name}) ⊓ {other.name}'
        elif self.length >= 2 and other.length >= 2:
            name = f'({self.name}) ⊓ ({other.name})'
        else:
            name = f'{self.name} ⊓ {other.name}'

        return IntersectionClassExpression(
            name=name,
            length=self.length + other.length + 1,
            concepts=(self, other),
            str_individuals=self.str_individuals.intersection(other.str_individuals),
            expression_chain=((self.expression_chain + (self.name,)), 'AND',
                            (other.expression_chain + (other.name,)))
        )

    def __add__(self, other):
        """Create union of two class expressions (A ⊔ B)"""
        if self.length <= 2 and other.length <= 2:
            name = f'{self.name} ⊔ {other.name}'
        elif self.length <= 2 < other.length:
            name = f'{self.name} ⊔ ({other.name})'
        elif self.length > other.length <= 2:
            name = f'({self.name}) ⊔ {other.name}'
        elif self.length >= 2 and other.length >= 2:
            name = f'({self.name}) ⊔ ({other.name})'
        else:
            name = f'{self.name} ⊔ {other.name}'

        return UnionClassExpression(
            name=name,
            length=self.length + other.length + 1,
            str_individuals=self.str_individuals.union(other.str_individuals),
            concepts=(self, other),
            expression_chain=((self.expression_chain + (self.name,)), 'OR',
                            (other.expression_chain + (other.name,)))
        )


class AtomicExpression(ClassExpression):
    """Represents an atomic class expression."""
    def __init__(self, *, name: str, str_individuals: Set, expression_chain: List,
                 owl_class=None, quality=None, label_id=None, idx_individuals=None):
        super().__init__(name=name, str_individuals=str_individuals,
                         expression_chain=expression_chain, quality=quality, owl_class=owl_class)
        self.length = 1
        self.type = 'atomic_expression'
        self.idx_individuals = idx_individuals
        self.label_id = label_id


class ComplementOfAtomicExpression(ClassExpression):
    """Represents a negated atomic class expression."""
    def __init__(self, *, name: str, atomic_expression, str_individuals: Set,
                 expression_chain: List, quality=None, owl_class=None,
                 label_id=None, idx_individuals=None):
        super().__init__(name=name, str_individuals=str_individuals,
                         expression_chain=expression_chain, quality=quality, owl_class=owl_class)
        self.atomic_expression = atomic_expression
        self.length = 2
        self.type = 'negated_expression'
        self.label_id = label_id
        self.idx_individuals = idx_individuals


class UniversalQuantifierExpression(ClassExpression):
    """Represents a universal quantifier expression (∀)."""
    def __init__(self, *, name: str, role=None, filler=None, label_id=None,
                 idx_individuals=None, str_individuals: Set, expression_chain: List, quality=None):
        assert isinstance(name, str)
        assert isinstance(str_individuals, set)
        assert isinstance(expression_chain, (list, tuple))
        super().__init__(name=name, str_individuals=str_individuals,
                         expression_chain=expression_chain, quality=quality)
        self.role = role
        self.filler = filler
        self.type = "universal_quantifier_expression"
        self.label_id = label_id
        self.idx_individuals = idx_individuals
        self.length = 3


class ExistentialQuantifierExpression(ClassExpression):
    """Represents an existential quantifier expression (∃)."""
    def __init__(self, *, name: str, role=None, filler=None, str_individuals: Set,
                 expression_chain: List, quality=None, label_id=None, idx_individuals=None):
        assert isinstance(name, str)
        assert isinstance(str_individuals, set)
        assert isinstance(expression_chain, (list, tuple))
        super().__init__(name=name, str_individuals=str_individuals,
                         expression_chain=expression_chain, quality=quality)
        self.role = role
        self.filler = filler
        self.type = "existantial_quantifier_expression"
        self.label_id = label_id
        self.idx_individuals = idx_individuals
        self.length = 3


class IntersectionClassExpression(ClassExpression):
    """Represents an intersection of class expressions."""
    def __init__(self, *, name: str, length: int, str_individuals: Set,
                 expression_chain: List, owl_class=None, quality=None,
                 label_id=None, concepts=None, idx_individuals=None):
        super().__init__(name=name, str_individuals=str_individuals,
                         expression_chain=expression_chain, quality=quality, owl_class=owl_class)
        assert length >= 3
        self.length = length
        self.type = 'intersection_expression'
        self.label_id = label_id
        self.idx_individuals = idx_individuals
        self.concepts = concepts


class UnionClassExpression(ClassExpression):
    """Represents a union of class expressions."""
    def __init__(self, *, name: str, length: int, str_individuals: Set,
                 expression_chain: List, owl_class=None, concepts=None,
                 quality=None, label_id=None, idx_individuals=None):
        super().__init__(name=name, str_individuals=str_individuals,
                         expression_chain=expression_chain, quality=quality, owl_class=owl_class)
        assert length >= 3
        self.length = length
        self.type = 'union_expression'
        self.label_id = label_id
        self.idx_individuals = idx_individuals
        self.concepts = concepts

class State:
    def __init__(self, quality: float, tce: TargetClassExpression, str_individuals: set):
        self.quality = quality
        self.tce = tce
        self.name = self.tce.name
        self.str_individuals = str_individuals

    def __lt__(self, other):
        return self.quality < other.quality

    def __str__(self):
        return f'{self.tce} | Quality:{self.quality:.3f}'

class SearchTree:
    """Priority queue for managing search states."""
    def __init__(self, maxsize=0):
        self.items_in_queue = PriorityQueue(maxsize)
        self.gate = dict()

    def __contains__(self, key):
        return key in self.gate

    def put(self, expression, key=None, condition=None):
        if condition is None:
            if expression.name not in self.gate:
                if key is None:
                    key = -expression.quality
                # The lowest valued entries are retrieved first
                self.items_in_queue.put((key, expression))
                self.gate[expression.name] = expression
        else:
            raise ValueError('Define the condition')

    def get(self):
        _, expression = self.items_in_queue.get(timeout=1)
        del self.gate[expression.name]
        return expression

    def get_all(self):
        return list(self.gate.values())

    def __len__(self):
        return len(self.gate)

    def __iter__(self):
        # No garantie of returing best
        return (exp for q, exp in self.items_in_queue.queue)

    def extend_queue(self, other) -> None:
        """
        Extend queue with other queue.
        :param other:
        """
        for i in other:
            self.put(i)
