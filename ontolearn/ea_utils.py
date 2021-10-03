from enum import Enum
from typing import Final
from owlapy.model import OWLClassExpression, OWLObjectPropertyExpression
from ontolearn.knowledge_base import KnowledgeBase
import re


class PrimitiveFactory:

    __slots__ = 'knowledge_base'

    def __init__(self, knowledge_base: KnowledgeBase):
        self.knowledge_base = knowledge_base

    def create_union(self):

        def union(A: OWLClassExpression, B: OWLClassExpression) -> OWLClassExpression:
            return self.knowledge_base.union([A, B])

        return union

    def create_intersection(self):

        def intersection(A: OWLClassExpression, B: OWLClassExpression) -> OWLClassExpression:
            return self.knowledge_base.intersection([A, B])

        return intersection

    def create_existential_universal(self, property_: OWLObjectPropertyExpression):

        def existential_restriction(filler: OWLClassExpression) -> OWLClassExpression:
            return self.knowledge_base.existential_restriction(filler, property_)

        def universal_restriction(filler: OWLClassExpression) -> OWLClassExpression:
            return self.knowledge_base.universal_restriction(filler, property_)

        return existential_restriction, universal_restriction


class OperatorVocabulary(str, Enum):
    UNION: Final = "union"  #:
    INTERSECTION: Final = "intersection"  #:
    NEGATION: Final = "negation"  #:
    EXISTENTIAL: Final = "exists"  #:
    UNIVERSAL: Final = "forall"  #:


def escape(name: str) -> str:
    name = re.sub(r'\W+', '', name)
    return name
