from abc import ABCMeta, abstractmethod
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from owlapy.model._iri import IRI
    from owlapy.model import OWLLiteral


class OWLObject(metaclass=ABCMeta):
    __slots__ = ()

    @abstractmethod
    def __eq__(self, other):
        pass

    @abstractmethod
    def __hash__(self):
        pass

    @abstractmethod
    def __repr__(self):
        pass


class OWLAnnotationObject(OWLObject, metaclass=ABCMeta):
    __slots__ = ()

    # noinspection PyMethodMayBeStatic
    def as_iri(self) -> Optional['IRI']:
        return None

    # noinspection PyMethodMayBeStatic
    def as_anonymous_individual(self):
        return None


class OWLAnnotationSubject(OWLAnnotationObject, metaclass=ABCMeta):
    __slots__ = ()
    pass


class OWLAnnotationValue(OWLAnnotationObject, metaclass=ABCMeta):
    __slots__ = ()

    def is_literal(self) -> bool:
        return False

    # noinspection PyMethodMayBeStatic
    def as_literal(self) -> Optional['OWLLiteral']:
        return None
