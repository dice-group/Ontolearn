from abc import ABCMeta
from enum import Enum, EnumMeta
from typing import Final, Callable, TypeVar
from operator import lt, le, gt, ge

from owlapy import namespaces
from owlapy.model._iri import HasIRI, IRI
from owlapy.namespaces import Namespaces


class _Vocabulary(HasIRI):
    __slots__ = '_namespace', '_remainder', '_iri'

    _namespace: Namespaces
    _remainder: str
    _iri: IRI

    def __init__(self, namespace: Namespaces, remainder: str):
        self._namespace = namespace
        self._remainder = remainder
        self._iri = IRI(namespace, remainder)

    def get_iri(self) -> IRI:
        return self._iri

    def __repr__(self):
        return f"<<{self._namespace.prefix}:{self._remainder}>>"


class _meta_Enum(ABCMeta, EnumMeta):
    __slots__ = ()
    pass


class OWLRDFVocabulary(_Vocabulary, Enum, metaclass=_meta_Enum):
    def __new__(cls, namespace: Namespaces, remainder: str, *args):
        obj = object.__new__(cls)
        obj._value_ = f"{namespace.prefix}:{remainder}"
        return obj
    OWL_THING = (namespaces.OWL, "Thing")  #:
    OWL_NOTHING = (namespaces.OWL, "Nothing")  #:
    OWL_TOP_OBJECT_PROPERTY = (namespaces.OWL, "topObjectProperty")  #:
    OWL_BOTTOM_OBJECT_PROPERTY = (namespaces.OWL, "bottomObjectProperty")  #:
    OWL_TOP_DATA_PROPERTY = (namespaces.OWL, "topDataProperty")  #:
    OWL_BOTTOM_DATA_PROPERTY = (namespaces.OWL, "bottomDataProperty")  #:
    RDFS_LITERAL = (namespaces.RDFS, "Literal")  #:


class XSDVocabulary(_Vocabulary, Enum, metaclass=_meta_Enum):
    def __new__(cls, remainder: str, *args):
        obj = object.__new__(cls)
        obj._value_ = f"{namespaces.XSD.prefix}:{remainder}"
        return obj

    def __init__(self, remainder: str):
        super().__init__(namespaces.XSD, remainder)
    DECIMAL: Final = "decimal"  #:
    INTEGER: Final = "integer"  #:
    LONG: Final = "long"  #:
    DOUBLE: Final = "double"  #:
    FLOAT: Final = "float"  #:
    BOOLEAN: Final = "boolean"  #:
    DATE: Final = "date"  #:
    DATE_TIME: Final = "dateTime"  #:
    DATE_TIME_STAMP: Final = "dateTimeStamp"  #:
    DURATION: Final = "duration"  #:


_X = TypeVar('_X')


class OWLFacet(_Vocabulary, Enum, metaclass=_meta_Enum):
    def __new__(cls, remainder: str, *args):
        obj = object.__new__(cls)
        obj._value_ = f"{namespaces.XSD.prefix}:{remainder}"
        return obj

    def __init__(self, remainder: str, symbolic_form: str, operator: Callable[[_X, _X], bool]):
        super().__init__(namespaces.XSD, remainder)
        self._symbolic_form = symbolic_form
        self._operator = operator

    @property
    def symbolic_form(self):
        return self._symbolic_form

    @property
    def operator(self):
        return self._operator

    MIN_INCLUSIVE: Final = ("minInclusive", "≥", ge)  #:
    MIN_EXCLUSIVE: Final = ("minExclusive", ">", gt)  #:
    MAX_INCLUSIVE: Final = ("maxInclusive", "≤", le)  #:
    MAX_EXCLUSIVE: Final = ("maxExclusive", "<", lt)  #:
