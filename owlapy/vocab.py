from abc import ABCMeta
from enum import Enum, EnumMeta
from typing import Final

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
    OWL_THING = (namespaces.OWL, "Thing")  #:
    OWL_NOTHING = (namespaces.OWL, "Nothing")  #:
    OWL_TOP_OBJECT_PROPERTY = (namespaces.OWL, "topObjectProperty")  #:
    OWL_BOTTOM_OBJECT_PROPERTY = (namespaces.OWL, "bottomObjectProperty")  #:
    OWL_TOP_DATA_PROPERTY = (namespaces.OWL, "topDataProperty")  #:
    OWL_BOTTOM_DATA_PROPERTY = (namespaces.OWL, "bottomDataProperty")  #:
    RDFS_LITERAL = (namespaces.RDFS, "Literal")  #:


class XSDVocabulary(_Vocabulary, Enum, metaclass=_meta_Enum):
    def __init__(self, remainder: str):
        super().__init__(namespaces.XSD, remainder)
    DECIMAL: Final = "decimal"  #:
    INTEGER: Final = "integer"  #:
    LONG: Final = "long"  #:
    DOUBLE: Final = "double"  #:
    FLOAT: Final = "float"  #:
    BOOLEAN: Final = "boolean"  #:
    DATE_TIME_STAMP: Final = "dateTimeStamp"  #:
