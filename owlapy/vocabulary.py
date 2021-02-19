from typing import Final

from owlapy.base import HasIRI, IRI
from owlapy import namespaces
from owlapy.namespaces import Namespaces


class OWLRDFVocabulary(HasIRI):
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


OWL_THING: Final = OWLRDFVocabulary(namespaces.OWL, "Thing")
OWL_NOTHING: Final = OWLRDFVocabulary(namespaces.OWL, "Nothing")
OWL_TOP_OBJECT_PROPERTY: Final = OWLRDFVocabulary(namespaces.OWL, "topObjectProperty")
OWL_BOTTOM_OBJECT_PROPERTY: Final = OWLRDFVocabulary(namespaces.OWL, "bottomObjectProperty")
OWL_TOP_DATA_PROPERTY: Final = OWLRDFVocabulary(namespaces.OWL, "topDataProperty")
OWL_BOTTOM_DATA_PROPERTY: Final = OWLRDFVocabulary(namespaces.OWL, "bottomDataProperty")
