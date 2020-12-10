from ontolearn.owlapy.base import HasIRI, IRI
from ontolearn.owlapy import namespaces
from ontolearn.owlapy.namespaces import Namespaces


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


OWL_THING = OWLRDFVocabulary(namespaces.OWL, "Thing")
OWL_NOTHING = OWLRDFVocabulary(namespaces.OWL, "Nothing")
