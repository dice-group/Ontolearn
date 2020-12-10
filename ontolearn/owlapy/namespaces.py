class Namespaces:
    """A Namespace and its prefix"""
    __slots__ = '_prefix', '_ns'

    _prefix: str
    _ns: str

    def __init__(self, prefix: str, ns: str):
        assert ns[-1] in ("/", ":", "#")
        self._prefix = prefix
        self._ns = ns

    @property
    def ns(self) -> str:
        return self._ns

    @property
    def prefix(self) -> str:
        return self._prefix

    def __repr__(self):
        return f'Namespaces({repr(self._prefix)}, {repr(self._ns)})'

    def __eq__(self, other):
        if type(other) is type(self):
            return self._ns == other._ns
        elif type(other) is str:
            return self._ns == other
        return NotImplemented


OWL = Namespaces("owl", "http://www.w3.org/2002/07/owl#")
RDFS = Namespaces("rdfs", "http://www.w3.org/2000/01/rdf-schema#")
RDF = Namespaces("rdf", "http://www.w3.org/1999/02/22-rdf-syntax-ns#")
XSD = Namespaces("xsd", "http://www.w3.org/2001/XMLSchema#")
