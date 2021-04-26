import weakref
from abc import ABCMeta, abstractmethod
from typing import Final, Union, overload, Optional
from weakref import WeakKeyDictionary

from owlapy import namespaces
from owlapy.model._base import OWLAnnotationSubject, OWLAnnotationValue
from owlapy.namespaces import Namespaces


class HasIRI(metaclass=ABCMeta):
    __slots__ = ()

    @abstractmethod
    def get_iri(self) -> 'IRI':
        pass


class _WeakCached(type):
    __slots__ = ()

    def __init__(cls, what, bases, dct):
        super().__init__(what, bases, dct)
        cls._cache = WeakKeyDictionary()

    def __call__(cls, *args, **kwargs):
        _temp = super().__call__(*args, **kwargs)
        ret = cls._cache.get(_temp)
        if ret is None:
            cls._cache[_temp] = weakref.ref(_temp)
            return _temp
        else:
            return ret()


class _meta_IRI(ABCMeta, _WeakCached):
    __slots__ = ()
    pass


class IRI(OWLAnnotationSubject, OWLAnnotationValue, metaclass=_meta_IRI):
    """An IRI, consisting of a namespace and a remainder"""
    __slots__ = '_namespace', '_remainder', '__weakref__'
    type_index: Final = 0

    _namespace: str
    _remainder: str

    def __init__(self, namespace: Union[str, Namespaces], remainder: str):
        if isinstance(namespace, Namespaces):
            namespace = namespace.ns
        else:
            assert namespace[-1] in ("/", ":", "#")
        import sys
        self._namespace = sys.intern(namespace)
        self._remainder = remainder

    @overload
    @staticmethod
    def create(namespace: Namespaces, remainder: str) -> 'IRI': ...

    @overload
    @staticmethod
    def create(namespace: str, remainder: str) -> 'IRI': ...

    @overload
    @staticmethod
    def create(string: str) -> 'IRI': ...

    @staticmethod
    def create(string, remainder=None) -> 'IRI':
        if remainder is not None:
            return IRI(string, remainder)
        index = 1 + max(string.rfind("/"), string.rfind(":"), string.rfind("#"))
        return IRI(string[0:index], string[index:])

    def __repr__(self):
        return f"IRI({repr(self._namespace)},{repr(self._remainder)})"

    def __eq__(self, other):
        if type(other) is type(self):
            return self._namespace is other._namespace and self._remainder == other._remainder
        return NotImplemented

    def __hash__(self):
        return hash((self._namespace, self._remainder))

    def is_nothing(self):
        from owlapy import vocab
        return self == vocab.OWL_NOTHING.get_iri()

    def is_thing(self):
        from owlapy import vocab
        return self == vocab.OWL_THING.get_iri()

    def is_reserved_vocabulary(self) -> bool:
        return self._namespace == namespaces.OWL or self._namespace == namespaces.RDF \
               or self._namespace == namespaces.RDFS or self._namespace == namespaces.XSD

    def as_iri(self) -> 'IRI':
        return self

    def as_str(self) -> str:
        return self._namespace + self._remainder

    def get_short_form(self) -> str:
        return self._remainder

    def get_namespace(self) -> str:
        return self._namespace

    def get_remainder(self) -> str:
        return self._remainder