from abc import ABCMeta, abstractmethod
from typing import Generic, Iterable, Sequence, TypeVar

from ontolearn.owlapy import vocabulary
from ontolearn.owlapy.base import HasIRI, IRI

_T = TypeVar('_T')


class OWLObject(metaclass=ABCMeta):
    __slots__ = ()
    pass


class HasOperands(Generic[_T], metaclass=ABCMeta):
    __slots__ = ()

    @abstractmethod
    def operands(self) -> Iterable[_T]:
        pass


class OWLClassExpression(OWLObject):
    """An OWL 2 Class Expression"""
    __slots__ = ()

    @abstractmethod
    def is_owl_thing(self) -> bool:
        pass

    @abstractmethod
    def is_owl_nothing(self) -> bool:
        pass

    @abstractmethod
    def get_object_complement_of(self) -> 'OWLObjectComplementOf':
        pass


class OWLAnonymousClassExpression(OWLClassExpression, metaclass=ABCMeta):
    """A Class Expression which is not a named Class"""

    def is_owl_nothing(self) -> bool:
        return False

    def is_owl_thing(self) -> bool:
        return False

    def get_object_complement_of(self) -> 'OWLObjectComplementOf':
        return OWLObjectComplementOf(self)


class OWLBooleanClassExpression(OWLAnonymousClassExpression):
    __slots__ = ()
    pass


class OWLObjectComplementOf(OWLBooleanClassExpression, HasOperands[OWLClassExpression]):
    __slots__ = '_operand'

    _operand: OWLClassExpression

    def __init__(self, op: OWLClassExpression):
        self._operand = op

    def get_operand(self) -> OWLClassExpression:
        return self._operand

    def operands(self) -> Iterable[OWLClassExpression]:
        yield self._operand

    def __repr__(self):
        return f"OWLObjectComplementOf({repr(self._operand)})"

    def __eq__(self, other):
        if type(other) is type(self):
            return self._operand == other._operand
        return NotImplemented

    def __hash__(self):
        return hash(self._operand)


class OWLNamedObject(OWLObject, HasIRI, metaclass=ABCMeta):
    __slots__ = ()

    _iri: IRI

    def __eq__(self, other):
        if type(other) is type(self):
            return self._iri == other._iri
        return NotImplemented

    def __lt__(self, other):
        if type(other) is type(self):
            return self._iri.as_str() < other._iri.as_str()
        return NotImplemented

    def __hash__(self):
        return hash(self._iri)

    def __repr__(self):
        return f"{type(self).__name__}({repr(self._iri)})"

    pass


class OWLClass(OWLClassExpression, OWLNamedObject):
    """An OWL 2 named Class"""
    __slots__ = '_iri', '_is_nothing', '_is_thing'

    _iri: IRI
    _is_nothing: bool
    _is_thing: bool

    def __init__(self, iri: IRI):
        self._is_nothing = iri.is_nothing()
        self._is_thing = iri.is_thing()
        self._iri = iri

    def get_iri(self) -> IRI:
        return self._iri

    def is_owl_thing(self) -> bool:
        return self._is_thing

    def is_owl_nothing(self) -> bool:
        return self._is_nothing

    def get_object_complement_of(self) -> OWLObjectComplementOf:
        return OWLObjectComplementOf(self)


class OWLPropertyExpression(OWLObject, metaclass=ABCMeta):
    __slots__ = ()

    def is_data_property_expression(self) -> bool:
        return False

    def is_object_property_expression(self) -> bool:
        return False


class OWLRestriction(OWLAnonymousClassExpression):
    __slots__ = ()

    @abstractmethod
    def get_property(self) -> OWLPropertyExpression:
        pass

    def is_data_restriction(self) -> bool:
        return False

    def is_object_restriction(self) -> bool:
        return False


class OWLObjectPropertyExpression(OWLPropertyExpression):
    __slots__ = ()

    @abstractmethod
    def get_inverse_property(self) -> 'OWLObjectPropertyExpression':
        pass

    @abstractmethod
    def get_named_property(self) -> 'OWLObjectProperty':
        pass

    def is_object_property_expression(self) -> bool:
        return True


class OWLDataPropertyExpression(OWLPropertyExpression):
    __slots__ = ()

    def is_data_property_expression(self):
        return True


class OWLProperty(OWLPropertyExpression, OWLNamedObject, metaclass=ABCMeta):
    __slots__ = ()
    pass


class OWLDataProperty(OWLDataPropertyExpression, OWLProperty):
    __slots__ = '_iri'

    _iri: IRI

    def __init__(self, iri: IRI):
        self._iri = iri

    def get_iri(self) -> IRI:
        return self._iri


class OWLObjectProperty(OWLObjectPropertyExpression, OWLProperty):
    __slots__ = '_iri'

    _iri: IRI

    def get_named_property(self) -> 'OWLObjectProperty':
        return self

    def __init__(self, iri: IRI):
        self._iri = iri

    def get_inverse_property(self) -> OWLObjectPropertyExpression:
        return OWLObjectInverseOf(self)

    def get_iri(self) -> IRI:
        return self._iri


class OWLObjectInverseOf(OWLObjectPropertyExpression):
    __slots__ = '_inverse_property'

    _inverse_property: OWLObjectProperty

    def __init__(self, inverse_property: OWLObjectProperty):
        self._inverse_property = inverse_property

    def get_inverse(self) -> OWLObjectProperty:
        return self._inverse_property

    def get_inverse_property(self) -> OWLObjectPropertyExpression:
        return self.get_inverse()

    def get_named_property(self) -> OWLObjectProperty:
        return self._inverse_property

    def __repr__(self):
        return f"OWLObjectInverseOf({repr(self._inverse_property)})"

    def __eq__(self, other):
        if type(other) is type(self):
            return self._inverse_property == other._inverse_property
        return NotImplemented

    def __hash__(self):
        return hash(self._inverse_property)


class OWLDataRestriction(OWLRestriction, metaclass=ABCMeta):
    __slots__ = ()

    def is_data_restriction(self) -> bool:
        return True

    pass


class OWLObjectRestriction(OWLRestriction, metaclass=ABCMeta):
    __slots__ = ()

    def is_object_restriction(self) -> bool:
        return True

    @abstractmethod
    def get_property(self) -> OWLObjectPropertyExpression:
        pass


class HasFiller(Generic[_T], metaclass=ABCMeta):
    __slots__ = ()

    @abstractmethod
    def get_filler(self) -> _T:
        pass


class OWLHasValueRestriction(Generic[_T], OWLRestriction, HasFiller[_T], metaclass=ABCMeta):
    __slots__ = ()

    _v: _T

    def __init__(self, value: _T):
        self._v = value

    def __eq__(self, other):
        if type(other) is type(self):
            return self._v == other._v
        return NotImplemented

    def __hash__(self):
        return hash(self._v)

    def get_filler(self) -> _T:
        return self._v


class OWLQuantifiedRestriction(Generic[_T], OWLRestriction, HasFiller[_T], metaclass=ABCMeta):
    __slots__ = ()
    pass


class OWLQuantifiedObjectRestriction(OWLQuantifiedRestriction[OWLClassExpression], OWLObjectRestriction,
                                     metaclass=ABCMeta):
    __slots__ = ()

    _filler: OWLClassExpression

    def __init__(self, filler: OWLClassExpression):
        self._filler = filler

    def get_filler(self) -> OWLClassExpression:
        return self._filler


class OWLObjectSomeValuesFrom(OWLQuantifiedObjectRestriction):
    __slots__ = '_property', '_filler'

    def __init__(self, property: OWLObjectPropertyExpression, filler: OWLClassExpression):
        super().__init__(filler)
        self._property = property

    def __repr__(self):
        return f"OWLObjectSomeValuesFrom(property={repr(self._property)},filler={repr(self._filler)})"

    def __eq__(self, other):
        if type(other) is type(self):
            return self._filler == other._filler and self._property == other._property
        return NotImplemented

    def __hash__(self):
        return hash((self._filler, self._property))

    def get_property(self) -> OWLObjectPropertyExpression:
        return self._property


class OWLObjectAllValuesFrom(OWLQuantifiedObjectRestriction):
    __slots__ = '_property', '_filler'

    def __init__(self, property: OWLObjectPropertyExpression, filler: OWLClassExpression):
        super().__init__(filler)
        self._property = property

    def __repr__(self):
        return f"OWLObjectAllValuesFrom(property={repr(self._property)},filler={repr(self._filler)})"

    def __eq__(self, other):
        if type(other) is type(self):
            return self._filler == other._filler and self._property == other._property
        return NotImplemented

    def __hash__(self):
        return hash((self._filler, self._property))

    def get_property(self) -> OWLObjectPropertyExpression:
        return self._property


class OWLNaryBooleanClassExpression(OWLBooleanClassExpression, HasOperands[OWLClassExpression]):
    __slots__ = ()

    _operands: Sequence[OWLClassExpression]

    def __init__(self, operands: Iterable[OWLClassExpression]):
        self._operands = tuple(operands)

    def operands(self) -> Iterable[OWLClassExpression]:
        for o in self._operands:
            yield o

    def __repr__(self):
        return f'{type(self).__name__}({repr(self._operands)})'

    def __eq__(self, other):
        if type(other) == type(self):
            return self._operands == other._operands
        return NotImplemented

    def __hash__(self):
        return hash(self._operands)


class OWLObjectUnionOf(OWLNaryBooleanClassExpression):
    __slots__ = '_operands'

    _operands: Sequence[OWLClassExpression]


class OWLObjectIntersectionOf(OWLNaryBooleanClassExpression):
    __slots__ = '_operands'

    _operands: Sequence[OWLClassExpression]


class OWLIndividual(OWLObject):
    __slots__ = ()
    pass


class OWLNamedIndividual(OWLIndividual, OWLNamedObject):
    __slots__ = '_iri'

    _iri: IRI

    def __init__(self, iri: IRI):
        self._iri = iri

    def get_iri(self) -> IRI:
        return self._iri


class OWLOntology(OWLObject, metaclass=ABCMeta):
    __slots__ = ()

    @abstractmethod
    def classes_in_signature(self) -> Iterable[OWLClass]:
        pass

    @abstractmethod
    def data_properties_in_signature(self) -> Iterable[OWLDataProperty]:
        pass

    @abstractmethod
    def object_properties_in_signature(self) -> Iterable[OWLObjectProperty]:
        pass

    @abstractmethod
    def individuals_in_signature(self) -> Iterable[OWLNamedIndividual]:
        pass


class OWLOntologyManager(metaclass=ABCMeta):
    @abstractmethod
    def create_ontology(self, iri: IRI) -> OWLOntology:
        pass

    @abstractmethod
    def load_ontology(self, iri: IRI) -> OWLOntology:
        pass


class OWLReasoner:
    __slots__ = ()

    @abstractmethod
    def __init__(self, ontology: OWLOntology):
        pass

    @abstractmethod
    def data_property_domains(self, pe: OWLDataProperty, direct: bool = False) -> Iterable[OWLClass]:
        pass

    @abstractmethod
    def object_property_domains(self, pe: OWLObjectProperty, direct: bool = False) -> Iterable[OWLClass]:
        pass

    @abstractmethod
    def object_property_ranges(self, pe: OWLObjectProperty, direct: bool = False) -> Iterable[OWLClass]:
        pass

    @abstractmethod
    def equivalent_classes(self, ce: OWLClassExpression) -> Iterable[OWLClass]:
        pass

    @abstractmethod
    def data_property_values(self, ind: OWLNamedIndividual, pe: OWLDataProperty) -> Iterable:
        pass

    @abstractmethod
    def object_property_values(self, ind: OWLNamedIndividual, pe: OWLObjectProperty) -> Iterable[OWLNamedIndividual]:
        pass

    @abstractmethod
    def flush(self) -> None:
        pass

    @abstractmethod
    def instances(self, ce: OWLClassExpression, direct: bool = False) -> Iterable[OWLNamedIndividual]:
        pass

    @abstractmethod
    def sub_classes(self, ce: OWLClassExpression, direct: bool = False) -> Iterable[OWLClass]:
        pass

    @abstractmethod
    def types(self, ind: OWLNamedIndividual, direct: bool = False) -> Iterable[OWLClass]:
        pass

    @abstractmethod
    def get_root_ontology(self) -> OWLOntology:
        pass


OWLThing = OWLClass(vocabulary.OWL_THING.get_iri())
OWLNothing = OWLClass(vocabulary.OWL_NOTHING.get_iri())
