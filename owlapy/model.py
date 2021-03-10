from abc import ABCMeta, abstractmethod
from typing import Generic, Iterable, Sequence, TypeVar, Union, Final, Optional

from owlapy import vocabulary
from owlapy.base import HasIRI, IRI

_T = TypeVar('_T')

"""The OWL-APy Model class and method names should match those of OWL-API [1]

If OWL-API has streaming and getter API, it is enough to provide the streaming API only.

[1] https://github.com/owlcs/owlapi"""


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


class HasOperands(Generic[_T], metaclass=ABCMeta):
    __slots__ = ()

    @abstractmethod
    def operands(self) -> Iterable[_T]:
        pass


class OWLPropertyRange(OWLObject, metaclass=ABCMeta):
    """OWL Objects that can be the ranges of properties"""


class OWLDataRange(OWLPropertyRange, metaclass=ABCMeta):
    """Data Range"""


class OWLClassExpression(OWLPropertyRange):
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


class OWLBooleanClassExpression(OWLAnonymousClassExpression, metaclass=ABCMeta):
    __slots__ = ()
    pass


class OWLObjectComplementOf(OWLBooleanClassExpression, HasOperands[OWLClassExpression]):
    __slots__ = '_operand'
    type_index: Final = 3003

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


class OWLEntity(OWLNamedObject, metaclass=ABCMeta):
    __slots__ = ()
    pass


class OWLClass(OWLClassExpression, OWLEntity):
    """An OWL 2 named Class"""
    __slots__ = '_iri', '_is_nothing', '_is_thing'
    type_index: Final = 1001

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


class OWLDataPropertyExpression(OWLPropertyExpression, metaclass=ABCMeta):
    __slots__ = ()

    def is_data_property_expression(self):
        return True


class OWLProperty(OWLPropertyExpression, OWLEntity, metaclass=ABCMeta):
    __slots__ = ()
    pass


class OWLDataProperty(OWLDataPropertyExpression, OWLProperty):
    __slots__ = '_iri'
    type_index: Final = 1004

    _iri: IRI

    def __init__(self, iri: IRI):
        self._iri = iri

    def get_iri(self) -> IRI:
        return self._iri


class OWLObjectProperty(OWLObjectPropertyExpression, OWLProperty):
    __slots__ = '_iri'
    type_index: Final = 1002

    _iri: IRI

    def get_named_property(self) -> 'OWLObjectProperty':
        return self

    def __init__(self, iri: IRI):
        self._iri = iri

    def get_inverse_property(self) -> 'OWLObjectInverseOf':
        return OWLObjectInverseOf(self)

    def get_iri(self) -> IRI:
        return self._iri


class OWLObjectInverseOf(OWLObjectPropertyExpression):
    __slots__ = '_inverse_property'
    type_index: Final = 1003

    _inverse_property: OWLObjectProperty

    def __init__(self, inverse_property: OWLObjectProperty):
        self._inverse_property = inverse_property

    def get_inverse(self) -> OWLObjectProperty:
        return self._inverse_property

    def get_inverse_property(self) -> OWLObjectProperty:
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
    type_index: Final = 3005

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
    type_index: Final = 3006

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
    type_index: Final = 3002

    _operands: Sequence[OWLClassExpression]


class OWLObjectIntersectionOf(OWLNaryBooleanClassExpression):
    __slots__ = '_operands'
    type_index: Final = 3001

    _operands: Sequence[OWLClassExpression]


class HasCardinality(metaclass=ABCMeta):
    __slots__ = ()

    @abstractmethod
    def get_cardinality(self) -> int:
        pass


_F = TypeVar('_F', bound=OWLPropertyRange)


class OWLCardinalityRestriction(Generic[_F], OWLQuantifiedRestriction[_F], HasCardinality, metaclass=ABCMeta):
    __slots__ = ()

    _cardinality: int
    _filler: _F

    def __init__(self, cardinality: int, filler: _F):
        self._cardinality = cardinality
        self._filler = filler

    def get_cardinality(self) -> int:
        return self._cardinality

    def get_filler(self) -> _F:
        return self._filler


class OWLObjectCardinalityRestriction(OWLCardinalityRestriction[OWLClassExpression], OWLQuantifiedObjectRestriction):
    __slots__ = ()

    _property: OWLObjectPropertyExpression

    @abstractmethod
    def __init__(self, property: OWLObjectPropertyExpression, cardinality: int, filler: OWLClassExpression):
        super().__init__(cardinality, filler)
        self._property = property

    def get_property(self) -> OWLObjectPropertyExpression:
        return self._property

    def __repr__(self):
        return f"{type(self).__name__}(" \
               f"property={repr(self.get_property())},{self.get_cardinality()},filler={repr(self.get_filler())})"

    def __eq__(self, other):
        if type(other) == type(self):
            return self._property == other._property \
                   and self._cardinality == other._cardinality \
                   and self._filler == other._filler
        return NotImplemented

    def __hash__(self):
        hash((self._property, self._cardinality, self._filler))


class OWLObjectMinCardinality(OWLObjectCardinalityRestriction):
    __slots__ = '_cardinality', '_filler', '_property'
    type_index: Final = 3008

    def __init__(self, property: OWLObjectPropertyExpression, cardinality: int, filler: OWLClassExpression):
        super().__init__(property, cardinality, filler)


class OWLObjectMaxCardinality(OWLObjectCardinalityRestriction):
    __slots__ = '_cardinality', '_filler', '_property'
    type_index: Final = 3010

    def __init__(self, property: OWLObjectPropertyExpression, cardinality: int, filler: OWLClassExpression):
        super().__init__(property, cardinality, filler)


class OWLObjectExactCardinality(OWLObjectCardinalityRestriction):
    __slots__ = '_cardinality', '_filler', '_property'
    type_index: Final = 3009

    def __init__(self, property: OWLObjectPropertyExpression, cardinality: int, filler: OWLClassExpression):
        super().__init__(property, cardinality, filler)

    def as_intersection_of_min_max(self) -> OWLObjectIntersectionOf:
        args = self.get_property(), self.get_cardinality(), self.get_filler()
        return OWLObjectIntersectionOf((OWLObjectMinCardinality(*args), OWLObjectMaxCardinality(*args)))


class OWLObjectHasSelf(OWLObjectRestriction):
    __slots__ = '_property'
    type_index: Final = 3011

    _property: OWLObjectPropertyExpression

    def __init__(self, property: OWLObjectPropertyExpression):
        self._property = property

    def get_property(self) -> OWLObjectPropertyExpression:
        return self._property

    def __eq__(self, other):
        if type(other) == type(self):
            return self._property == other._property
        return NotImplemented

    def __hash__(self):
        return hash(self._property)

    def __repr__(self):
        return f'OWLObjectHasSelf({self._property})'


class OWLIndividual(OWLObject, metaclass=ABCMeta):
    __slots__ = ()
    pass


class OWLObjectHasValue(OWLHasValueRestriction[OWLIndividual], OWLObjectRestriction):
    __slots__ = '_property', '_v'
    type_index: Final = 3007

    _property: OWLObjectPropertyExpression
    _v: OWLIndividual

    def __init__(self, property: OWLObjectPropertyExpression, value: OWLIndividual):
        super().__init__(value)
        self._property = property

    def get_property(self) -> OWLObjectPropertyExpression:
        return self._property

    def __repr__(self):
        return f'OWLObjectHasValue(property={self.get_property()}, value={self._v})'


class OWLObjectOneOf(OWLAnonymousClassExpression, HasOperands[OWLIndividual]):
    __slots__ = '_values'
    type_index: Final = 3004

    def __init__(self, values: Union[OWLIndividual, Iterable[OWLIndividual]]):
        if isinstance(values, OWLIndividual):
            self._values = values,
        else:
            for _ in values:
                assert isinstance(_, OWLIndividual)
            self._values = tuple(values)

    def individuals(self) -> Iterable[OWLIndividual]:
        yield from self._values

    def operands(self) -> Iterable[OWLIndividual]:
        yield from self.individuals()

    def as_object_union_of(self) -> OWLClassExpression:
        if len(self._values) == 1:
            return self
        return OWLObjectUnionOf(map(lambda _: OWLObjectOneOf(_), self.individuals()))

    def __hash__(self):
        return hash(self._values)

    def __eq__(self, other):
        if type(other) == type(self):
            return self._values == other._values
        return NotImplemented

    def __repr__(self):
        return f'OWLObjectOneOf({self._values})'


class OWLNamedIndividual(OWLIndividual, OWLEntity):
    __slots__ = '_iri'
    type_index: Final = 1005

    _iri: IRI

    def __init__(self, iri: IRI):
        self._iri = iri

    def get_iri(self) -> IRI:
        return self._iri


_M = TypeVar('_M', bound='OWLOntologyManager')


class OWLOntologyID:
    __slots__ = '_ontology_iri', '_version_iri'

    _ontology_iri: Optional[IRI]
    _version_iri: Optional[IRI]

    def __init__(self, ontology_iri: Optional[IRI] = None, version_iri: Optional[IRI] = None):
        self._ontology_iri = ontology_iri
        self._version_iri = version_iri

    def get_ontology_iri(self) -> Optional[IRI]:
        return self._ontology_iri

    def get_version_iri(self) -> Optional[IRI]:
        return self._version_iri

    def get_default_document_iri(self) -> Optional[IRI]:
        if self._ontology_iri is not None:
            if self._version_iri is not None:
                return self._version_iri
        return self._ontology_iri

    def __repr__(self):
        return f"OWLOntologyID({repr(self._ontology_iri)}, {repr(self._version_iri)})"

    def __eq__(self, other):
        if type(other) is type(self):
            return self._ontology_iri == other._ontology_iri and self._version_iri == other._version_iri
        return NotImplemented


class OWLOntology(OWLObject, metaclass=ABCMeta):
    __slots__ = ()
    type_index: Final = 1

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

    @abstractmethod
    def get_manager(self) -> _M:
        pass

    @abstractmethod
    def get_ontology_id(self) -> OWLOntologyID:
        pass


class OWLOntologyManager(metaclass=ABCMeta):
    @abstractmethod
    def create_ontology(self, iri: IRI) -> OWLOntology:
        pass

    @abstractmethod
    def load_ontology(self, iri: IRI) -> OWLOntology:
        pass


class OWLReasoner(metaclass=ABCMeta):
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
    def sub_data_properties(self, dp: OWLDataProperty, direct: bool = False) -> Iterable[OWLDataProperty]:
        pass

    @abstractmethod
    def sub_object_properties(self, op: OWLObjectPropertyExpression, direct: bool = False) \
            -> Iterable[OWLObjectPropertyExpression]:
        pass

    @abstractmethod
    def types(self, ind: OWLNamedIndividual, direct: bool = False) -> Iterable[OWLClass]:
        pass

    @abstractmethod
    def get_root_ontology(self) -> OWLOntology:
        pass


# TODO: a big todo plus intermediate classes (missing)
#
# class OWLAnnotation(metaclass=ABCMeta):
#     type_index: Final = 5001
#
# class OWLAnnotationProperty(metaclass=ABCMeta):
#     type_index: Final = 1006
#
# class OWLAnonymousIndividual(metaclass=ABCMeta):
#     type_index: Final = 1007
#
# class OWLAxiom(metaclass=ABCMeta):
#     type_index: Final = 2000 + get_axiom_type().get_index()
#
# class OWLDataAllValuesFrom(metaclass=ABCMeta):
#     type_index: Final = 3013
#
# class OWLDataComplementOf(metaclass=ABCMeta):
#     type_index: Final = 4002
#
# class OWLDataExactCardinality(metaclass=ABCMeta):
#     type_index: Final = 3016
#
# class OWLDataHasValue(metaclass=ABCMeta):
#     type_index: Final = 3014
#
# class OWLDataIntersectionOf(metaclass=ABCMeta):
#     type_index: Final = 4004
#
# class OWLDataMaxCardinality(metaclass=ABCMeta):
#     type_index: Final = 3017
#
# class OWLDataMinCardinality(metaclass=ABCMeta):
#     type_index: Final = 3015
#
# class OWLDataOneOf(metaclass=ABCMeta):
#     type_index: Final = 4003
#
# class OWLDataSomeValuesFrom(metaclass=ABCMeta):
#     type_index: Final = 3012
#
# class OWLDataUnionOf(metaclass=ABCMeta):
#     type_index: Final = 4005
#
# class OWLDatatype(OWLEntity, metaclass=ABCMeta):
#     type_index: Final = 4001
#
# class OWLDatatypeRestriction(metaclass=ABCMeta):
#     type_index: Final = 4006
#
# class OWLFacetRestriction(metaclass=ABCMeta):
#     type_index: Final = 4007
#
# class OWLLiteral(metaclass=ABCMeta):
#     type_index: Final = 4008


"""Important constant objects section"""

OWLThing: Final = OWLClass(vocabulary.OWL_THING.get_iri())
OWLNothing: Final = OWLClass(vocabulary.OWL_NOTHING.get_iri())
OWLTopObjectProperty: Final = OWLObjectProperty(vocabulary.OWL_TOP_OBJECT_PROPERTY.get_iri())
OWLBottomObjectProperty: Final = OWLObjectProperty(vocabulary.OWL_BOTTOM_OBJECT_PROPERTY.get_iri())
OWLTopDataProperty: Final = OWLDataProperty(vocabulary.OWL_TOP_DATA_PROPERTY.get_iri())
OWLBottomDataProperty: Final = OWLDataProperty(vocabulary.OWL_BOTTOM_DATA_PROPERTY.get_iri())
