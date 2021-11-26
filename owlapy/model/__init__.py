"""The OWL-APy Model class and method names should match those of OWL API [1]

If OWL API has streaming and getter API, it is enough to provide the streaming API only.

many help texts copied from OWL API

[1] https://github.com/owlcs/owlapi"""

from abc import ABCMeta, abstractmethod
from functools import total_ordering
from typing import Generic, Iterable, Sequence, Set, TypeVar, Union, Final, Optional, Protocol, ClassVar, List
from pandas import Timedelta
from datetime import datetime, date

from owlapy.vocab import OWLRDFVocabulary, XSDVocabulary, OWLFacet
from owlapy._utils import MOVE
from owlapy.model._base import OWLObject, OWLAnnotationObject, OWLAnnotationSubject, OWLAnnotationValue
from owlapy.model._iri import HasIRI, IRI

MOVE(OWLObject, OWLAnnotationObject, OWLAnnotationSubject, OWLAnnotationValue, HasIRI, IRI)

_T = TypeVar('_T')  #:
_C = TypeVar('_C', bound='OWLObject')  #:
Literals = Union['OWLLiteral', int, float, bool, Timedelta, datetime, date]  #:


class HasIndex(Protocol):
    """Interface for types with an index; this is used to group objects by type when sorting."""
    type_index: ClassVar[int]  #: index for this type. This is a sorting index for the types.

    def __eq__(self, other): ...


class HasOperands(Generic[_T], metaclass=ABCMeta):
    """An interface to objects that have a collection of operands.

    Args:
        _T: operand type
    """
    __slots__ = ()

    @abstractmethod
    def operands(self) -> Iterable[_T]:
        """Gets the operands - e.g., the individuals in a sameAs axiom, or the classes in an equivalent
        classes axiom.

        Returns:
            The operands.
        """
        pass


class OWLPropertyRange(OWLObject, metaclass=ABCMeta):
    """OWL Objects that can be the ranges of properties"""


class OWLDataRange(OWLPropertyRange, metaclass=ABCMeta):
    """Represents a DataRange in the OWL 2 Specification"""


class OWLClassExpression(OWLPropertyRange):
    """An OWL 2 Class Expression"""
    __slots__ = ()

    @abstractmethod
    def is_owl_thing(self) -> bool:
        """Determines if this expression is the built in class owl:Thing. This method does not determine if the class
        is equivalent to owl:Thing.

        Returns:
            :True if this expression is owl:Thing
        """
        pass

    @abstractmethod
    def is_owl_nothing(self) -> bool:
        """Determines if this expression is the built in class owl:Nothing. This method does not determine if the class
        is equivalent to owl:Nothing.
        """
        pass

    @abstractmethod
    def get_object_complement_of(self) -> 'OWLObjectComplementOf':
        """Gets the object complement of this class expression

        Returns:
            A class expression that is the complement of this class expression.
        """
        pass

    @abstractmethod
    def get_nnf(self) -> 'OWLClassExpression':
        """Gets the negation normal form of the complement of this expression.

        Returns:
            A expression that represents the NNF of the complement of this expression.
        """
        pass


class OWLAnonymousClassExpression(OWLClassExpression, metaclass=ABCMeta):
    """A Class Expression which is not a named Class"""

    def is_owl_nothing(self) -> bool:
        # documented in parent
        return False

    def is_owl_thing(self) -> bool:
        # documented in parent
        return False

    def get_object_complement_of(self) -> 'OWLObjectComplementOf':
        # documented in parent
        return OWLObjectComplementOf(self)

    def get_nnf(self) -> 'OWLClassExpression':
        # documented in parent
        from owlapy.util import NNF
        return NNF().get_class_nnf(self)


class OWLBooleanClassExpression(OWLAnonymousClassExpression, metaclass=ABCMeta):
    __slots__ = ()
    pass


class OWLObjectComplementOf(OWLBooleanClassExpression, HasOperands[OWLClassExpression]):
    """Represents an ObjectComplementOf class expression in the OWL 2 Specification."""
    __slots__ = '_operand'
    type_index: Final = 3003

    _operand: OWLClassExpression

    def __init__(self, op: OWLClassExpression):
        """
        Args:
            op: class expression to complement
        """
        self._operand = op

    def get_operand(self) -> OWLClassExpression:
        """
        Returns:
            the wrapped expression
        """
        return self._operand

    def operands(self) -> Iterable[OWLClassExpression]:
        # documented in parent
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
    """Represents a named object for example, class, property, ontology etc. - i.e. anything that has an
     IRI as its name."""
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
    """Represents Entities in the OWL 2 Specification."""
    __slots__ = ()

    def to_string_id(self) -> str:
        return self.get_iri().as_str()

    def is_anonymous(self) -> bool:
        return False

    pass


class OWLClass(OWLClassExpression, OWLEntity):
    """An OWL 2 named Class"""
    __slots__ = '_iri', '_is_nothing', '_is_thing'
    type_index: Final = 1001

    _iri: IRI
    _is_nothing: bool
    _is_thing: bool

    def __init__(self, iri: IRI):
        """Gets an instance of OWLClass that has the specified IRI.

        Args:
            iri: The IRI.
        """
        self._is_nothing = iri.is_nothing()
        self._is_thing = iri.is_thing()
        self._iri = iri

    def get_iri(self) -> IRI:
        # documented in parent
        return self._iri

    def is_owl_thing(self) -> bool:
        # documented in parent
        return self._is_thing

    def is_owl_nothing(self) -> bool:
        # documented in parent
        return self._is_nothing

    def get_object_complement_of(self) -> OWLObjectComplementOf:
        # documented in parent
        return OWLObjectComplementOf(self)

    def get_nnf(self) -> 'OWLClass':
        # documented in parent
        return self


class OWLPropertyExpression(OWLObject, metaclass=ABCMeta):
    """Represents a property or possibly the inverse of a property."""
    __slots__ = ()

    def is_data_property_expression(self) -> bool:
        """
        Returns:
            True if this is a data property
        """
        return False

    def is_object_property_expression(self) -> bool:
        """
        Returns:
            True if this is an object property
        """
        return False

    def is_owl_top_object_property(self) -> bool:
        """Determines if this is the owl:topObjectProperty.

        Returns:
            :True if this property is the owl:topObjectProperty
        """
        return False

    def is_owl_top_data_property(self) -> bool:
        """Determines if this is the owl:topDataProperty.

        Returns:
            :True if this property is the owl:topDataProperty
        """
        return False


class OWLRestriction(OWLAnonymousClassExpression):
    """Represents a Object Property Restriction or Data Property Restriction in the OWL 2 specification."""
    __slots__ = ()

    @abstractmethod
    def get_property(self) -> OWLPropertyExpression:
        """
        Returns:
            property being restricted
        """
        pass

    def is_data_restriction(self) -> bool:
        """Determines if this is a data restriction

        Returns:
            True if this is a data restriction
        """
        return False

    def is_object_restriction(self) -> bool:
        """Determines if this is an object restriction

        Returns:
            True if this is an object restriction
        """
        return False


class OWLObjectPropertyExpression(OWLPropertyExpression):
    __slots__ = ()

    @abstractmethod
    def get_inverse_property(self) -> 'OWLObjectPropertyExpression':
        """Obtains the property that corresponds to the inverse of this property.

        Returns:
            The inverse of this property. Note that this property will not necessarily be in the simplest form.
        """
        pass

    @abstractmethod
    def get_named_property(self) -> 'OWLObjectProperty':
        """Get the named object property used in this property expression.

        Returns:
            P if this expression is either inv(P) or P.
        """
        pass

    def is_object_property_expression(self) -> bool:
        # documented in parent
        return True


class OWLDataPropertyExpression(OWLPropertyExpression, metaclass=ABCMeta):
    """A high level interface to describe different types of data properties."""
    __slots__ = ()

    def is_data_property_expression(self):
        # documented in parent
        return True


class OWLProperty(OWLPropertyExpression, OWLEntity, metaclass=ABCMeta):
    """A marker interface for properties that aren't expression i.e. named properties. By definition, properties
    are either data properties or object properties."""
    __slots__ = ()
    pass


class OWLDataProperty(OWLDataPropertyExpression, OWLProperty):
    """Represents a Data Property in the OWL 2 Specification."""
    __slots__ = '_iri'
    type_index: Final = 1004

    _iri: IRI

    def __init__(self, iri: IRI):
        """Gets an instance of OWLDataProperty that has the specified IRI.

        Args:
            iri: The IRI.
        """
        self._iri = iri

    def get_iri(self) -> IRI:
        # documented in parent
        return self._iri

    def is_owl_top_data_property(self) -> bool:
        # documented in parent
        return self.get_iri() == OWLRDFVocabulary.OWL_TOP_DATA_PROPERTY.get_iri()


class OWLObjectProperty(OWLObjectPropertyExpression, OWLProperty):
    """Represents an Object Property in the OWL 2 Specification."""
    __slots__ = '_iri'
    type_index: Final = 1002

    _iri: IRI

    def get_named_property(self) -> 'OWLObjectProperty':
        # documented in parent
        return self

    def __init__(self, iri: IRI):
        """Gets an instance of OWLObjectProperty that has the specified IRI.

        Args:
            iri: The IRI.
        """
        self._iri = iri

    def get_inverse_property(self) -> 'OWLObjectInverseOf':
        # documented in parent
        return OWLObjectInverseOf(self)

    def get_iri(self) -> IRI:
        # documented in parent
        return self._iri

    def is_owl_top_object_property(self) -> bool:
        # documented in parent
        return self.get_iri() == OWLRDFVocabulary.OWL_TOP_OBJECT_PROPERTY.get_iri()


class OWLObjectInverseOf(OWLObjectPropertyExpression):
    """Represents the inverse of a property expression (ObjectInverseOf). This can be used to refer to the inverse of
    a property, without actually naming the property. For example, consider the property hasPart, the inverse property
    of hasPart (isPartOf) can be referred to using this interface inverseOf(hasPart), which can be used in
    restrictions e.g. inverseOf(hasPart) some Car refers to the set of things that are part of at least one car."""
    __slots__ = '_inverse_property'
    type_index: Final = 1003

    _inverse_property: OWLObjectProperty

    def __init__(self, property: OWLObjectProperty):
        """Gets the inverse of an object property.

        Args:
            property: The property of which the inverse will be returned
        """
        self._inverse_property = property

    def get_inverse(self) -> OWLObjectProperty:
        """Gets the property expression that this is the inverse of.

        Returns:
            The object property expression such that this object property expression is an inverse of it.
        """
        return self._inverse_property

    def get_inverse_property(self) -> OWLObjectProperty:
        # documented in parent
        return self.get_inverse()

    def get_named_property(self) -> OWLObjectProperty:
        # documented in parent
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
    """Represents a Data Property Restriction in the OWL 2 specification."""
    __slots__ = ()

    def is_data_restriction(self) -> bool:
        # documented in parent
        return True

    pass


class OWLObjectRestriction(OWLRestriction, metaclass=ABCMeta):
    """Represents a Object Property Restriction in the OWL 2 specification."""
    __slots__ = ()

    def is_object_restriction(self) -> bool:
        # documented in parent
        return True

    @abstractmethod
    def get_property(self) -> OWLObjectPropertyExpression:
        # documented in parent
        pass


class HasFiller(Generic[_T], metaclass=ABCMeta):
    """An interface to objects that have a filler.

    Args:
        _T: filler type
    """
    __slots__ = ()

    @abstractmethod
    def get_filler(self) -> _T:
        """Gets the filler for this restriction. In the case of an object restriction this will be an individual, in
        the case of a data restriction this will be a constant (data value). For quantified restriction this will be
        a class expression or a data range.

        Returns:
            the value
        """
        pass


class OWLHasValueRestriction(Generic[_T], OWLRestriction, HasFiller[_T], metaclass=ABCMeta):
    """OWLHasValueRestriction.

    Args:
        _T: the value type
    """
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
        # documented in parent
        return self._v


class OWLQuantifiedRestriction(Generic[_T], OWLRestriction, HasFiller[_T], metaclass=ABCMeta):
    """A quantified restriction.

    Args:
        _T: value type
    """
    __slots__ = ()
    pass


class OWLQuantifiedObjectRestriction(OWLQuantifiedRestriction[OWLClassExpression], OWLObjectRestriction,
                                     metaclass=ABCMeta):
    """A quantified object restriction."""
    __slots__ = ()

    _filler: OWLClassExpression

    def __init__(self, filler: OWLClassExpression):
        self._filler = filler

    def get_filler(self) -> OWLClassExpression:
        # documented in parent (HasFiller)
        return self._filler


class OWLObjectSomeValuesFrom(OWLQuantifiedObjectRestriction):
    """Represents an ObjectSomeValuesFrom class expression in the OWL 2 Specification."""
    __slots__ = '_property', '_filler'
    type_index: Final = 3005

    def __init__(self, property: OWLObjectPropertyExpression, filler: OWLClassExpression):
        """Gets an OWLObjectSomeValuesFrom restriction

        Args:
            property: The object property that the restriction acts along.
            filler: The class expression that is the filler.

        Returns:
            An OWLObjectSomeValuesFrom restriction along the specified property with the specified filler
        """
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
        # documented in parent
        return self._property


class OWLObjectAllValuesFrom(OWLQuantifiedObjectRestriction):
    """Represents an ObjectAllValuesFrom class expression in the OWL 2 Specification."""
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
        # documented in parent
        return self._property


class OWLNaryBooleanClassExpression(OWLBooleanClassExpression, HasOperands[OWLClassExpression]):
    """OWLNaryBooleanClassExpression."""
    __slots__ = ()

    _operands: Sequence[OWLClassExpression]

    def __init__(self, operands: Iterable[OWLClassExpression]):
        """
        Args:
            operands: class expressions
        """
        self._operands = tuple(operands)

    def operands(self) -> Iterable[OWLClassExpression]:
        # documented in parent
        yield from self._operands

    def __repr__(self):
        return f'{type(self).__name__}({repr(self._operands)})'

    def __eq__(self, other):
        if type(other) == type(self):
            return self._operands == other._operands
        return NotImplemented

    def __hash__(self):
        return hash(self._operands)


class OWLObjectUnionOf(OWLNaryBooleanClassExpression):
    """Represents an ObjectUnionOf class expression in the OWL 2 Specification."""
    __slots__ = '_operands'
    type_index: Final = 3002

    _operands: Sequence[OWLClassExpression]


class OWLObjectIntersectionOf(OWLNaryBooleanClassExpression):
    """Represents an OWLObjectIntersectionOf class expression in the OWL 2 Specification."""
    __slots__ = '_operands'
    type_index: Final = 3001

    _operands: Sequence[OWLClassExpression]


class HasCardinality(metaclass=ABCMeta):
    """An interface to objects that have a cardinality."""
    __slots__ = ()

    @abstractmethod
    def get_cardinality(self) -> int:
        """Gets the cardinality of a restriction.

        Returns:
            The cardinality. A non-negative integer.
        """
        pass


_F = TypeVar('_F', bound=OWLPropertyRange)  #:


class OWLCardinalityRestriction(Generic[_F], OWLQuantifiedRestriction[_F], HasCardinality, metaclass=ABCMeta):
    """.

    Args:
        _F: type of filler
    """
    __slots__ = ()

    _cardinality: int
    _filler: _F

    def __init__(self, cardinality: int, filler: _F):
        self._cardinality = cardinality
        self._filler = filler

    def get_cardinality(self) -> int:
        # documented in parent
        return self._cardinality

    def get_filler(self) -> _F:
        # documented in parent
        return self._filler


class OWLObjectCardinalityRestriction(OWLCardinalityRestriction[OWLClassExpression], OWLQuantifiedObjectRestriction):
    __slots__ = ()

    _property: OWLObjectPropertyExpression

    @abstractmethod
    def __init__(self, cardinality: int, property: OWLObjectPropertyExpression, filler: OWLClassExpression):
        super().__init__(cardinality, filler)
        self._property = property

    def get_property(self) -> OWLObjectPropertyExpression:
        # documented in parent
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
        return hash((self._property, self._cardinality, self._filler))


class OWLObjectMinCardinality(OWLObjectCardinalityRestriction):
    """Represents a ObjectMinCardinality restriction in the OWL 2 Specification."""
    __slots__ = '_cardinality', '_filler', '_property'
    type_index: Final = 3008

    def __init__(self, cardinality: int, property: OWLObjectPropertyExpression, filler: OWLClassExpression):
        """
        Args:
            cardinality: Cannot be negative.
            property: The property that the restriction acts along.
            filler: class expression for restriction

        Returns:
            an ObjectMinCardinality on the specified property
        """
        super().__init__(cardinality, property, filler)


class OWLObjectMaxCardinality(OWLObjectCardinalityRestriction):
    """Represents a ObjectMaxCardinality restriction in the OWL 2 Specification."""
    __slots__ = '_cardinality', '_filler', '_property'
    type_index: Final = 3010

    def __init__(self, cardinality: int, property: OWLObjectPropertyExpression, filler: OWLClassExpression):
        """
        Args:
            cardinality: Cannot be negative.
            property: The property that the restriction acts along.
            filler: class expression for restriction

        Returns:
            an ObjectMaxCardinality on the specified property
        """
        super().__init__(cardinality, property, filler)


class OWLObjectExactCardinality(OWLObjectCardinalityRestriction):
    """Represents an ObjectExactCardinality  restriction in the OWL 2 Specification."""
    __slots__ = '_cardinality', '_filler', '_property'
    type_index: Final = 3009

    def __init__(self, cardinality: int, property: OWLObjectPropertyExpression, filler: OWLClassExpression):
        """
        Args:
            cardinality: Cannot be negative.
            property: The property that the restriction acts along.
            filler: class expression for restriction

        Returns:
            an ObjectExactCardinality on the specified property
        """
        super().__init__(cardinality, property, filler)

    def as_intersection_of_min_max(self) -> OWLObjectIntersectionOf:
        """Obtains an equivalent form that is a conjunction of a min cardinality and max cardinality restriction.

        Returns:
            The semantically equivalent but structurally simpler form (= 1 R C) = >= 1 R C and <= 1 R C
        """
        args = self.get_property(), self.get_cardinality(), self.get_filler()
        return OWLObjectIntersectionOf((OWLObjectMinCardinality(*args), OWLObjectMaxCardinality(*args)))


class OWLObjectHasSelf(OWLObjectRestriction):
    """Represents an ObjectHasSelf class expression in the OWL 2 Specification."""
    __slots__ = '_property'
    type_index: Final = 3011

    _property: OWLObjectPropertyExpression

    def __init__(self, property: OWLObjectPropertyExpression):
        """Object has self restriction

        Args:
            property: The property that the restriction acts along.

        Returns:
            a ObjectHasSelf class expression on the specified property
        """
        self._property = property

    def get_property(self) -> OWLObjectPropertyExpression:
        # documented in parent
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
    """Represents a named or anonymous individual."""
    __slots__ = ()
    pass


class OWLObjectHasValue(OWLHasValueRestriction[OWLIndividual], OWLObjectRestriction):
    """Represents an ObjectHasValue class expression in the OWL 2 Specification."""
    __slots__ = '_property', '_v'
    type_index: Final = 3007

    _property: OWLObjectPropertyExpression
    _v: OWLIndividual

    def __init__(self, property: OWLObjectPropertyExpression, individual: OWLIndividual):
        """
        Args:
            property: The property that the restriction acts along.
            individual: individual for restriction

        Returns:
            a HasValue restriction with specified property and value
        """
        super().__init__(individual)
        self._property = property

    def get_property(self) -> OWLObjectPropertyExpression:
        # documented in parent
        return self._property

    def as_some_values_from(self) -> OWLClassExpression:
        """A convenience method that obtains this restriction as an existential restriction with a nominal filler.

        Returns:
            The existential equivalent of this value restriction. simp(HasValue(p a)) = some(p {a})
        """
        return OWLObjectSomeValuesFrom(self.get_property(), OWLObjectOneOf(self.get_filler()))

    def __repr__(self):
        return f'OWLObjectHasValue(property={self.get_property()}, individual={self._v})'


class OWLObjectOneOf(OWLAnonymousClassExpression, HasOperands[OWLIndividual]):
    """Represents an ObjectOneOf class expression in the OWL 2 Specification."""
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
        """Gets the individuals that are in the oneOf. These individuals represent the exact instances (extension)
         of this class expression.

         Returns:
             The individuals that are the values of this {@code ObjectOneOf} class expression.
        """
        yield from self._values

    def operands(self) -> Iterable[OWLIndividual]:
        # documented in parent
        yield from self.individuals()

    def as_object_union_of(self) -> OWLClassExpression:
        """Simplifies this enumeration to a union of singleton nominals.

        Returns:
            This enumeration in a more standard DL form.
            simp({a}) = {a} simp({a0, ... , {an}) = unionOf({a0}, ... , {an})
        """
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
    """Represents a Named Individual in the OWL 2 Specification."""
    __slots__ = '_iri'
    type_index: Final = 1005

    _iri: IRI

    def __init__(self, iri: IRI):
        """Gets an instance of OWLNamedIndividual that has the specified IRI.

        Args:
            iri: The IRI.

        Returns:
            An OWLNamedIndividual that has the specified IRI.
        """
        self._iri = iri

    def get_iri(self) -> IRI:
        # documented in parent
        return self._iri


_M = TypeVar('_M', bound='OWLOntologyManager')  #:


class OWLOntologyID:
    """An object that identifies an ontology. Since OWL 2, ontologies do not have to have an ontology IRI, or if they
    have an ontology IRI then they can optionally also have a version IRI. Instances of this OWLOntologyID class bundle
    identifying information of an ontology together. If an ontology doesn't have an ontology IRI then we say that it is
    "anonymous".
    """
    __slots__ = '_ontology_iri', '_version_iri'

    _ontology_iri: Optional[IRI]
    _version_iri: Optional[IRI]

    def __init__(self, ontology_iri: Optional[IRI] = None, version_iri: Optional[IRI] = None):
        """Constructs an ontology identifier specifying the ontology IRI and version IRI.

        Args:
            ontology_iri: The ontology IRI (optional)
            version_iri: The version IRI (must be None if no ontology_iri is provided)
        """
        self._ontology_iri = ontology_iri
        self._version_iri = version_iri

    def get_ontology_iri(self) -> Optional[IRI]:
        """Gets the ontology IRI.

        Returns:
            Ontology IRI. If the ontology is anonymous, it will return None
        """
        return self._ontology_iri

    def get_version_iri(self) -> Optional[IRI]:
        """Gets the version IRI.

        Returns:
            Version IRI or None
        """
        return self._version_iri

    def get_default_document_iri(self) -> Optional[IRI]:
        """Gets the IRI which is used as a default for the document that contain a representation of an ontology with
        this ID. This will be the version IRI if there is an ontology IRI and version IRI, else it will be the ontology
        IRI if there is an ontology IRI but no version IRI, else it will be None if there is no ontology IRI. See
        Ontology Documents in the OWL 2 Structural Specification.

        Returns:
            the IRI that can be used as a default for an ontology document, or None.
        """
        if self._ontology_iri is not None:
            if self._version_iri is not None:
                return self._version_iri
        return self._ontology_iri

    def is_anonymous(self) -> bool:
        return self._ontology_iri is None

    def __repr__(self):
        return f"OWLOntologyID({repr(self._ontology_iri)}, {repr(self._version_iri)})"

    def __eq__(self, other):
        if type(other) is type(self):
            return self._ontology_iri == other._ontology_iri and self._version_iri == other._version_iri
        return NotImplemented


class OWLAxiom(OWLObject, metaclass=ABCMeta):
    """Represents Axioms in the OWL 2 Specification.

    An OWL ontology contains a set of axioms. These axioms can be annotation axioms, declaration axioms, imports axioms
    or logical axioms
    """
    __slots__ = ()
    # TODO: XXX
    pass


class OWLDatatype(OWLEntity, OWLDataRange):
    """Represents a Datatype (named data range) in the OWL 2 Specification."""
    __slots__ = '_iri'

    type_index: Final = 4001

    _iri: IRI

    def __init__(self, iri: Union[IRI, HasIRI]):
        """Gets an instance of OWLDatatype that has the specified IRI.

        Args:
            iri: The IRI.
        """
        if isinstance(iri, HasIRI):
            self._iri = iri.get_iri()
        else:
            assert isinstance(iri, IRI)
            self._iri = iri

    def get_iri(self) -> 'IRI':
        # documented in parent
        return self._iri


class OWLDatatypeRestriction(OWLDataRange):
    """Represents a DatatypeRestriction data range in the OWL 2 Specification."""
    __slots__ = '_type', '_facet_restrictions'

    type_index: Final = 4006

    _type: OWLDatatype
    _facet_restrictions: Sequence['OWLFacetRestriction']

    def __init__(self, type_: OWLDatatype, facet_restrictions: Union['OWLFacetRestriction',
                                                                     Iterable['OWLFacetRestriction']]):
        self._type = type_
        if isinstance(facet_restrictions, OWLFacetRestriction):
            facet_restrictions = facet_restrictions,
        self._facet_restrictions = tuple(facet_restrictions)

    def get_datatype(self) -> OWLDatatype:
        return self._type

    def get_facet_restrictions(self) -> Sequence['OWLFacetRestriction']:
        return self._facet_restrictions

    def __eq__(self, other):
        if type(other) is type(self):
            return self._type == other._type \
                   and self._facet_restrictions == other._facet_restrictions
        return NotImplemented

    def __hash__(self):
        return hash((self._type, self._facet_restrictions))

    def __repr__(self):
        return f'OWLDatatypeRestriction({repr(self._type)}, {repr(self._facet_restrictions)})'


class OWLFacetRestriction(OWLObject):
    """A facet restriction is used to restrict a particular datatype."""

    __slots__ = '_facet', '_literal'

    type_index: Final = 4007

    _facet: OWLFacet
    _literal: 'OWLLiteral'

    def __init__(self, facet: OWLFacet, literal: Literals):
        self._facet = facet
        if isinstance(literal, OWLLiteral):
            self._literal = literal
        else:
            self._literal = OWLLiteral(literal)

    def get_facet(self) -> OWLFacet:
        return self._facet

    def get_facet_value(self) -> 'OWLLiteral':
        return self._literal

    def __eq__(self, other):
        if type(other) is type(self):
            return self._facet == other._facet and self._literal == other._literal
        return NotImplemented

    def __hash__(self):
        return hash((self._facet, self._literal))

    def __repr__(self):
        return f'OWLFacetRestriction({self._facet}, {repr(self._literal)})'


class OWLLiteral(OWLAnnotationValue, metaclass=ABCMeta):
    """Represents a Literal in the OWL 2 Specification."""
    __slots__ = ()

    type_index: Final = 4008

    def __new__(cls, value):
        """Convenience method that obtains a literal

        Args:
            value: The value of the literal
        """
        if isinstance(value, bool):
            return super().__new__(_OWLLiteralImplBoolean)
        elif isinstance(value, int):
            return super().__new__(_OWLLiteralImplInteger)
        elif isinstance(value, float):
            return super().__new__(_OWLLiteralImplDouble)
        elif isinstance(value, date):
            return super().__new__(_OWLLiteralImplDate)
        elif isinstance(value, datetime):
            return super().__new__(_OWLLiteralImplDateTime)
        elif isinstance(value, Timedelta):
            return super().__new__(_OWLLiteralImplDuration)
        # TODO XXX
        raise NotImplementedError

    def get_literal(self) -> str:
        """Gets the lexical value of this literal. Note that the language tag is not included.

        Returns:
            The lexical value of this literal.
        """
        return str(self._v)

    def is_boolean(self) -> bool:
        """Whether this literal is typed as boolean"""
        return False

    def parse_boolean(self) -> bool:
        """Parses the lexical value of this literal into a bool. The lexical value of this literal should be in the
        lexical space of the boolean datatype ("http://www.w3.org/2001/XMLSchema#boolean").

        Returns:
            A bool value that is represented by this literal.
        """
        raise ValueError

    def is_double(self) -> bool:
        """Whether this literal is typed as double"""
        return False

    def parse_double(self) -> float:
        """Parses the lexical value of this literal into a double. The lexical value of this literal should be in the
        lexical space of the double datatype ("http://www.w3.org/2001/XMLSchema#double").

        Returns:
            A double value that is represented by this literal.
        """
        raise ValueError

    def is_integer(self) -> bool:
        """Whether this literal is typed as integer"""
        return False

    def parse_integer(self) -> int:
        """Parses the lexical value of this literal into an integer. The lexical value of this literal should be in the
        lexical space of the integer datatype ("http://www.w3.org/2001/XMLSchema#integer").

        Returns:
            An integer value that is represented by this literal.
        """
        raise ValueError

    def is_date(self) -> bool:
        """Whether this literal is typed as date"""
        return False

    def parse_date(self) -> date:
        """Parses the lexical value of this literal into a date. The lexical value of this literal should be in the
        lexical space of the date datatype ("http://www.w3.org/2001/XMLSchema#date").

        Returns:
            A date value that is represented by this literal.
        """
        raise ValueError

    def is_datetime(self) -> bool:
        """Whether this literal is typed as dateTime"""
        return False

    def parse_datetime(self) -> datetime:
        """Parses the lexical value of this literal into a datetime. The lexical value of this literal should be in the
        lexical space of the dateTime datatype ("http://www.w3.org/2001/XMLSchema#dateTime").

        Returns:
            A datetime value that is represented by this literal.
        """
        raise ValueError

    def is_duration(self) -> bool:
        """Whether this literal is typed as duration"""
        return False

    def parse_duration(self) -> Timedelta:
        """Parses the lexical value of this literal into a Timedelta. The lexical value of this literal should be in the
        lexical space of the duration datatype ("http://www.w3.org/2001/XMLSchema#duration").

        Returns:
            A Timedelta value that is represented by this literal.
        """
        raise ValueError

    # noinspection PyMethodMayBeStatic
    def is_literal(self) -> bool:
        # documented in parent
        return True

    def as_literal(self) -> 'OWLLiteral':
        # documented in parent
        return self

    def to_python(self) -> Literals:
        return self._v

    @abstractmethod
    def get_datatype(self) -> OWLDatatype:
        """Gets the OWLDatatype which types this literal.

        Returns:
            The OWLDatatype that types this literal.
        """
        pass


@total_ordering
class _OWLLiteralImplDouble(OWLLiteral):
    __slots__ = '_v'

    _v: float

    def __init__(self, value):
        assert isinstance(value, float)
        self._v = value

    def __eq__(self, other):
        if type(other) is type(self):
            return self._v == other._v
        return NotImplemented

    def __lt__(self, other):
        if type(other) is type(self):
            return self._v < other._v
        return NotImplemented

    def __hash__(self):
        return hash(self._v)

    def __repr__(self):
        return f'OWLLiteral({self._v})'

    def is_double(self) -> bool:
        return True

    def parse_double(self) -> float:
        # documented in parent
        return self._v

    # noinspection PyMethodMayBeStatic
    def get_datatype(self) -> OWLDatatype:
        # documented in parent
        return DoubleOWLDatatype


@total_ordering
class _OWLLiteralImplInteger(OWLLiteral):
    __slots__ = '_v'

    _v: int

    def __init__(self, value):
        assert isinstance(value, int)
        self._v = value

    def __eq__(self, other):
        if type(other) is type(self):
            return self._v == other._v
        return NotImplemented

    def __lt__(self, other):
        if type(other) is type(self):
            return self._v < other._v
        return NotImplemented

    def __hash__(self):
        return hash(self._v)

    def __repr__(self):
        return f'OWLLiteral({self._v})'

    def is_integer(self) -> bool:
        return True

    def parse_integer(self) -> int:
        # documented in parent
        return self._v

    # noinspection PyMethodMayBeStatic
    def get_datatype(self) -> OWLDatatype:
        # documented in parent
        return IntegerOWLDatatype


class _OWLLiteralImplBoolean(OWLLiteral):
    __slots__ = '_v'

    _v: bool

    def __init__(self, value):
        assert isinstance(value, bool)
        self._v = value

    def __eq__(self, other):
        if type(other) is type(self):
            return self._v == other._v
        return NotImplemented

    def __hash__(self):
        return hash(self._v)

    def __repr__(self):
        return f'OWLLiteral({self._v})'

    def is_boolean(self) -> bool:
        return True

    def parse_boolean(self) -> bool:
        # documented in parent
        return self._v

    # noinspection PyMethodMayBeStatic
    def get_datatype(self) -> OWLDatatype:
        # documented in parent
        return BooleanOWLDatatype


@total_ordering
class _OWLLiteralImplDate(OWLLiteral):
    __slots__ = '_v'

    _v: date

    def __init__(self, value):
        assert isinstance(value, date)
        self._v = value

    def __eq__(self, other):
        if type(other) is type(self):
            return self._v == other._v
        return NotImplemented

    def __lt__(self, other):
        if type(other) is type(self):
            return self._v < other._v
        return NotImplemented

    def __hash__(self):
        return hash(self._v)

    def __repr__(self):
        return f'OWLLiteral({self._v})'

    def is_date(self) -> bool:
        return True

    def parse_date(self) -> date:
        # documented in parent
        return self._v

    # noinspection PyMethodMayBeStatic
    def get_datatype(self) -> OWLDatatype:
        # documented in parent
        return DateOWLDatatype


@total_ordering
class _OWLLiteralImplDateTime(OWLLiteral):
    __slots__ = '_v'

    _v: datetime

    def __init__(self, value):
        assert isinstance(value, datetime)
        self._v = value

    def __eq__(self, other):
        if type(other) is type(self):
            return self._v == other._v
        return NotImplemented

    def __lt__(self, other):
        if type(other) is type(self):
            return self._v < other._v
        return NotImplemented

    def __hash__(self):
        return hash(self._v)

    def __repr__(self):
        return f'OWLLiteral({self._v})'

    def is_datetime(self) -> bool:
        return True

    def parse_datetime(self) -> datetime:
        # documented in parent
        return self._v

    # noinspection PyMethodMayBeStatic
    def get_datatype(self) -> OWLDatatype:
        # documented in parent
        return DateTimeOWLDatatype


@total_ordering
class _OWLLiteralImplDuration(OWLLiteral):
    __slots__ = '_v'

    _v: Timedelta

    def __init__(self, value):
        assert isinstance(value, Timedelta)
        self._v = value

    def __eq__(self, other):
        if type(other) is type(self):
            return self._v == other._v
        return NotImplemented

    def __lt__(self, other):
        if type(other) is type(self):
            return self._v < other._v
        return NotImplemented

    def __hash__(self):
        return hash(self._v)

    def __repr__(self):
        return f'OWLLiteral({self._v})'

    def is_duration(self) -> bool:
        return True

    def parse_duration(self) -> Timedelta:
        # documented in parent
        return self._v

    # noinspection PyMethodMayBeStatic
    def get_datatype(self) -> OWLDatatype:
        # documented in parent
        return DurationOWLDatatype


class OWLQuantifiedDataRestriction(OWLQuantifiedRestriction[OWLDataRange],
                                   OWLDataRestriction, metaclass=ABCMeta):
    """A quantified data restriction."""
    __slots__ = ()

    _filler: OWLDataRange

    def __init__(self, filler: OWLDataRange):
        self._filler = filler

    def get_filler(self) -> OWLDataRange:
        # documented in parent (HasFiller)
        return self._filler


class OWLDataCardinalityRestriction(OWLCardinalityRestriction[OWLDataRange],
                                    OWLQuantifiedDataRestriction,
                                    OWLDataRestriction, metaclass=ABCMeta):
    """Represents Data Property Cardinality Restrictions in the OWL 2 specification"""
    __slots__ = ()

    _property: OWLDataPropertyExpression

    @abstractmethod
    def __init__(self, cardinality: int, property: OWLDataPropertyExpression, filler: OWLDataRange):
        super().__init__(cardinality, filler)
        self._property = property

    def get_property(self) -> OWLDataPropertyExpression:
        # documented in parent
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
        return hash((self._property, self._cardinality, self._filler))


# TODO: a big todo plus intermediate classes (missing)

# class OWLAnnotation(metaclass=ABCMeta):
#     """Annotations are used in the various types of annotation axioms, which bind annotations to their subjects (i.e.
#      axioms or declarations)."""
#     type_index: Final = 5001
#
#
# class OWLAnnotationProperty(metaclass=ABCMeta):
#     """Represents an AnnotationProperty in the OWL 2 specification."""
#     type_index: Final = 1006
#
#
# class OWLAnonymousIndividual(OWLIndividual, OWLAnnotationValue, OWLAnnotationSubject, metaclass=ABCMeta):
#     """Represents Anonymous Individuals in the OWL 2 Specification."""
#     type_index: Final = 1007
#
#
# class OWLAxiom(metaclass=ABCMeta):
#     """Represents Axioms in the OWL 2 Specification.
#
#     An OWL ontology contains a set of axioms. These axioms can be annotation axioms, declaration axioms, imports
#     axioms or logical axioms
#     """
#     type_index: Final = 2000 + get_axiom_type().get_index()


class OWLDataAllValuesFrom(OWLQuantifiedDataRestriction):
    """Represents DataAllValuesFrom class expressions in the OWL 2 Specification."""
    __slots__ = '_property'

    type_index: Final = 3013

    _property: OWLDataPropertyExpression

    def __init__(self, property: OWLDataPropertyExpression, filler: OWLDataRange):
        """Gets an OWLDataAllValuesFrom restriction

        Args:
            property: The data property that the restriction acts along.
            filler: The data range that is the filler.

        Returns:
            An OWLDataAllValuesFrom restriction along the specified property with the specified filler
        """
        super().__init__(filler)
        self._property = property

    def __repr__(self):
        return f"OWLDataAllValuesFrom(property={repr(self._property)},filler={repr(self._filler)})"

    def __eq__(self, other):
        if type(other) is type(self):
            return self._filler == other._filler and self._property == other._property
        return NotImplemented

    def __hash__(self):
        return hash((self._filler, self._property))

    def get_property(self) -> OWLDataPropertyExpression:
        # documented in parent
        return self._property


class OWLDataComplementOf(OWLDataRange):
    """Represents DataComplementOf in the OWL 2 Specification."""
    type_index: Final = 4002

    _data_range: OWLDataRange

    def __init__(self, data_range: OWLDataRange):
        """
        Args:
            data_range: data range to complement
        """
        self._data_range = data_range

    def get_data_range(self) -> OWLDataRange:
        """
        Returns:
            the wrapped data range
        """
        return self._data_range

    def __repr__(self):
        return f"OWLDataComplementOf({repr(self._data_range)})"

    def __eq__(self, other):
        if type(other) is type(self):
            return self._data_range == other._data_range
        return NotImplemented

    def __hash__(self):
        return hash(self._data_range)


class OWLDataExactCardinality(OWLDataCardinalityRestriction):
    """Represents DataExactCardinality restrictions in the OWL 2 Specification."""
    __slots__ = '_cardinality', '_filler', '_property'

    type_index: Final = 3016

    def __init__(self, cardinality: int, property: OWLDataPropertyExpression, filler: OWLDataRange):
        """
        Args:
            cardinality: Cannot be negative.
            property: The property that the restriction acts along.
            filler: data range for restriction

        Returns:
            a DataExactCardinality on the specified property
        """
        super().__init__(cardinality, property, filler)

    def as_intersection_of_min_max(self) -> OWLObjectIntersectionOf:
        """Obtains an equivalent form that is a conjunction of a min cardinality and max cardinality restriction.

        Returns:
            The semantically equivalent but structurally simpler form (= 1 R D) = >= 1 R D and <= 1 R D
        """
        args = self.get_property(), self.get_cardinality(), self.get_filler()
        return OWLObjectIntersectionOf((OWLDataMinCardinality(*args), OWLDataMaxCardinality(*args)))


class OWLDataHasValue(OWLHasValueRestriction[OWLLiteral], OWLDataRestriction):
    """Represents DataHasValue restrictions in the OWL 2 Specification."""
    __slots__ = '_property'

    type_index: Final = 3014

    _property: OWLDataPropertyExpression

    def __init__(self, property: OWLDataPropertyExpression, value: OWLLiteral):
        """Gets an OWLDataHasValue restriction

        Args:
            property: The data property that the restriction acts along.
            filler: The literal value

        Returns:
            An OWLDataHasValue restriction along the specified property with the specified literal
        """
        super().__init__(value)
        self._property = property

    def __repr__(self):
        return f"OWLDataHasValue(property={repr(self._property)},value={repr(self._v)})"

    def __eq__(self, other):
        if type(other) is type(self):
            return self._v == other._v and self._property == other._property
        return NotImplemented

    def __hash__(self):
        return hash((self._v, self._property))

    def as_some_values_from(self) -> OWLClassExpression:
        """A convenience method that obtains this restriction as an existential restriction with a nominal filler.

        Returns:
            The existential equivalent of this value restriction. simp(HasValue(p a)) = some(p {a})
        """
        return OWLDataSomeValuesFrom(self.get_property(), OWLDataOneOf(self.get_filler()))

    def get_property(self) -> OWLDataPropertyExpression:
        # documented in parent
        return self._property


class OWLDataMaxCardinality(OWLDataCardinalityRestriction):
    """Represents DataMaxCardinality restrictions in the OWL 2 Specification."""
    __slots__ = '_cardinality', '_filler', '_property'

    type_index: Final = 3017

    def __init__(self, cardinality: int, property: OWLDataPropertyExpression, filler: OWLDataRange):
        """
        Args:
            cardinality: Cannot be negative.
            property: The property that the restriction acts along.
            filler: data range for restriction

        Returns:
            a DataMaxCardinality on the specified property
        """
        super().__init__(cardinality, property, filler)


class OWLDataMinCardinality(OWLDataCardinalityRestriction):
    """Represents DataMinCardinality restrictions in the OWL 2 Specification."""
    __slots__ = '_cardinality', '_filler', '_property'

    type_index: Final = 3015

    def __init__(self, cardinality: int, property: OWLDataPropertyExpression, filler: OWLDataRange):
        """
        Args:
            cardinality: Cannot be negative.
            property: The property that the restriction acts along.
            filler: data range for restriction

        Returns:
            a DataMinCardinality on the specified property
        """
        super().__init__(cardinality, property, filler)


class OWLDataOneOf(OWLDataRange, HasOperands[OWLLiteral]):
    """Represents DataOneOf in the OWL 2 Specification."""
    type_index: Final = 4003

    _values: Sequence[OWLLiteral]

    def __init__(self, values: Union[OWLLiteral, Iterable[OWLLiteral]]):
        if isinstance(values, OWLLiteral):
            self._values = values,
        else:
            for _ in values:
                assert isinstance(_, OWLLiteral)
            self._values = tuple(values)

    def values(self) -> Iterable[OWLLiteral]:
        """Gets the values that are in the oneOf.

         Returns:
             The values of this {@code DataOneOf} class expression.
        """
        yield from self._values

    def operands(self) -> Iterable[OWLLiteral]:
        # documented in parent
        yield from self.values()

    def __hash__(self):
        return hash(self._values)

    def __eq__(self, other):
        if type(other) == type(self):
            return self._values == other._values
        return NotImplemented

    def __repr__(self):
        return f'OWLDataOneOf({self._values})'


class OWLDataSomeValuesFrom(OWLQuantifiedDataRestriction):
    """Represents a DataSomeValuesFrom restriction in the OWL 2 Specification."""
    __slots__ = '_property'

    type_index: Final = 3012

    _property: OWLDataPropertyExpression

    def __init__(self, property: OWLDataPropertyExpression, filler: OWLDataRange):
        """Gets an OWLDataSomeValuesFrom restriction

        Args:
            property: The data property that the restriction acts along.
            filler: The data range that is the filler.

        Returns:
            An OWLDataSomeValuesFrom restriction along the specified property with the specified filler
        """
        super().__init__(filler)
        self._property = property

    def __repr__(self):
        return f"OWLDataSomeValuesFrom(property={repr(self._property)},filler={repr(self._filler)})"

    def __eq__(self, other):
        if type(other) is type(self):
            return self._filler == other._filler and self._property == other._property
        return NotImplemented

    def __hash__(self):
        return hash((self._filler, self._property))

    def get_property(self) -> OWLDataPropertyExpression:
        # documented in parent
        return self._property


class OWLNaryDataRange(OWLDataRange, HasOperands[OWLDataRange]):
    """OWLNaryDataRange."""
    __slots__ = ()

    _operands: Sequence[OWLDataRange]

    def __init__(self, operands: Iterable[OWLDataRange]):
        """
        Args:
            operands: data ranges
        """
        self._operands = tuple(operands)

    def operands(self) -> Sequence[OWLDataRange]:
        # documented in parent
        yield from self._operands

    def __repr__(self):
        return f'{type(self).__name__}({repr(self._operands)})'

    def __eq__(self, other):
        if type(other) == type(self):
            return self._operands == other._operands
        return NotImplemented

    def __hash__(self):
        return hash(self._operands)


class OWLDataUnionOf(OWLNaryDataRange):
    """Represents a DataUnionOf data range in the OWL 2 Specification."""
    __slots__ = '_operands'
    type_index: Final = 4005

    _operands: Sequence[OWLDataRange]


class OWLDataIntersectionOf(OWLNaryDataRange):
    """Represents DataIntersectionOf  in the OWL 2 Specification."""
    __slots__ = '_operands'
    type_index: Final = 4004

    _operands: Sequence[OWLDataRange]


class OWLImportsDeclaration(HasIRI):
    """Represents an import statement in an ontology."""
    __slots__ = '_iri'

    def __init__(self, import_iri: IRI):
        """
        Args:
            import_import_iri: imported ontology

        Returns:
            an imports declaration
        """
        self._iri = import_iri

    def get_iri(self) -> IRI:
        """Gets the import IRI.

        Returns:
            The import IRI that points to the ontology to be imported. The imported ontology might have this IRI as
            its ontology IRI but this is not mandated. For example, an ontology with a non resolvable ontology IRI
            can be deployed at a resolvable URL.
        """
        return self._iri


class OWLLogicalAxiom(OWLAxiom, metaclass=ABCMeta):
    """A base interface of all axioms that affect the logical meaning of an ontology. This excludes declaration axioms
    (including imports declarations) and annotation axioms.
    """
    __slots__ = ()
    pass


class OWLClassAxiom(OWLLogicalAxiom, metaclass=ABCMeta):
    __slots__ = ()
    pass


class OWLNaryAxiom(Generic[_C], OWLAxiom, metaclass=ABCMeta):
    """Represents an axiom that contains two or more operands that could also be represented with multiple pairwise
    axioms.

    Args:
        _C: class of contained objects
    """
    __slots__ = ()
    pass


# noinspection PyUnresolvedReferences
# noinspection PyDunderSlots
class OWLNaryClassAxiom(OWLClassAxiom, OWLNaryAxiom[OWLClassExpression], metaclass=ABCMeta):
    __slots__ = ()

    _class_expressions: List[OWLClassExpression]

    @abstractmethod
    def __init__(self, class_expressions: List[OWLClassExpression]):
        self._class_expressions = [*class_expressions]

    def class_expressions(self) -> Iterable[OWLClassExpression]:
        """Gets all of the top level class expressions that appear in this axiom.

        Returns:
            Sorted stream of class expressions that appear in the axiom.
        """
        yield from self._class_expressions

    def __eq__(self, other):
        if type(other) is type(self):
            return self._class_expressions == other._class_expressions
        return NotImplemented

    def __hash__(self):
        return hash(self._class_expressions)

    def __repr__(self):
        return f'{type(self).__name__}({self._class_expressions})'


class OWLEquivalentClassesAxiom(OWLNaryClassAxiom):
    """Represents an EquivalentClasses axiom in the OWL 2 Specification."""
    __slots__ = '_class_expressions'

    def __init__(self, cls_a: OWLClassExpression, cls_b: OWLClassExpression):
        """Get an equivalent classes axiom with specified operands and no annotations

        Args:
            cls_a: one class for equivalence
            cls_b: one class for equivalence
        """
        super().__init__([cls_a, cls_b])


class OWLAnnotationAxiom(OWLAxiom, metaclass=ABCMeta):
    """A super interface for annotation axioms."""
    __slots__ = ()
    # TODO: XXX
    pass


class OWLAnnotationProperty(OWLProperty):
    """Represents an AnnotationProperty in the OWL 2 specification."""
    __slots__ = '_iri'

    def __init__(self, iri: IRI):
        """Get a new OWLAnnotationProperty object

        Args:
            iri: new OWLAnnotationProperty IRI
        """
        self._iri = iri

    def get_iri(self) -> IRI:
        # documented in parent
        return self._iri


class OWLAnnotation(OWLObject):
    """Annotations are used in the various types of annotation axioms, which bind annotations to their subjects
    (i.e. axioms or declarations)."""
    __slots__ = '_property', '_value'

    def __init__(self, property: OWLAnnotationProperty, value: OWLAnnotationValue):
        """Gets an annotation

        Args:
            property: the annotation property.
            value: The annotation value.
        """
        self._property = property
        self._value = value

    def get_property(self) -> OWLAnnotationProperty:
        """Gets the property that this annotation acts along.

        Returns:
            The annotation property
        """
        return self._property

    def get_value(self) -> OWLAnnotationValue:
        """Gets the annotation value. The type of value will depend upon the type of the annotation e.g. whether the
        annotation is an OWLLiteral, an IRI or an OWLAnonymousIndividual.

        Returns:
            The annotation value.
        """
        return self._value

    def __eq__(self, other):
        if type(other) is type(self):
            return self._property == other._property and self._value == other._value
        return NotImplemented

    def __hash__(self):
        return hash((self._property, self._value))

    def __repr__(self):
        return f'OWLAnnotation({self._property}, {self._value})'


class OWLAnnotationAssertionAxiom(OWLAnnotationAxiom):
    """Represents AnnotationAssertion axioms in the OWL 2 specification."""
    __slots__ = '_subject', '_annotation'

    def __init__(self, subject: OWLAnnotationSubject, annotation: OWLAnnotation):
        """Get an annotation assertion axiom - with annotations

        Args:
            subject: subject
            annotation: annotation
        """
        assert isinstance(subject, OWLAnnotationSubject)
        assert isinstance(annotation, OWLAnnotation)

        self._subject = subject
        self._annotation = annotation

    def get_subject(self) -> OWLAnnotationSubject:
        """Gets the subject of this object.

        Returns:
            The subject
        """
        return self._subject

    def get_property(self) -> OWLAnnotationProperty:
        """Gets the property.

        Returns:
            The property.
        """
        return self._annotation.get_property()

    def get_value(self) -> OWLAnnotationValue:
        """Gets the annotation value. This is either an IRI, an OWLAnonymousIndividual or an OWLLiteral.

        Returns:
            The annotation value.
        """
        return self._annotation.get_value()

    def __eq__(self, other):
        if type(other) is type(self):
            return self._subject == other._subject and self._annotation == other._annotation
        return NotImplemented

    def __hash__(self):
        return hash((self._subject, self._annotation))

    def __repr__(self):
        return f'OWLAnnotationAssertionAxiom({self._subject}, {self._annotation})'


class OWLDataPropertyRangeAxiom(OWLLogicalAxiom):
    __slots__ = '_property', '_range', '_annotations'
    _property: OWLDataPropertyExpression
    _range: OWLDataRange
    _annotations: Optional[List[OWLAnnotation]]

    def __init__(self, property: OWLDataPropertyExpression, range: OWLDataRange,
                 annotations: Optional[Iterable[OWLAnnotation]] = None):
        self._property = property
        self._range = range
        if annotations is not None:
            self._annotations = list(annotations)

    def get_range(self) -> OWLDataRange:
        return self._range

    def __eq__(self, other):
        if type(other) is type(self):
            return self._property == other._property and self._range == other._range \
                   and self._annotations == other._annotation
        return NotImplemented

    def __hash__(self):
        return hash((self._property, self._range, self._annotations))

    def __repr__(self):
        return f'OWLDataPropertyRangeAxiom({self._property}, {self._range}, {self._annotations})'


class OWLOntology(OWLObject, metaclass=ABCMeta):
    """Represents an OWL 2 Ontology  in the OWL 2 specification.

    An OWLOntology consists of a possibly empty set of OWLAxioms and a possibly empty set of OWLAnnotations.
    An ontology can have an ontology IRI which can be used to identify the ontology. If it has an ontology IRI then
    it may also have an ontology version IRI. Since OWL 2, an ontology need not have an ontology IRI. (See the OWL 2
    Structural Specification)

    An ontology cannot be modified directly. Changes must be applied via its OWLOntologyManager.
    """
    __slots__ = ()
    type_index: Final = 1

    @abstractmethod
    def classes_in_signature(self) -> Iterable[OWLClass]:
        """Gets the classes in the signature of this object.

        Returns:
            Classes in the signature of this object
        """
        pass

    @abstractmethod
    def data_properties_in_signature(self) -> Iterable[OWLDataProperty]:
        """Get the data properties that are in the signature of this object.

        Returns:
            Data properties that are in the signature of this object
        """
        pass

    @abstractmethod
    def object_properties_in_signature(self) -> Iterable[OWLObjectProperty]:
        """A convenience method that obtains the object properties that are in the signature of this object.

        Returns:
            Object properties that are in the signature of this object
        """
        pass

    @abstractmethod
    def individuals_in_signature(self) -> Iterable[OWLNamedIndividual]:
        """A convenience method that obtains the individuals that are in the signature of this object.

        Returns:
            Individuals that are in the signature of this object.
        """
        pass

    @abstractmethod
    def data_property_range_axioms(self, property: OWLDataProperty) -> Iterable[OWLDataPropertyRangeAxiom]:
        """Gets the OWLDataPropertyRangeAxiom objects where the property is equal to the specified property.

        Args:
            property: The property which is equal to the property of the retrieved axioms.

        Returns:
            the axioms matching the search.
        """
        pass

    @abstractmethod
    def get_owl_ontology_manager(self) -> _M:
        """Gets the manager that manages this ontology"""
        pass

    @abstractmethod
    def get_ontology_id(self) -> OWLOntologyID:
        """Gets the OWLOntologyID belonging to this object.

        Returns:
            The OWLOntologyID
        """
        pass

    def is_anonymous(self) -> bool:
        return self.get_ontology_id().is_anonymous()


# noinspection PyUnresolvedReferences
# noinspection PyDunderSlots
class OWLOntologyChange(metaclass=ABCMeta):
    __slots__ = ()

    _ont: OWLOntology

    @abstractmethod
    def __init__(self, ontology: OWLOntology):
        self._ont = ontology

    def get_ontology(self) -> OWLOntology:
        """Gets the ontology that the change is/was applied to.

        Returns:
            The ontology that the change is applicable to
        """
        return self._ont


class AddImport(OWLOntologyChange):
    """Represents an ontology change where an import statement is added to an ontology."""
    __slots__ = '_ont', '_declaration'

    def __init__(self, ontology: OWLOntology, import_declaration: OWLImportsDeclaration):
        """
        Args:
            ontology: the ontology to which the change is to be applied
            import_declaration: the import declaration
        """
        super().__init__(ontology)
        self._declaration = import_declaration

    def get_import_declaration(self) -> OWLImportsDeclaration:
        """Gets the import declaration that the change pertains to.

        Returns:
            The import declaration
        """
        return self._declaration


class OWLOntologyManager(metaclass=ABCMeta):
    """An OWLOntologyManager manages a set of ontologies. It is the main point for creating, loading and accessing
    ontologies."""

    @abstractmethod
    def create_ontology(self, iri: IRI) -> OWLOntology:
        """Creates a new (empty) ontology that that has the specified ontology IRI (and no version IRI).

        Args:
            iri: The IRI of the ontology to be created.

        Returns:
            The newly created ontology, or if an ontology with the specified IRI already exists then this existing
            ontology will be returned.
        """
        pass

    @abstractmethod
    def load_ontology(self, iri: IRI) -> OWLOntology:
        """Loads an ontology that is assumed to have the specified ontology IRI as its IRI or version IRI. The ontology
        IRI will be mapped to an ontology document IRI.

        Args:
            iri: The IRI that identifies the ontology. It is expected that the ontology will also have this IRI
                (although the OWL API should tolerated situations where this is not the case).

        Returns:
            The OWLOntology representation of the ontology that was loaded.
        """
        pass

    @abstractmethod
    def apply_change(self, change: OWLOntologyChange):
        """A convenience method that applies just one change to an ontology. When this method is used through an
        OWLOntologyManager implementation, the instance used should be the one that the ontology returns through the
        get_owl_ontology_manager() call.

        Args:
            change: The change to be applied

        Raises:
            ChangeApplied.UNSUCCESSFULLY: if the change was not applied successfully.
        """
        pass

    @abstractmethod
    def add_axiom(self, ontology: OWLOntology, axiom: OWLAxiom):
        """A convenience method that adds a single axiom to an ontology. The appropriate AddAxiom change object is
        automatically generated.

        Args:
            ontology: The ontology to add the axiom to.
            axiom: The axiom to be added

        Raises:
            ChangeApplied.UNSUCCESSFULLY: if the axiom could not be added.
        """
        pass

    @abstractmethod
    def save_ontology(self, ontology: OWLOntology, document_iri: IRI):
        """Saves the specified ontology, using the specified document IRI to determine where/how the ontology should be
        saved.

        Args:
            ontology: The ontology to be saved.
            document_iri: The document IRI where the ontology should be saved to
        """
        pass


class OWLReasoner(metaclass=ABCMeta):
    """An OWLReasoner reasons over a set of axioms (the set of reasoner axioms) that is based on the imports closure of
    a particular ontology - the "root" ontology."""
    __slots__ = ()

    @abstractmethod
    def __init__(self, ontology: OWLOntology):
        pass

    @abstractmethod
    def data_property_domains(self, pe: OWLDataProperty, direct: bool = False) -> Iterable[OWLClass]:
        """Gets the named classes that are the direct or indirect domains of this property with respect to the imports
        closure of the root ontology.

        Args:
            pe: The property expression whose domains are to be retrieved.
            direct: Specifies if the direct domains should be retrieved (True), or if all domains should be retrieved
                (False).

        Returns:
            :Let N = equivalent_classes(DataSomeValuesFrom(pe rdfs:Literal)). If direct is True: then if N is not
            empty then the return value is N, else the return value is the result of
            super_classes(DataSomeValuesFrom(pe rdfs:Literal), true). If direct is False: then the result of
            super_classes(DataSomeValuesFrom(pe rdfs:Literal), false) together with N if N is non-empty.
            (Note, rdfs:Literal is the top datatype).
        """
        pass

    @abstractmethod
    def object_property_domains(self, pe: OWLObjectProperty, direct: bool = False) -> Iterable[OWLClass]:
        """Gets the named classes that are the direct or indirect domains of this property with respect to the imports
        closure of the root ontology.

        Args:
            pe: The property expression whose domains are to be retrieved.
            direct: Specifies if the direct domains should be retrieved (True), or if all domains should be retrieved
                (False).

        Returns:
            :Let N = equivalent_classes(ObjectSomeValuesFrom(pe owl:Thing)). If direct is True: then if N is not empty
            then the return value is N, else the return value is the result of
            super_classes(ObjectSomeValuesFrom(pe owl:Thing), true). If direct is False: then the result of
            super_classes(ObjectSomeValuesFrom(pe owl:Thing), false) together with N if N is non-empty.
        """
        pass

    @abstractmethod
    def object_property_ranges(self, pe: OWLObjectProperty, direct: bool = False) -> Iterable[OWLClass]:
        """Gets the named classes that are the direct or indirect ranges of this property with respect to the imports
        closure of the root ontology.

        Args:
            pe: The property expression whose ranges are to be retrieved.
            direct: Specifies if the direct ranges should be retrieved (True), or if all ranges should be retrieved
                (False).

        Returns:
            :Let N = equivalent_classes(ObjectSomeValuesFrom(ObjectInverseOf(pe) owl:Thing)). If direct is True: then
            if N is not empty then the return value is N, else the return value is the result of
            super_classes(ObjectSomeValuesFrom(ObjectInverseOf(pe) owl:Thing), true). If direct is False: then
            the result of super_classes(ObjectSomeValuesFrom(ObjectInverseOf(pe) owl:Thing), false) together with N
            if N is non-empty.
        """
        pass

    @abstractmethod
    def equivalent_classes(self, ce: OWLClassExpression) -> Iterable[OWLClass]:
        """Gets the named classes that are equivalent to the specified class expression with respect to the set of
        reasoner axioms.

        Args:
            ce: The class expression whose equivalent classes are to be retrieved.

        Returns:
            All named class C where the root ontology imports closure entails EquivalentClasses(ce C). If ce is not a
            class name (i.e. it is an anonymous class expression) and there are no such classes C then there will be
            no result. If ce is unsatisfiable with respect to the set of reasoner axioms then  owl:Nothing, i.e. the
            bottom node, will be returned.
        """
        pass

    @abstractmethod
    def data_property_values(self, ind: OWLNamedIndividual, pe: OWLDataProperty) -> Iterable['OWLLiteral']:
        """Gets the data property values for the specified individual and data property expression.

        Args:
            ind: The individual that is the subject of the data property values
            pe: The data property expression whose values are to be retrieved for the specified individual

        Returns:
            A set of OWLLiterals containing literals such that for each literal l in the set, the set of reasoner
            axioms entails DataPropertyAssertion(pe ind l).
        """
        pass

    @abstractmethod
    def object_property_values(self, ind: OWLNamedIndividual, pe: OWLObjectPropertyExpression) \
            -> Iterable[OWLNamedIndividual]:
        """Gets the object property values for the specified individual and object property expression.

        Args:
            ind: The individual that is the subject of the object property values
            pe: The object property expression whose values are to be retrieved for the specified individual

        Returns:
            The named individuals such that for each individual j, the set of reasoner axioms entails
            ObjectPropertyAssertion(pe ind j).
        """
        pass

    @abstractmethod
    def flush(self) -> None:
        """Flushes any changes stored in the buffer, which causes the reasoner to take into consideration the changes
        the current root ontology specified by the changes"""
        pass

    @abstractmethod
    def instances(self, ce: OWLClassExpression, direct: bool = False) -> Iterable[OWLNamedIndividual]:
        """Gets the individuals which are instances of the specified class expression.

        Args:
            ce: The class expression whose instances are to be retrieved.
            direct: Specifies if the direct instances should be retrieved (True), or if all instances should be
                retrieved (False).

        Returns:
            If direct is True, each named individual j where the set of reasoner axioms entails
            DirectClassAssertion(ce, j). If direct is False, each named individual j where the set of reasoner axioms
            entails ClassAssertion(ce, j). If ce is unsatisfiable with respect to the set of reasoner axioms then
            nothing returned.
        """
        pass

    @abstractmethod
    def sub_classes(self, ce: OWLClassExpression, direct: bool = False) -> Iterable[OWLClass]:
        """Gets the set of named classes that are the strict (potentially direct) subclasses of the specified class
        expression with respect to the reasoner axioms.

        Args:
            ce: The class expression whose strict (direct) subclasses are to be retrieved.
            direct: Specifies if the direct subclasses should be retrieved (True) or if the all subclasses
                (descendant) classes should be retrieved (False).

        Returns:
            If direct is True, each class C where reasoner axioms entails DirectSubClassOf(C, ce). If direct is False,
            each class C where reasoner axioms entails StrictSubClassOf(C, ce). If ce is equivalent to owl:Nothing then
            nothing will be returned.
        """
        pass

    @abstractmethod
    def sub_data_properties(self, dp: OWLDataProperty, direct: bool = False) -> Iterable[OWLDataProperty]:
        """Gets the set of named data properties that are the strict (potentially direct) subproperties of the
        specified data property expression with respect to the imports closure of the root ontology.

        Args:
            dp: The data property whose strict (direct) subproperties are to be retrieved.
            direct: Specifies if the direct subproperties should be retrieved (True) or if the all subproperties
                (descendants) should be retrieved (False).

        Returns:
            If direct is True, each property P where the set of reasoner axioms entails DirectSubDataPropertyOf(P, pe).
            If direct is False, each property P where the set of reasoner axioms entails
            StrictSubDataPropertyOf(P, pe). If pe is equivalent to owl:bottomDataProperty then nothing will be
            returned.
        """
        pass

    @abstractmethod
    def sub_object_properties(self, op: OWLObjectPropertyExpression, direct: bool = False) \
            -> Iterable[OWLObjectPropertyExpression]:
        """Gets the stream of simplified object property expressions that are the strict (potentially direct)
        subproperties of the specified object property expression with respect to the imports closure of the root
        ontology.

        Args:
            op: The object property expression whose strict (direct) subproperties are to be retrieved.
            direct: Specifies if the direct subproperties should be retrieved (True) or if the all subproperties
                (descendants) should be retrieved (False).

        Returns:
            If direct is True, simplified object property expressions, such that for each simplified object property
            expression, P, the set of reasoner axioms entails DirectSubObjectPropertyOf(P, pe).
            If direct is False, simplified object property expressions, such that for each simplified object property
            expression, P, the set of reasoner axioms entails StrictSubObjectPropertyOf(P, pe).
            If pe is equivalent to owl:bottomObjectProperty then nothing will be returned.
        """
        pass

    @abstractmethod
    def types(self, ind: OWLNamedIndividual, direct: bool = False) -> Iterable[OWLClass]:
        """Gets the named classes which are (potentially direct) types of the specified named individual.

        Args:
            ind: The individual whose types are to be retrieved.
            direct: Specifies if the direct types should be retrieved (True), or if all types should be retrieved
                (False).

        Returns:
            If direct is True, each named class C where the set of reasoner axioms entails
            DirectClassAssertion(C, ind). If direct is False, each named class C where the set of reasoner axioms
            entails ClassAssertion(C, ind).
        """
        pass

    @abstractmethod
    def get_root_ontology(self) -> OWLOntology:
        """Gets the "root" ontology that is loaded into this reasoner. The reasoner takes into account the axioms in
        this ontology and its imports closure."""
        pass

    @abstractmethod
    def super_classes(self, ce: OWLClassExpression, direct: bool = False) -> Iterable[OWLClass]:
        """Gets the stream of named classes that are the strict (potentially direct) super classes of the specified
        class expression with respect to the imports closure of the root ontology.

        Args:
            ce: The class expression whose strict (direct) super classes are to be retrieved.
            direct: Specifies if the direct super classes should be retrieved (True) or if the all super classes
                (ancestors) classes should be retrieved (False).

        Returns:
            If direct is True, each class C where the set of reasoner axioms entails DirectSubClassOf(ce, C).
            If direct is False, each class C where  set of reasoner axioms entails StrictSubClassOf(ce, C).
            If ce is equivalent to owl:Thing then nothing will be returned.
        """
        pass


"""Important constant objects section"""

OWLThing: Final = OWLClass(OWLRDFVocabulary.OWL_THING.get_iri())  #: : :The OWL Class corresponding to owl:Thing
OWLNothing: Final = OWLClass(OWLRDFVocabulary.OWL_NOTHING.get_iri())  #: : :The OWL Class corresponding to owl:Nothing
#: the built in top object property
OWLTopObjectProperty: Final = OWLObjectProperty(OWLRDFVocabulary.OWL_TOP_OBJECT_PROPERTY.get_iri())
#: the built in bottom object property
OWLBottomObjectProperty: Final = OWLObjectProperty(OWLRDFVocabulary.OWL_BOTTOM_OBJECT_PROPERTY.get_iri())
#: the built in top data property
OWLTopDataProperty: Final = OWLDataProperty(OWLRDFVocabulary.OWL_TOP_DATA_PROPERTY.get_iri())
#: the built in bottom data property
OWLBottomDataProperty: Final = OWLDataProperty(OWLRDFVocabulary.OWL_BOTTOM_DATA_PROPERTY.get_iri())

DoubleOWLDatatype: Final = OWLDatatype(XSDVocabulary.DOUBLE)  #: An object representing a double datatype.
IntegerOWLDatatype: Final = OWLDatatype(XSDVocabulary.INTEGER)  #: An object representing an integer datatype.
BooleanOWLDatatype: Final = OWLDatatype(XSDVocabulary.BOOLEAN)  #: An object representing the boolean datatype.
DateOWLDatatype: Final = OWLDatatype(XSDVocabulary.DATE)  #: An object representing the date datatype.
DateTimeOWLDatatype: Final = OWLDatatype(XSDVocabulary.DATE_TIME)  #: An object representing the dateTime datatype.
DurationOWLDatatype: Final = OWLDatatype(XSDVocabulary.DURATION)  #: An object representing the duration datatype.
TopDatatype: Final = OWLDatatype(OWLRDFVocabulary.RDFS_LITERAL)  #: The OWL Datatype corresponding to the top data type

NUMERIC_DATATYPES: Set[OWLDatatype] = {DoubleOWLDatatype, IntegerOWLDatatype}
TIME_DATATYPES: Final[Set[OWLDatatype]] = {DateOWLDatatype, DateTimeOWLDatatype, DurationOWLDatatype}
