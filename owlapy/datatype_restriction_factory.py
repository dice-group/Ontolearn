from functools import singledispatchmethod
from typing import Iterable, Union
from owlapy.model import OWLDatatype, OWLDatatypeRestriction, OWLFacet, OWLFacetRestriction, OWLLiteral


class DatatypeRestrictionFactory():
    """Factory for OWLDatatypeRestrictions and OWLFacetRestrictions."""

    slots = ()

    def get_max_exclusive_restriction(self, max_: float) -> OWLDatatypeRestriction:
        r = self.get_facet_restriction(max_, OWLFacet.MAX_EXCLUSIVE)
        return self.get_owl_datatype_restriction(r.get_facet_value().get_datatype(), r)

    def get_min_exclusive_restriction(self, min_: float) -> OWLDatatypeRestriction:
        r = self.get_facet_restriction(min_, OWLFacet.MIN_EXCLUSIVE)
        return self.get_owl_datatype_restriction(r.get_facet_value().get_datatype(), r)

    def get_max_inclusive_restriction(self, max_: float) -> OWLDatatypeRestriction:
        r = self.get_facet_restriction(max_, OWLFacet.MAX_INCLUSIVE)
        return self.get_owl_datatype_restriction(r.get_facet_value().get_datatype(), r)

    def get_min_inclusive_restriction(self, min_: float) -> OWLDatatypeRestriction:
        r = self.get_facet_restriction(min_, OWLFacet.MIN_INCLUSIVE)
        return self.get_owl_datatype_restriction(r.get_facet_value().get_datatype(), r)

    def get_min_max_exclusive_restriction(self, min_: float, max_: float) -> OWLDatatypeRestriction:
        if isinstance(min_, float) and isinstance(max_, int):
            max_ = float(max_)
        if isinstance(max_, float) and isinstance(min_, int):
            min_ = float(min_)
        r_min = self.get_facet_restriction(min_, OWLFacet.MIN_EXCLUSIVE)
        r_max = self.get_facet_restriction(max_, OWLFacet.MAX_EXCLUSIVE)
        restrictions = (r_min, r_max)
        return self.get_owl_datatype_restriction(r_min.get_facet_value().get_datatype(), restrictions)

    def get_min_max_inclusive_restriction(self, min_: float, max_: float) -> OWLDatatypeRestriction:
        if isinstance(min_, float) and isinstance(max_, int):
            max_ = float(max_)
        if isinstance(max_, float) and isinstance(min_, int):
            min_ = float(min_)
        r_min = self.get_facet_restriction(min_, OWLFacet.MIN_INCLUSIVE)
        r_max = self.get_facet_restriction(max_, OWLFacet.MAX_INCLUSIVE)
        restrictions = (r_min, r_max)
        return self.get_owl_datatype_restriction(r_min.get_facet_value().get_datatype(), restrictions)

    def get_owl_datatype_restriction(self, type_: OWLDatatype,
                                     restrictions: Union[OWLFacetRestriction, Iterable[OWLFacetRestriction]]) \
            -> OWLDatatypeRestriction:
        if isinstance(restrictions, OWLFacetRestriction):
            restrictions = restrictions,
        return OWLDatatypeRestriction(type_, restrictions)

    @singledispatchmethod
    def get_facet_restriction(self, n, facet: OWLFacet):
        return NotImplementedError

    @get_facet_restriction.register
    def _(self, n: float, facet: OWLFacet) -> OWLFacetRestriction:
        return OWLFacetRestriction(facet, OWLLiteral(n))

    @get_facet_restriction.register
    def _(self, n: int, facet: OWLFacet) -> OWLFacetRestriction:
        return OWLFacetRestriction(facet, OWLLiteral(n))

    @get_facet_restriction.register
    def _(self, n: OWLLiteral, facet: OWLFacet) -> OWLFacetRestriction:
        return OWLFacetRestriction(facet, n)
