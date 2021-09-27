from owlapy.model import OWLDatatypeRestriction, OWLFacet, OWLFacetRestriction


def OWLDatatypeMaxExclusiveRestriction(max_: float) -> OWLDatatypeRestriction:
    r = OWLFacetRestriction(OWLFacet.MAX_EXCLUSIVE, max_)
    return OWLDatatypeRestriction(r.get_facet_value().get_datatype(), r)


def OWLDatatypeMinExclusiveRestriction(min_: float) -> OWLDatatypeRestriction:
    r = OWLFacetRestriction(OWLFacet.MIN_EXCLUSIVE, min_)
    return OWLDatatypeRestriction(r.get_facet_value().get_datatype(), r)


def OWLDatatypeMaxInclusiveRestriction(max_: float) -> OWLDatatypeRestriction:
    r = OWLFacetRestriction(OWLFacet.MAX_INCLUSIVE, max_)
    return OWLDatatypeRestriction(r.get_facet_value().get_datatype(), r)


def OWLDatatypeMinInclusiveRestriction(min_: float) -> OWLDatatypeRestriction:
    r = OWLFacetRestriction(OWLFacet.MIN_INCLUSIVE, min_)
    return OWLDatatypeRestriction(r.get_facet_value().get_datatype(), r)


def OWLDatatypeMinMaxExclusiveRestriction(min_: float, max_: float) -> OWLDatatypeRestriction:
    if isinstance(min_, float) and isinstance(max_, int):
        max_ = float(max_)
    if isinstance(max_, float) and isinstance(min_, int):
        min_ = float(min_)
    r_min = OWLFacetRestriction(OWLFacet.MIN_EXCLUSIVE, min_)
    r_max = OWLFacetRestriction(OWLFacet.MAX_EXCLUSIVE, max_)
    restrictions = (r_min, r_max)
    return OWLDatatypeRestriction(r_min.get_facet_value().get_datatype(), restrictions)


def OWLDatatypeMinMaxInclusiveRestriction(min_: float, max_: float) -> OWLDatatypeRestriction:
    if isinstance(min_, float) and isinstance(max_, int):
        max_ = float(max_)
    if isinstance(max_, float) and isinstance(min_, int):
        min_ = float(min_)
    r_min = OWLFacetRestriction(OWLFacet.MIN_INCLUSIVE, min_)
    r_max = OWLFacetRestriction(OWLFacet.MAX_INCLUSIVE, max_)
    restrictions = (r_min, r_max)
    return OWLDatatypeRestriction(r_min.get_facet_value().get_datatype(), restrictions)
