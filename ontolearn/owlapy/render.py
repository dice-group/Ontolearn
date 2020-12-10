import types
from functools import singledispatchmethod
from types import MappingProxyType
from typing import List

from ontolearn.owlapy import IRI
from ontolearn.owlapy.io import OWLObjectRenderer
from ontolearn.owlapy.model import OWLObject, OWLClass, OWLObjectProperty, OWLObjectSomeValuesFrom, \
    OWLObjectAllValuesFrom, OWLObjectUnionOf, OWLBooleanClassExpression, OWLNaryBooleanClassExpression, \
    OWLObjectIntersectionOf, OWLObjectComplementOf, OWLObjectInverseOf, OWLClassExpression, OWLRestriction

_DL_SYNTAX = types.SimpleNamespace(
    SUBCLASS="⊑",
    EQUIVALENT_TO="≡",
    NOT="¬",
    DISJOINT_WITH="⊑" + " " + "¬",
    EXISTS="∃",
    FORALL="∀",
    IN="∈",
    MIN="≥",
    EQUAL="=",
    NOT_EQUAL="≠",
    MAX="≤",
    INVERSE="⁻",
    AND="⊓",
    TOP="⊤",
    BOTTOM="⊥",
    OR="⊔",
    COMP="∘",
    WEDGE="⋀",
    IMPLIES="←",
    COMMA=",",
    SELF="self",
)

# TODO
# _FACETS = MappingProxyType({
#     OWLFacet.MIN_INCLUSIVE: "\u2265", # >=
#     OWLFacet.MIN_EXCLUSIVE: "\u003e", # >
#     OWLFacet.MAX_INCLUSIVE: "\u2264", # <=
#     OWLFacet.MAX_EXCLUSIVE: "\u003c", # <
# })

def _simple_short_form_provider(iri: IRI):
    sf = iri.get_short_form()
    if iri.get_namespace() == "http://www.w3.org/2001/XMLSchema#":
        return "xsd:%s" % sf
    else:
        return sf

class DLSyntaxRenderer(OWLObjectRenderer):
    __slots__ = '_sfp'

    def __init__(self, short_form_provider = _simple_short_form_provider):
        self._sfp = short_form_provider

    def set_short_form_provider(self, short_form_provider) -> None:
        self._sfp = short_form_provider

    @singledispatchmethod
    def render(self, o: OWLObject) -> str:
        raise NotImplementedError

    @render.register
    def _(self, o: OWLClass) -> str:
        if o.is_owl_nothing():
            return _DL_SYNTAX.BOTTOM
        elif o.is_owl_thing():
            return _DL_SYNTAX.TOP
        else:
            return self._sfp(o.get_iri())

    @render.register
    def _(self, p: OWLObjectProperty) -> str:
        return self._sfp(p.get_iri())
    
    @render.register
    def _(self, e: OWLObjectSomeValuesFrom) -> str:
        return "%s %s.%s" % (_DL_SYNTAX.EXISTS, self.render(e.get_property()), self._render_nested(e.get_filler()))
    
    @render.register
    def _(self, e: OWLObjectAllValuesFrom) -> str:
        return "%s %s.%s" % (_DL_SYNTAX.FORALL, self.render(e.get_property()), self._render_nested(e.get_filler()))

    @render.register
    def _(self, c: OWLObjectUnionOf) -> str:
        return (" %s " % _DL_SYNTAX.OR).join(self._render_operands(c))

    @render.register
    def _(self, c: OWLObjectIntersectionOf) -> str:
        return (" %s " % _DL_SYNTAX.AND).join(self._render_operands(c))

    @render.register
    def _(self, n: OWLObjectComplementOf) -> str:
        return "%s%s" % (_DL_SYNTAX.NOT, self._render_nested(n.get_operand()))

    @render.register
    def _(self, p: OWLObjectInverseOf) -> str:
        return "%s%s" % (self.render(p.get_named_property()), _DL_SYNTAX.INVERSE)

    # TODO
    # @render.register
    # def _(self, r: OWLObjectMinCardinality) -> str:
    #     return "%s %s %s .%s" % (
    #         _DL_SYNTAX.MIN, r.get_cardinality(), self.render(r.get_property()), self._render_nested(r.get_filler()))

    # TODO
    # @render.register
    # def _(self, r: OWLObjectExactCardinality) -> str:
    #     return "%s %s %s .%s" % (
    #         _DL_SYNTAX.EQUAL, r.get_cardinality(), self.render(r.get_property()), self._render_nested(r.get_filler()))

    # TODO
    # @render.register
    # def _(self, r: OWLObjectMaxCardinality) -> str:
    #     return "%s %s %s .%s" % (
    #         _DL_SYNTAX.MAX, r.get_cardinality(), self.render(r.get_property()), self._render_nested(r.get_filler()))

    # TODO
    # @render.register
    # def _(self, r: OWLObjectHasSelf) -> str:
    #     return "%s %s .%s" % (_DL_SYNTAX.EXISTS, self.render(r.get_property()), _DL_SYNTAX.SELF)

    # TODO
    # @render.register
    # def _(self, r: OWLObjectHasValue):
    #     return "%s %s .{%s}" % (_DL_SYNTAX.EXISTS, self.render(r.get_property()),
    #                             self.render(r.get_filler()))

    # TODO
    # @render.register
    # def _(self, r: OWLObjectOneOf):
    #     return "{%s}" % (" %s " % _DL_SYNTAX.OR).join(
    #         "%s" % (self.render(_)) for _ in r.individuals())

    # TODO
    # @render.register
    # def _(self, r: OWLFacetRestriction):
    #     return "%s %s" % (_FACETS.get(r.get_facet(), r.get_facet().get_symbolic_form()), r.get_facet_value())

    # TODO
    # @render.register
    # def _(self, r: OWLDatatypeRestriction):
    #     s = [self.render(_) for _ in r.facet_restrictions()]
    #     return "%s[%s]" % (self.render(r.get_datatype()), (" %s " % _DL_SYNTAX.COMMA).join(s))

    # TODO
    # @render.register
    # def _(self, r: OWLObjectPropertyChain):
    #     return (" %s " % _DL_SYNTAX.COMP).join(self.render(_) for _ in r.property_chain())

    # TODO
    # @render.register
    # def _(self, t: OWLDatatype):
    #     return self._sfp(t.get_iri())

    def _render_operands(self, c: OWLNaryBooleanClassExpression) -> List[str]:
        return [self._render_nested(_) for _ in c.operands()]

    def _render_nested(self, c: OWLClassExpression) -> str:
        if isinstance(c, OWLBooleanClassExpression) or isinstance(c, OWLRestriction):
            return "(%s)" % self.render(c)
        else:
            return self.render(c)
