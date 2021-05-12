# -*- coding: utf-8 -*-

import types
from functools import singledispatchmethod
from typing import List, Callable

from owlapy import IRI, namespaces
from owlapy.io import OWLObjectRenderer
from owlapy.model import OWLObject, OWLClass, OWLObjectProperty, OWLObjectSomeValuesFrom, \
    OWLObjectAllValuesFrom, OWLObjectUnionOf, OWLBooleanClassExpression, OWLNaryBooleanClassExpression, \
    OWLObjectIntersectionOf, OWLObjectComplementOf, OWLObjectInverseOf, OWLClassExpression, OWLRestriction, \
    OWLObjectMinCardinality, OWLObjectExactCardinality, OWLObjectMaxCardinality, OWLObjectHasSelf, OWLObjectHasValue, \
    OWLObjectOneOf, OWLNamedIndividual, OWLEntity

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

def _simple_short_form_provider(e: OWLEntity) -> str:
    iri: IRI = e.get_iri()
    sf = iri.get_short_form()
    for ns in [namespaces.XSD, namespaces.OWL, namespaces.RDFS, namespaces.RDF]:
        if iri.get_namespace() == ns:
            return "%s:%s" % (ns.prefix, sf)
    else:
        return sf


class DLSyntaxObjectRenderer(OWLObjectRenderer):
    """DL Syntax renderer for OWL Objects"""
    __slots__ = '_sfp'

    _sfp: Callable[[OWLEntity], str]

    def __init__(self, short_form_provider: Callable[[OWLEntity], str] = _simple_short_form_provider):
        """Create a new DL Syntax renderer

        Args:
            short_form_provider: custom short form provider
        """
        self._sfp = short_form_provider

    def set_short_form_provider(self, short_form_provider: Callable[[OWLEntity], str]) -> None:
        self._sfp = short_form_provider

    @singledispatchmethod
    def render(self, o: OWLObject) -> str:
        assert isinstance(o, OWLObject), f"Tried to render non-OWLObject {o} of {type(o)}"
        raise NotImplementedError

    @render.register
    def _(self, o: OWLClass) -> str:
        if o.is_owl_nothing():
            return _DL_SYNTAX.BOTTOM
        elif o.is_owl_thing():
            return _DL_SYNTAX.TOP
        else:
            return self._sfp(o)

    @render.register
    def _(self, p: OWLObjectProperty) -> str:
        return self._sfp(p)

    @render.register
    def _(self, i: OWLNamedIndividual) -> str:
        return self._sfp(i)

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

    @render.register
    def _(self, r: OWLObjectMinCardinality) -> str:
        return "%s %s %s.%s" % (
            _DL_SYNTAX.MIN, r.get_cardinality(), self.render(r.get_property()), self._render_nested(r.get_filler()))

    @render.register
    def _(self, r: OWLObjectExactCardinality) -> str:
        return "%s %s %s.%s" % (
            _DL_SYNTAX.EQUAL, r.get_cardinality(), self.render(r.get_property()), self._render_nested(r.get_filler()))

    @render.register
    def _(self, r: OWLObjectMaxCardinality) -> str:
        return "%s %s %s.%s" % (
            _DL_SYNTAX.MAX, r.get_cardinality(), self.render(r.get_property()), self._render_nested(r.get_filler()))

    @render.register
    def _(self, r: OWLObjectHasSelf) -> str:
        return "%s %s.%s" % (_DL_SYNTAX.EXISTS, self.render(r.get_property()), _DL_SYNTAX.SELF)

    @render.register
    def _(self, r: OWLObjectHasValue):
        return "%s %s.{%s}" % (_DL_SYNTAX.EXISTS, self.render(r.get_property()),
                               self.render(r.get_filler()))

    @render.register
    def _(self, r: OWLObjectOneOf):
        return "{%s}" % (" %s " % _DL_SYNTAX.OR).join(
            "%s" % (self.render(_)) for _ in r.individuals())

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


_MAN_SYNTAX = types.SimpleNamespace(
    SUBCLASS="SubClassOf",
    EQUIVALENT_TO="EquivalentTo",
    NOT="not",
    DISJOINT_WITH="DisjointWith",
    EXISTS="some",
    FORALL="only",
    MIN="min",
    EQUAL="exactly",
    MAX="max",
    AND="and",
    TOP="Thing",
    BOTTOM="Nothing",
    OR="or",
    INVERSE="inverse",
    COMMA=",",
    SELF="self",
    VALUE="value",
)


class ManchesterOWLSyntaxOWLObjectRenderer(OWLObjectRenderer):
    """Manchester Syntax renderer for OWL Objects"""
    __slots__ = '_sfp'

    _sfp: Callable[[OWLEntity], str]

    def __init__(self, short_form_provider: Callable[[OWLEntity], str] = _simple_short_form_provider):
        """Create a new Manchester Syntax renderer

        Args:
            short_form_provider: custom short form provider
        """
        self._sfp = short_form_provider

    def set_short_form_provider(self, short_form_provider: Callable[[OWLEntity], str]) -> None:
        self._sfp = short_form_provider

    @singledispatchmethod
    def render(self, o: OWLObject) -> str:
        assert isinstance(o, OWLObject), f"Tried to render non-OWLObject {o} of {type(o)}"
        raise NotImplementedError

    @render.register
    def _(self, o: OWLClass) -> str:
        if o.is_owl_nothing():
            return _MAN_SYNTAX.BOTTOM
        elif o.is_owl_thing():
            return _MAN_SYNTAX.TOP
        else:
            return self._sfp(o)

    @render.register
    def _(self, p: OWLObjectProperty) -> str:
        return self._sfp(p)

    @render.register
    def _(self, i: OWLNamedIndividual) -> str:
        return self._sfp(i)

    @render.register
    def _(self, e: OWLObjectSomeValuesFrom) -> str:
        return "%s %s %s" % (self.render(e.get_property()), _MAN_SYNTAX.EXISTS, self._render_nested(e.get_filler()))

    @render.register
    def _(self, e: OWLObjectAllValuesFrom) -> str:
        return "%s %s %s" % (self.render(e.get_property()), _MAN_SYNTAX.FORALL, self._render_nested(e.get_filler()))

    @render.register
    def _(self, c: OWLObjectUnionOf) -> str:
        return (" %s " % _MAN_SYNTAX.OR).join(self._render_operands(c))

    @render.register
    def _(self, c: OWLObjectIntersectionOf) -> str:
        return (" %s " % _MAN_SYNTAX.AND).join(self._render_operands(c))

    @render.register
    def _(self, n: OWLObjectComplementOf) -> str:
        return "%s %s" % (_MAN_SYNTAX.NOT, self._render_nested(n.get_operand()))

    @render.register
    def _(self, p: OWLObjectInverseOf) -> str:
        return "%s(%s)" % (self.render(p.get_named_property()), _MAN_SYNTAX.INVERSE)

    @render.register
    def _(self, r: OWLObjectMinCardinality) -> str:
        return "%s %s %s %s" % (
            self.render(r.get_property()), _MAN_SYNTAX.MIN, r.get_cardinality(), self._render_nested(r.get_filler()))

    @render.register
    def _(self, r: OWLObjectExactCardinality) -> str:
        return "%s %s %s %s" % (
            self.render(r.get_property()), _MAN_SYNTAX.EQUAL, r.get_cardinality(), self._render_nested(r.get_filler()))

    @render.register
    def _(self, r: OWLObjectMaxCardinality) -> str:
        return "%s %s %s %s" % (
            self.render(r.get_property()), _MAN_SYNTAX.MAX, r.get_cardinality(), self._render_nested(r.get_filler()))

    @render.register
    def _(self, r: OWLObjectHasSelf) -> str:
        return "%s %s %s" % (self.render(r.get_property()), _MAN_SYNTAX.EXISTS, _MAN_SYNTAX.SELF)

    @render.register
    def _(self, r: OWLObjectHasValue):
        return "%s %s %s" % (self.render(r.get_property()), _MAN_SYNTAX.VALUE,
                             self.render(r.get_filler()))

    @render.register
    def _(self, r: OWLObjectOneOf):
        return "{%s}" % (" %s " % _MAN_SYNTAX.COMMA).join(
            "%s" % (self.render(_)) for _ in r.individuals())

    # TODO
    # @render.register
    # def _(self, r: OWLFacetRestriction):
    #     return "%s %s" % (_FACETS.get(r.get_facet(), r.get_facet().get_symbolic_form()), r.get_facet_value())

    # TODO
    # @render.register
    # def _(self, r: OWLDatatypeRestriction):
    #     s = [self.render(_) for _ in r.facet_restrictions()]
    #     return "%s[%s]" % (self.render(r.get_datatype()), (" %s " % _MAN_SYNTAX.COMMA).join(s))

    # TODO
    # @render.register
    # def _(self, r: OWLObjectPropertyChain):
    #     return (" %s " % _MAN_SYNTAX.COMP).join(self.render(_) for _ in r.property_chain())

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
