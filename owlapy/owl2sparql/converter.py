from collections import defaultdict
from contextlib import contextmanager
from functools import singledispatchmethod
from types import MappingProxyType
from typing import Set, List, Dict, Optional, Iterable

from rdflib.plugins.sparql.parser import parseQuery

from owlapy.model import OWLClassExpression, OWLClass, OWLEntity, OWLObjectProperty, OWLObjectIntersectionOf, \
    OWLObjectUnionOf, OWLObjectComplementOf, OWLObjectSomeValuesFrom, OWLObjectAllValuesFrom, OWLObjectHasValue, \
    OWLNamedIndividual, OWLObjectCardinalityRestriction, OWLObjectMinCardinality, OWLObjectExactCardinality, \
    OWLObjectMaxCardinality, OWLDataCardinalityRestriction, OWLDataProperty, OWLObjectHasSelf, OWLObjectOneOf, \
    OWLDataSomeValuesFrom, OWLDataAllValuesFrom, OWLDataHasValue, OWLDatatype, TopOWLDatatype, OWLDataOneOf, \
    OWLLiteral, OWLDatatypeRestriction
from owlapy.vocab import OWLFacet, OWLRDFVocabulary

_Variable_facet_comp = MappingProxyType({
    OWLFacet.MIN_INCLUSIVE: ">=",
    OWLFacet.MIN_EXCLUSIVE: ">",
    OWLFacet.MAX_INCLUSIVE: "<=",
    OWLFacet.MAX_EXCLUSIVE: "<"
})


def peek(x):
    return x[-1]


class VariablesMapping:
    __slots__ = 'class_cnt', 'prop_cnt', 'ind_cnt', 'dict'

    def __init__(self):
        self.class_cnt = 0
        self.prop_cnt = 0
        self.ind_cnt = 0
        self.dict = dict()

    def get_variable(self, e: OWLEntity) -> str:
        if e in self.dict:
            return self.dict[e]

        if isinstance(e, OWLClass):
            self.class_cnt += 1
            var = f"?cls_{self.class_cnt}"
        elif isinstance(e, OWLObjectProperty) or isinstance(e, OWLDataProperty):
            self.prop_cnt += 1
            var = f"?p_{self.prop_cnt}"
        elif isinstance(e, OWLNamedIndividual):
            self.ind_cnt += 1
            var = f"?ind_{self.ind_cnt}"
        else:
            raise ValueError(e)

        self.dict[e] = var
        return var

    def new_individual_variable(self) -> str:
        self.ind_cnt += 1
        return f"?s_{self.ind_cnt}"

    def new_property_variable(self) -> str:
        self.prop_cnt += 1
        return f"?p_{self.prop_cnt}"

    def __contains__(self, item: OWLEntity) -> bool:
        return item in self.dict

    def __getitem__(self, item: OWLEntity) -> str:
        return self.dict[item]


class Owl2SparqlConverter:
    __slots__ = 'ce', 'sparql', 'variables', 'parent', 'parent_var', 'properties', 'variable_entities', 'cnt', \
                'mapping', 'grouping_vars', 'having_conditions', '_intersection'

    ce: OWLClassExpression
    sparql: List[str]
    variables: List[str]
    parent: List[OWLClassExpression]
    parent_var: List[str]
    variable_entities: Set[OWLEntity]
    properties: Dict[int, List[OWLEntity]]
    _intersection: Dict[int, bool]
    mapping: VariablesMapping
    grouping_vars: Dict[OWLClassExpression, Set[str]]
    having_conditions: Dict[OWLClassExpression, Set[str]]
    cnt: int

    def convert(self, root_variable: str, ce: OWLClassExpression):
        self.ce = ce
        self.sparql = []
        self.variables = []
        self.parent = []
        self.parent_var = []
        self.properties = defaultdict(list)
        self.variable_entities = set()
        self._intersection = defaultdict(bool)
        self.cnt = 0
        self.mapping = VariablesMapping()
        self.grouping_vars = defaultdict(set)
        self.having_conditions = defaultdict(set)
        with self.stack_variable(root_variable):
            with self.stack_parent(ce):
                self.process(ce)
        return self.sparql

    @property
    def modal_depth(self):
        return len(self.variables)

    @property
    def in_intersection(self):
        return self._intersection[self.modal_depth]

    @singledispatchmethod
    def render(self, e):
        raise NotImplementedError(e)

    @render.register
    def _(self, lit: OWLLiteral):
        return f'"{lit.get_literal()}"^^<{lit.get_datatype().to_string_id()}>'

    @render.register
    def _(self, e: OWLEntity):
        if e in self.variable_entities:
            s = self.mapping.get_variable(e)
        else:
            s = f"<{e.to_string_id()}>"
        if isinstance(e, OWLObjectProperty):
            self.properties[self.modal_depth].append(e)
        return s

    def _maybe_quote(self, e):
        assert isinstance(e, str)
        if e.startswith("?"):
            return e
        else:
            return f"<{e}>"

    def _maybe_quote_p(self, p):
        if isinstance(p, str):
            if p.startswith("?") or p == "a":
                return p
            else:
                return f"<{p}>"
        else:
            return self.render(p)

    def _maybe_render(self, o):
        if isinstance(o, str):
            return o
        else:
            return self.render(o)

    @contextmanager
    def intersection(self):
        self._intersection[self.modal_depth] = True
        try:
            yield
        finally:
            del self._intersection[self.modal_depth]

    @contextmanager
    def stack_variable(self, var):
        self.variables.append(var)
        try:
            yield
        finally:
            self.variables.pop()

    @contextmanager
    def stack_parent(self, parent: OWLClassExpression):
        self.parent.append(parent)
        self.parent_var.append(self.current_variable)
        try:
            yield
        finally:
            self.parent.pop()

    @property
    def current_variable(self):
        return peek(self.variables)

    @singledispatchmethod
    def process(self, ce: OWLClassExpression):
        raise NotImplementedError(ce)

    @process.register
    def _(self, ce: OWLClass):
        if self.ce == ce or not ce.is_owl_thing():
            self.append_triple(self.current_variable, "a", self.render(ce))

    @process.register
    def _(self, ce: OWLObjectIntersectionOf):
        with self.intersection():
            for op in ce.operands():
                self.process(op)
            props = self.properties[self.modal_depth]
            vars_ = set()
            if props:
                for p in props:
                    if p in self.mapping:
                        vars_.add(self.mapping[p])
                if len(vars_) == 2:
                    v0, v1 = sorted(vars_)
                    self.append(f"FILTER ( {v0} != {v1} )")

    @process.register
    def _(self, ce: OWLObjectUnionOf):
        first = True
        for op in ce.operands():
            if first:
                first = False
            else:
                self.append(" UNION ")
            self.append("{ ")
            with self.stack_parent(op):
                self.process(op)
            self.append(" }")

    @process.register
    def _(self, ce: OWLObjectComplementOf):
        subject = self.current_variable
        if not self.in_intersection and self.modal_depth == 1:
            self.append_triple(subject, "?p", "?o")

        self.append("FILTER NOT EXISTS { ")
        self.process(ce.get_operand())
        self.append(" }")

    @process.register
    def _(self, ce: OWLObjectSomeValuesFrom):
        object_variable = self.mapping.new_individual_variable()
        property_expression = ce.get_property()
        if property_expression.is_anonymous():
            # property expression is inverse of a property
            self.append_triple(object_variable, property_expression.get_named_property(), self.current_variable)
        else:
            self.append_triple(self.current_variable, property_expression.get_named_property(), object_variable)
        filler = ce.get_filler()
        with self.stack_variable(object_variable):
            self.process(filler)

    @process.register
    def _(self, ce: OWLObjectAllValuesFrom):
        subject = self.current_variable
        object_variable = self.mapping.new_individual_variable()
        property_expression = ce.get_property()
        predicate = property_expression.get_named_property()
        filler = ce.get_filler()

        if self.modal_depth == 1:
            self.append_triple(self.current_variable, "a", f"<{OWLRDFVocabulary.OWL_NAMED_INDIVIDUAL.as_str()}>")
        if filler.is_owl_thing():
            self.append_triple(self.current_variable, self.mapping.new_property_variable(), object_variable)
        else:
            if property_expression.is_anonymous():
                # property expression is inverse of a property
                self.append_triple(object_variable, predicate, self.current_variable)
            else:
                self.append_triple(self.current_variable, predicate, object_variable)

            # restrict filler
            var = self.mapping.new_individual_variable()
            cnt_var1 = self.new_count_var()
            self.append(f"{{ SELECT {subject} ( COUNT( {var} ) AS {cnt_var1} ) WHERE {{ ")
            self.append_triple(subject, predicate, var)
            with self.stack_variable(var):
                self.process(filler)
            self.append(f" }} GROUP BY {subject} }}")

            var = self.mapping.new_individual_variable()
            cnt_var2 = self.new_count_var()
            self.append(f"{{ SELECT {subject} ( COUNT( {var} ) AS {cnt_var2} ) WHERE {{ ")
            self.append_triple(subject, predicate, var)
            self.append(f" }} GROUP BY {subject} }}")

            self.append(f" FILTER( {cnt_var1} = {cnt_var2} )")

    @process.register
    def _(self, ce: OWLObjectHasValue):
        property_expression = ce.get_property()
        value = ce.get_filler()
        assert isinstance(value, OWLNamedIndividual)
        if property_expression.is_anonymous():
            self.append_triple(value.to_string_id(), property_expression.get_named_property(), self.current_variable)
        else:
            self.append_triple(self.current_variable, property_expression.get_named_property(), value)

    @process.register
    def _(self, ce: OWLObjectCardinalityRestriction):
        subject_variable = self.current_variable
        object_variable = self.mapping.new_individual_variable()
        property_expression = ce.get_property()
        cardinality = ce.get_cardinality()

        if isinstance(ce, OWLObjectMinCardinality):
            comparator = ">="
        elif isinstance(ce, OWLObjectMaxCardinality):
            comparator = "<="
        elif isinstance(ce, OWLObjectExactCardinality):
            comparator = "="
        else:
            raise ValueError(ce)

        if property_expression.is_anonymous():
            # property expression is inverse of a property
            self.append_triple(object_variable, property_expression.get_named_property(), subject_variable)
        else:
            self.append_triple(subject_variable, property_expression.get_named_property(), object_variable)

        filler = ce.get_filler()
        with self.stack_variable(object_variable):
            self.process(filler)

        having_condition = f"COUNT ( {object_variable} ) {comparator} {cardinality}"

        pe = peek(self.parent)
        self.grouping_vars[pe].add(peek(self.parent_var))
        self.having_conditions[pe].add(having_condition)

    @process.register
    def _(self, ce: OWLDataCardinalityRestriction):
        subject_variable = self.current_variable
        object_variable = self.mapping.new_individual_variable()
        property_expression = ce.get_property()
        assert isinstance(property_expression, OWLDataProperty)
        cardinality = ce.get_cardinality()

        if isinstance(ce, OWLObjectMinCardinality):
            comparator = ">="
        elif isinstance(ce, OWLObjectMaxCardinality):
            comparator = "<="
        elif isinstance(ce, OWLObjectExactCardinality):
            comparator = "="
        else:
            raise ValueError(ce)

        self.append(f"{{ SELECT {subject_variable} WHERE {{ ")
        self.append_triple(subject_variable, property_expression, object_variable)

        filler = ce.get_filler()
        with self.stack_variable(object_variable):
            self.process(filler)

        self.append(f" }} GROUP BY {subject_variable}"
                    f" HAVING ( COUNT ( {object_variable} ) {comparator} {cardinality} ) }}")

    @process.register
    def _(self, ce: OWLObjectHasSelf):
        subject = self.current_variable
        property = ce.get_property()
        self.append_triple(subject, property.get_named_property(), subject)

    @process.register
    def _(self, ce: OWLObjectOneOf):
        subject = self.current_variable
        if self.modal_depth == 1:
            self.append_triple(subject, "?p", "?o")

        self.append(f" FILTER ( {subject} IN ( ")
        first = True
        for ind in ce.individuals():
            if first:
                first = False
            else:
                self.append(",")
            assert isinstance(ind, OWLNamedIndividual)
            self.append(f"<{ind.to_string_id()}>")
        self.append(f" )")

    @process.register
    def _(self, ce: OWLDataSomeValuesFrom):
        object_variable = self.mapping.new_individual_variable()
        property_expression = ce.get_property()
        assert isinstance(property_expression, OWLDataProperty)
        self.append_triple(self.current_variable, property_expression, object_variable)
        filler = ce.get_filler()
        with self.stack_variable(object_variable):
            self.process(filler)

    @process.register
    def _(self, ce: OWLDataAllValuesFrom):
        subject = self.current_variable
        object_variable = self.mapping.new_individual_variable()
        property_expression = ce.get_property()
        assert isinstance(property_expression, OWLDataProperty)
        predicate = property_expression.to_string_id()
        filler = ce.get_filler()

        self.append_triple(self.current_variable, predicate, object_variable)

        var = self.mapping.new_individual_variable()
        cnt_var1 = self.new_count_var()
        self.append(f"{{ SELECT {subject} ( COUNT( {var} ) AS {cnt_var1} ) WHERE {{ ")
        self.append_triple(subject, predicate, var)
        with self.stack_variable(var):
            self.process(filler)
        self.append(f" }} GROUP BY {subject} }}")

        var = self.mapping.new_individual_variable()
        cnt_var2 = self.new_count_var()
        self.append(f"{{ SELECT {subject} ( COUNT( {var} ) AS {cnt_var2} ) WHERE {{ ")
        self.append_triple(subject, predicate, var)
        self.append(f" }} GROUP BY {subject} }}")

        self.append(f" FILTER( {cnt_var1} = {cnt_var2} )")

    @process.register
    def _(self, ce: OWLDataHasValue):
        property_expression = ce.get_property()
        value = ce.get_filler()
        assert isinstance(value, OWLDataProperty)
        self.append_triple(self.current_variable, property_expression, value)

    @process.register
    def _(self, node: OWLDatatype):
        if node != TopOWLDatatype:
            self.append(f" FILTER ( DATATYPE ( {self.current_variable} = <{node.to_string_id()}> ) ) ")

    @process.register
    def _(self, node: OWLDataOneOf):
        subject = self.current_variable
        if self.modal_depth == 1:
            self.append_triple(subject, "?p", "?o")
        self.append(f" FILTER ( {subject} IN ( ")
        first = True
        for value in node.values():
            if first:
                first = False
            else:
                self.append(",")
            if value:
                self.append(self.render(value))
        self.append(f" ) ) ")

    @process.register
    def _(self, node: OWLDatatypeRestriction):
        frs = node.get_facet_restrictions()

        for fr in frs:
            facet = fr.get_facet()
            value = fr.get_facet_value()

            if facet in _Variable_facet_comp:
                self.append(f' FILTER ( {self.current_variable} {_Variable_facet_comp[facet]}'
                            f' "{value.get_literal()}"^^<{value.get_datatype().to_string_id()}> ) ')

    def new_count_var(self) -> str:
        self.cnt += 1
        return f"?cnt_{self.cnt}"

    def append_triple(self, subject, predicate, object_):
        self.append(self.triple(subject, predicate, object_))

    def append(self, frag):
        self.sparql.append(frag)

    def triple(self, subject, predicate, object_):
        return f"{self._maybe_quote(subject)} {self._maybe_quote_p(predicate)} {self._maybe_render(object_)} . "

    def as_query(self,
                 root_variable: str, ce: OWLClassExpression, count: bool,
                 values: Optional[Iterable[OWLNamedIndividual]] = None):
        qs = ["SELECT"]
        tp = self.convert(root_variable, ce)
        if count:
            qs.append(f" ( COUNT ( DISTINCT {root_variable} ) AS ?cnt ) WHERE {{ ")
        else:
            qs.append(f" DISTINCT {root_variable} WHERE {{ ")
        if values is not None and root_variable.startswith("?"):
            q = [f"VALUES {root_variable} {{ "]
            for x in values:
                q.append(f"<{x.to_string_id()}>")
            q.append(f"}} . ")
            qs.extend(q)
        qs.extend(tp)
        qs.append(f" }}")

        group_by_vars = self.grouping_vars[ce]
        if group_by_vars:
            qs.append("GROUP BY " + " ".join(sorted(group_by_vars)))
        conditions = self.having_conditions[ce]
        if conditions:
            qs.append(" HAVING ( ")
            qs.append(" && ".join(sorted(conditions)))
            qs.append(" )")
        query = "\n".join(qs)
        parseQuery(query)
        return query

