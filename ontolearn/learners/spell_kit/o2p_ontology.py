SPECIAL_CHARS = {
    r"\sqsubseteq": "\u2291",
    r"\exists": "\u2203",
    r"\sqcap": "\u2293",
    r"\equiv": "\u2261",
    r"\top": "\u22A4",
}


class FeatureNotSupported(Exception):
    def __init__(self, *args):
        super(FeatureNotSupported, self).__init__(*args)


class Ontology(object):
    def __init__(self):
        self.__classes = {}
        self.__rules = []  # class rules
        self.__properties = []
        self.__prop_sub = {}  # property subclasses
        self.__prop_range = {}  # property ranges
        self.__prop_domain = {}  # property domains

    def add_rule(self, rule):
        self.__rules.append(rule)

    def normalize(self):
        result = Ontology()
        for rule in self.rules:
            for r in rule.normalize():
                result.add_rule(r)
        for prop in self.properties:
            result.add_property(prop)
        return result

    def add_property(self, prop):
        self.__prop_sub[prop.identifier] = prop.parent

        if prop.domain is not None:
            self.__prop_domain[prop.identifier] = prop.domain
        if prop.range is not None:
            self.__prop_range[prop.identifier] = prop.range
        elif prop.identifier not in self.__prop_range:
            self.__prop_range[prop.identifier] = TopClass()
        self.__properties.append(prop)

    def check_implicit_cycle(self):
        for prop in self.__properties:
            pass

    @property
    def properties(self):
        return iter(self.__properties)

    @property
    def rules(self):
        return iter(self.__rules)

    @property
    def subproperties(self):
        for sub, sups in self.__prop_sub.items():
            for sup in sups:
                yield SubProperty(PropertyReference(sub), sup)

    @property
    def property_ranges(self):
        return iter(self.__prop_range.items())

    @property
    def property_domains(self):
        return iter(self.__prop_domain.items())

    def __iter__(self):
        return iter(self.__rules)

    def __len__(self):
        return len(self.__rules)


class NameFactory(object):

    __i = 0
    created_names = set()

    @classmethod
    def next_name(cls):
        cls.__i += 1
        name = "NC_%d" % cls.__i
        cls.created_names.add(name)
        return name


class OntologyObject(object):
    """The base class for all entities in the ontology."""

    def to_latex(self):
        raise NotImplementedError()

    def __repr__(self):
        return self.__str__()

    @property
    def is_rule(self):
        return False


class SubProperty(object):
    def __init__(self, subject, object_):
        self.subject = subject
        self.object = object_

    def __str__(self):
        return "%s %s %s" % (self.subject, SPECIAL_CHARS["\sqsubseteq"], self.object)


class Property(object):
    pass


class PropertyReference(Property):
    def __init__(self, identifier):
        self.identifier = identifier

    def __str__(self):
        return self.identifier


class ObjectProperty(Property):
    def __init__(
        self,
        identifier,
        parent=None,
        is_functional=False,
        is_reflexive=False,
        is_inverse_functional=False,
        is_transitive=False,
        is_symmetric=False,
        domain=None,
        range=None,
        inverse_of=None,
    ):
        if parent is None:
            parent = []
        self.identifier = identifier
        self.parent = parent
        self.is_functional = is_functional
        self.is_inverse_functional = is_inverse_functional
        self.is_transitive = is_transitive
        self.is_symmetric = is_symmetric
        self.is_reflexive = is_reflexive
        self.domain = domain
        self.range = range
        self.inverse_of = inverse_of

    def __str__(self):
        if self.identifier == None and self.inverse_of != None:
            return "{}-".format(self.inverse_of)
        return self.identifier


class Expression(OntologyObject):

    varcount = 0

    @classmethod
    def get_safe_varname(cls):
        cls.varcount += 1
        return "Y%d" % cls.varcount

    expression_types = {}

    @classmethod
    def register_type(cls, tagname, factory):
        cls.expression_types[tagname] = factory

    @classmethod
    def get_factory(cls, tagname):
        return cls.expression_types.get(tagname)

    def __init__(self, identifier=None):
        self.identifier = identifier

    def normalize_rhs(self):
        raise NotImplementedError()

    def normalize_lhs(self):
        return self, []

    def __and__(self, other):
        return Intersection([self, other])

    def __lshift__(self, other):
        return SubClassOf(self, other)


class Rule(OntologyObject):
    """A ontology expression of the form A \sqsubseteq B."""

    def __init__(self, subject, object_):
        """

        :param subject:
        :type subject: Expression
        :param object_:
        :type object_: Expression
        """
        super(Rule, self).__init__()
        self.__subject = subject
        self.__object = object_

    @property
    def subject(self):
        return self.__subject

    @subject.setter
    def subject(self, value):
        self.__subject = value

    @property
    def object(self):
        return self.__object

    @property
    def is_rule(self):
        return True

    def normalize(self):
        """Return a normalization of this rule.

        :return: list of new rules
        """
        raise FeatureNotSupported()


def issimple(expr):
    return type(expr) == ClassIdentifier or type(expr) == Thing


class SubClassOf(Rule):
    def normalize(self):
        lhs, rules_lhs = self.subject.normalize_lhs()
        rhss, rules_rhs = self.object.normalize_rhs()

        rules = []
        for rhs in rhss:
            if issimple(lhs) or issimple(rhs):
                rules.append(SubClassOf(lhs, rhs))
            else:
                name = NameFactory.next_name()
                new_class = ClassIdentifier(name)
                rules.append(SubClassOf(lhs, new_class))
                rules.append(SubClassOf(new_class, rhs))

        rules += rules_lhs
        rules += rules_rhs

        return rules

    def __str__(self):
        return "%s %s %s" % (self.subject, SPECIAL_CHARS["\sqsubseteq"], self.object)
        # return '%s %s %s' % (self.subject, '->', self.object)

    def to_latex(self):
        return r"%s \sqsubseteq %s" % (self.subject.to_latex(), self.object.to_latex())


class EquivalentClass(Rule):
    def normalize(self):
        return (
            SubClassOf(self.subject, self.object).normalize()
            + SubClassOf(self.object, self.subject).normalize()
        )

    def __str__(self):
        return "%s %s %s" % (self.subject, SPECIAL_CHARS["\equiv"], self.object)


class DisjointWith(Rule):
    def normalize(self):
        return [self]
        # raise FeatureNotSupported('rule disjointWith')

    def __str__(self):
        return "%s %s %s" % (self.subject, " disjoint ", self.object)


class Thing(Expression):
    def __init__(self, identifier):
        super(Thing, self).__init__(identifier)

    def normalize_rhs(self):
        return [self], []

    def __str__(self):
        return self.identifier


class ClassIdentifier(Expression):
    """A class in the ontology."""

    def __init__(self, identifier):
        super(ClassIdentifier, self).__init__(identifier)

    def normalize_rhs(self):
        return [self], []

    def __str__(self):
        return self.identifier

    def to_latex(self):
        return r"%s" % self.identifier


class TopClass(Expression):
    """A class in the ontology."""

    def __init__(self):
        super(TopClass, self).__init__()

    def normalize_rhs(self):
        return [self], []

    def __str__(self):
        return SPECIAL_CHARS[r"\top"]

    def to_latex(self):
        return r"\top"


class Restriction(Expression):
    """"""

    def __init__(self, quantifier, prop):
        super(Restriction, self).__init__()
        self.quantifier = quantifier
        self.prop = prop

    def normalize_lhs(self):
        quantifier, rules = self.quantifier.normalize_lhs()
        return Restriction(quantifier, self.prop), rules

    def normalize_rhs(self):
        quantifier, rules = self.quantifier.normalize_rhs()
        return [Restriction(quantifier, self.prop)], rules

    def to_latex(self):
        return self.quantifier.to_latex(self.prop)

    def __str__(self):
        return self.quantifier.to_string(self.prop)


class Intersection(Expression):
    """"""

    def __init__(self, children):
        super(Intersection, self).__init__()
        self.children = children

    def normalize_lhs(self):
        # TODO extend to n-ary intersections, this here assumes binary
        rules = []
        children = []
        for child in self.children:
            nc, nr = child.normalize_lhs()
            rules += nr

            if issimple(nc):
                children.append(nc)
            else:
                name = NameFactory.next_name()
                new_class = ClassIdentifier(name)
                rules.append(SubClassOf(nc, new_class))
                children.append(new_class)

        return Intersection(children), rules

    def normalize_rhs(self):
        rules = []
        children = []
        for child in self.children:
            child_rhs, child_rules = child.normalize_rhs()
            assert len(child_rhs) == 1
            children += child_rhs
            rules += child_rules
        return children, rules

    def __str__(self):
        sep = " %s " % SPECIAL_CHARS["\sqcap"]
        return "(%s)" % sep.join(map(str, self.children))

    def to_latex(self):
        return r" \sqcap ".join(c.to_latex() for c in self.children)

    def __and__(self, other):
        return Intersection(self.children + [other])


class Union(Expression):
    """"""

    def __init__(self, children):
        super(Union, self).__init__()
        self.children = children

    def normalize_rhs(self):
        return [self], []
        # raise FeatureNotSupported("union in RHS")

    def __str__(self):
        sep = " | "
        return sep.join(map(str, self.children))


class Complement(Expression):
    def __init__(self, child):
        super(Complement, self).__init__()
        self.child = child

    def normalize_rhs(self):
        return [self], []
        # raise FeatureNotSupported('complement')

    def __str__(self):
        return "not {}".format(str(self.child))


class OneOf(Expression):
    def __init__(self, children):
        super(OneOf, self).__init__()
        self.children = children

    def normalize_rhs(self):
        raise FeatureNotSupported("oneOf in RHS")

    def __str__(self):
        sep = ", "
        return "one_of(%s)" % sep.join(map(str, self.children))


class Quantifier(object):
    def to_string(self, prop):
        raise NotImplementedError()

    def normalize_rhs(self):
        raise NotImplementedError()


class HasSelf(Quantifier):
    def to_string(self, prop):
        return "hasSelf {}".format(prop)

    def normalize_rhs(self):
        return self, []

    def normalize_lhs(self):
        return self, []


class SomeValues(Quantifier):
    def __init__(self, from_class):
        self.from_class = from_class

    def to_string(self, prop):
        return "%s %s.%s" % (SPECIAL_CHARS["\exists"], prop, self.from_class)

    def to_latex(self, prop):
        if self.from_class.identifier is not None:
            return r"\exists %s.\!%s" % (prop, self.from_class.to_latex())
        else:
            return r"\exists %s.\!(%s)" % (prop, self.from_class.to_latex())

    def normalize_lhs(self):
        rules = []
        if self.from_class.identifier is None:
            name = NameFactory.next_name()
            from_class = ClassIdentifier(name)
            fc_n, fc_r = self.from_class.normalize_lhs()
            rules += fc_r
            for fc in fc_n:
                rules.append(SubClassOf(fc, from_class))
        else:
            from_class = self.from_class
        return SomeValues(from_class), rules

    def normalize_rhs(self):
        rules = []
        if self.from_class.identifier is None:
            name = NameFactory.next_name()
            from_class = ClassIdentifier(name)
            fc_n, fc_r = self.from_class.normalize_rhs()
            rules += fc_r
            for fc in fc_n:
                rules.append(SubClassOf(from_class, fc))
        else:
            from_class = self.from_class
        return SomeValues(from_class), rules


class AllValues(Quantifier):
    def __init__(self, from_class):
        self.from_class = from_class

    def to_string(self, prop):
        return "all %s %s" % (prop, self.from_class)

    def normalize_rhs(self):
        raise FeatureNotSupported("quantifier allValues")


class Cardinality(Quantifier):
    def __init__(self, count):
        self.count = count

    def to_string(self, prop):
        return "count(%s) = %s" % (prop, self.count)

    def normalize_rhs(self):
        raise FeatureNotSupported("quantifier cardinality")


class MinCardinality(Quantifier):
    def __init__(self, count):
        self.count = count

    def to_string(self, prop):
        return "count(%s) >= %s" % (prop, self.count)

    def normalize_rhs(self):
        raise FeatureNotSupported("quantifier minCardinality")


class MaxCardinality(Quantifier):
    def __init__(self, count):
        self.count = count

    def to_string(self, prop):
        return "count(%s) <= %s" % (prop, self.count)

    def normalize_rhs(self):
        raise FeatureNotSupported("quantifier maxCardinality")


class HasValue(Quantifier):
    def __init__(self, value):
        self.value = value

    def to_string(self, prop):
        return "value(%s) = %s" % (prop, self.value)

    def normalize_rhs(self):
        return self, []
        # raise FeatureNotSupported('hasValue')

    def normalize_lhs(self):
        return self, []
