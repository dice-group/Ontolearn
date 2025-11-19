import sys

from lxml.etree import ElementTree

from .o2p_ontology import *

#  - TODO don't skip useful tags (see 'skipping')
#  - TODO parse DataRange


def print_element(element):
    """Pretty print an XML element (for debugging purposes).

    :param element: element to print
    :type element: xml.etree.ElementTree.Element
    """
    import lxml

    print(lxml.etree.tostring(element, pretty_print=True).decode(), file=sys.stderr)

def make_res_absolute(elem, res):
    if res[0] == "#":
        return elem.nsmap[None] + res[1:]
    else:
        return res

class OWLReader(object):
    namespaces = {
        "owl": "http://www.w3.org/2002/07/owl#",
        "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
        "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    }

    @classmethod
    def expand_namespace(cls, namespace, item):
        """Expand a namespace alias to the fully qualified name.

        :param namespace: a namespace alias (should be in namespaces).
        :param item: the name of the item in the namespace
        :return: fully qualified namespace
        """
        if namespace is None:
            return item
        else:
            return "{%s}%s" % (cls.namespaces.get(namespace), item)

    @classmethod
    def extract_namespace(cls, item):
        return item.split("}", 1)[0].strip("{")

    def register_factory(self, namespace, tagname, factory):
        self.factories[self.expand_namespace(namespace, tagname)] = factory

    def get_factory(self, tagname):
        return self.factories[tagname]

    def register_factories(self):
        self.register_factory("rdfs", "subClassOf", self.parse_subclass)
        self.register_factory("owl", "disjointWith", self.parse_disjoint)
        self.register_factory("owl", "equivalentClass", self.parse_eqclass)
        self.register_factory("owl", "Class", self.parse_class)
        self.register_factory("owl", "Restriction", self.parse_restriction)
        self.register_factory("owl", "intersectionOf", self.parse_intersection)
        self.register_factory("owl", "complementOf", self.parse_complement)
        self.register_factory("owl", "unionOf", self.parse_union)
        self.register_factory("rdf", "Description", self.parse_class)
        self.register_factory("owl", "oneOf", self.parse_one_of)
        self.register_factory("owl", "Thing", self.parse_thing)

        self.property_types = {
            self.expand_namespace("owl", "ObjectProperty"): {},
            self.expand_namespace("owl", "FunctionalProperty"): {"is_functional": True},
            self.expand_namespace("owl", "FunctionalObjectProperty"): {
                "is_functional": True
            },
            self.expand_namespace("owl", "TransitiveProperty"): {"is_transitive": True},
            self.expand_namespace("owl", "InverseFunctionalProperty"): {
                "is_inverse_functional": True
            },
            self.expand_namespace("owl", "DatatypeProperty"): {},
        }

        self.ignore = {
            self.expand_namespace("rdfs", "comment"),
            self.expand_namespace("rdfs", "label"),
            # self.expand_namespace('rdf', 'Description'),
            self.expand_namespace("rdfs", "Datatype"),
        }
        self.skipping = {
            self.expand_namespace("owl", "Ontology"),
            # self.expand_namespace('owl', 'DatatypeProperty'),
            self.expand_namespace("owl", "Axiom"),
            self.expand_namespace("owl", "NamedIndividual"),
            self.expand_namespace("owl", "DataRange"),
            self.expand_namespace("owl", "AllDifferent"),
            self.expand_namespace("owl", "AnnotationProperty"),
            # self.expand_namespace('owl', 'Prefix'),
            # self.expand_namespace('owl', 'Declaration'),
            # self.expand_namespace('owl', 'EquivalentClasses'),
            # self.expand_namespace('owl', 'SubClassOf'),
            # self.expand_namespace('owl', 'DisjointClasses'),
            # self.expand_namespace('owl', 'ClassAssertion'),
            # self.expand_namespace('owl', 'SameIndividual'),
            # self.expand_namespace('owl', 'DifferentIndividuals'),
            # self.expand_namespace('owl', 'ObjectPropertyAssertion'),
            # self.expand_namespace('owl', 'SubObjectPropertyOf'),
            # self.expand_namespace('owl', 'InverseObjectProperties'),
        }

    def __init__(self, filename, verbose=0, strictness=100):
        self.filename = filename
        self.factories = {}
        self.property_types = {}
        self.skipping = {}
        self.ignore = {}
        self.register_factories()
        self.verbose = verbose
        self.strictness = strictness

    def parse_error(self, severity, element, message):
        if self.strictness < severity:
            print_element(element)
            raise RuntimeError(message)
        elif self.verbose >= severity:
            print("WARNING:", message, file=sys.stderr)



    def read(self):
        ontology = Ontology()

        tree = ElementTree.parse(self.filename)

        tag_class = self.expand_namespace("owl", "Class")
        tag_properties = self.property_types

        for element in tree.getroot():
            # Parse all elements
            if element.tag == tag_class:
                rules = self.parse_rule(element)
                for rule in rules:
                    if rule is not None:
                        ontology.add_rule(rule)
            elif element.tag in self.skipping:
                self.parse_error(
                    1, element, "Unknown top-level element: %s" % element.tag
                )
            elif element.tag in self.ignore:
                pass
            elif element.tag in tag_properties:
                prop = self.parse_property(element)
                ontology.add_property(prop)
            elif self.extract_namespace(element.tag) not in self.namespaces.values():
                self.parse_error(
                    2,
                    element,
                    "Unknown top-level element in external namespace: %s" % element.tag,
                )
            else:
                # other elements
                self.parse_error(
                    3, element, "Unknown top-level element: %s" % element.tag
                )

        return ontology

    def parse_rule(self, element):
        # Get the identifier.
        identifier = self._one_of(element, ("rdf", "ID"), ("rdf", "about"))

        # Parse the children.
        children = [self.parse_element(child) for child in element]

        # Separate rules from non-rules.
        children_rules = [
            child for child in children if child is not None and child.is_rule
        ]
        children_not_rules = [
            child for child in children if child is not None and not child.is_rule
        ]

        # There can be at most one non-rule child.
        assert len(children_not_rules) <= 1

        if identifier is not None:
            subject = ClassIdentifier(make_res_absolute(element, identifier))
            if children_not_rules:
                children_rules.append(EquivalentClass(subject, children_not_rules[0]))
        else:
            assert len(children_not_rules) == 1
            subject = children_not_rules[0]

        results = []
        for child in children_rules:
            child.subject = subject
            results.append(child)
        return results

    def parse_thing(self, element):
        identifier = self._one_of(element, ("rdf", "ID"), ("rdf", "about"))
        return Thing(identifier)

    def parse_element(self, element):
        try:
            factory = self.get_factory(element.tag)
            return factory(element)
        except KeyError:
            if not element.tag in self.ignore:
                self.parse_error(1, element, "Unknown element: %s" % element.tag)
            return None

    def parse_disjoint(self, element):
        bs = []
        try:
            attrib_ref = self.expand_namespace("rdf", "resource")
            ref = make_res_absolute(element, element.attrib[attrib_ref])
            bs.append(ClassIdentifier(ref))
        except KeyError:
            pass
        for child in element:
            bs.append(self.parse_element(child))

        if len(bs) != 1:
            raise RuntimeError("rdfs:disjointWith can only have one child!")
        return DisjointWith(None, bs[0])

    def parse_subclass(self, element):
        bs = []
        try:
            attrib_ref = self.expand_namespace("rdf", "resource")
            ref = make_res_absolute(element, element.attrib[attrib_ref])
            bs.append(ClassIdentifier(ref))
        except KeyError:
            pass
        for child in element:
            bs.append(self.parse_element(child))

        if len(bs) != 1:
            raise RuntimeError("rdfs:subClassOf can only have one child!")
        return SubClassOf(None, bs[0])

    def parse_eqclass(self, element):
        bs = []
        try:
            attrib_ref = self.expand_namespace("rdf", "resource")
            ref = make_res_absolute(element, element.attrib[attrib_ref])
            bs.append(ClassIdentifier(ref))
        except KeyError:
            pass
        for child in element:
            bs.append(self.parse_element(child))

        if len(bs) != 1:
            raise RuntimeError("rdfs:equivalentClass can only have one child!")
        return EquivalentClass(None, bs[0])

    def parse_class(self, element):
        identifier = self._one_of(element, ("rdf", "ID"), ("rdf", "about"))

        # Parse the children.
        children = [self.parse_element(child) for child in element]

        # Separate rules from non-rules.
        children_rules = [
            child for child in children if child is not None and child.is_rule
        ]
        children_not_rules = [
            child for child in children if child is not None and not child.is_rule
        ]

        # A child can not contain rules.
        assert not children_rules

        # There can be at most one non-rule child.
        assert len(children_not_rules) <= 1

        if identifier is not None:
            subject = ClassIdentifier(make_res_absolute(element, identifier))
        else:
            assert len(children_not_rules) == 1
            subject = children_not_rules[0]

        return subject

    def parse_restriction(self, element):
        op = self.expand_namespace("owl", "onProperty")

        quantifier = self.parse_quantifier(element)
        property_wrap = element.find(op)

        prop = self.parse_propertyref(property_wrap)
        if prop is None:
            print_element(element)
            raise RuntimeError("Can't find property")
        return Restriction(quantifier, prop)

    def parse_datarange(self, element):
        # TODO implement
        return []

    def parse_intersection(self, element):
        return self.parse_collection(element, Intersection)

    def parse_complement(self, element):
        bs = []
        try:
            attrib_ref = self.expand_namespace("rdf", "resource")
            ref = make_res_absolute(element, element.attrib[attrib_ref])
            bs.append(ClassIdentifier(ref))
        except KeyError:
            pass
        for child in element:
            bs.append(self.parse_element(child))

        if len(bs) != 1:
            raise RuntimeError("owl:complementOf can only have one child!")
        return Complement(bs[0])

    def parse_union(self, element):
        return self.parse_collection(element, Union)

    def parse_one_of(self, element):
        return self.parse_collection(element, OneOf)

    def parse_collection(self, element, cls):
        objs = []
        for child in element:
            obj = self.parse_element(child)
            if obj is not None:
                objs.append(obj)

        if not objs:
            print_element(element)
            raise RuntimeError("Collection can't be empty!")

        return cls(objs)

    def parse_property_rule(self, element):
        pass

    def parse_classref(self, element):
        rdf_resource = self.expand_namespace("rdf", "resource")

        results = []
        try:
            results.append(ClassIdentifier(make_res_absolute(element, element.attrib[rdf_resource])))
        except KeyError:
            pass

        for child in element:
            if child.tag == self.expand_namespace("owl", "Class"):
                results.append(self.parse_class(child))
            else:
                print_element(element)
                raise RuntimeError("Unexpected child!")

        if len(results) == 0:
            print_element(element)
            raise RuntimeError("Can't find class reference!")
        elif len(results) > 1:
            print_element(element)
            raise RuntimeError("Too many class references!")

        return results[0]

    def parse_propertyref(self, element):
        rdf_resource = self.expand_namespace("rdf", "resource")

        results = []
        try:
            results.append(PropertyReference(make_res_absolute(element, element.attrib[rdf_resource])))
        except KeyError:
            pass

        for child in element:
            prop = self.parse_property(child)
            if prop is None:
                print_element(element)
                raise RuntimeError("Unexpected child!")
            else:
                results.append(prop)

        if len(results) == 0:
            print_element(element)
            raise RuntimeError("Can't find property reference!")
        elif len(results) > 1:
            print_element(element)
            raise RuntimeError("Too many property references!")

        return results[0]

    def parse_property(self, element):
        if element.tag not in self.property_types and "Description" not in element.tag:
            return None

        identifier = self._one_of(
            element, ("rdf", "about"), ("rdf", "ID"), (None, "IRI")
        )
        if identifier is not None:
            identifier = make_res_absolute(element, identifier)
            # self.parse_error(5, element, 'No property identifier found!')

        rdf_resource = self.expand_namespace("rdf", "resource")

        if "Description" in element.tag:
            options = dict()
        else:
            # Get the property options - very important to make a copy!
            options = dict(self.property_types[element.tag])

        # Process all the children
        parents = []
        for child in element:
            if child.tag == self.expand_namespace("rdf", "type"):
                try:
                    prop_type = child.attrib[rdf_resource]
                    if prop_type.endswith("InverseFunctionalProperty"):
                        options["is_inverse_functional"] = True
                    elif prop_type.endswith("FunctionalProperty"):
                        options["is_functional"] = True
                    elif prop_type.endswith("TransitiveProperty"):
                        options["is_transitive"] = True
                    elif prop_type.endswith("SymmetricProperty"):
                        options["is_symmetric"] = True
                    elif prop_type.endswith("ReflexiveProperty"):
                        options["is_reflexive"] = True
                    elif prop_type.endswith("DatatypeProperty"):
                        pass  # TODO skipping DatatypeProperty
                    elif prop_type.endswith("ObjectProperty"):
                        pass
                    elif prop_type.endswith("#Property"):
                        pass
                    else:
                        self.parse_error(
                            2, element, "Unhandled property type: '%s'" % prop_type
                        )
                except KeyError:
                    print_element(child)
                    raise RuntimeError("Missing identifier!")
            elif child.tag == self.expand_namespace("rdfs", "subPropertyOf"):
                parents.append(self.parse_subpropertyof(child))
            elif child.tag == self.expand_namespace("rdfs", "domain"):
                domain = self.parse_classref(child)
                options["domain"] = domain
            elif child.tag == self.expand_namespace("rdfs", "range"):
                domain = self.parse_classref(child)
                options["range"] = domain
            elif child.tag == self.expand_namespace("owl", "inverseOf"):
                inverse = self.parse_propertyref(child)
                options["inverse_of"] = inverse
            elif child.tag not in self.ignore:
                self.parse_error(
                    2, element, "Unhandled property modifier: '%s'" % child.tag
                )

        return ObjectProperty(identifier, parents, **options)

    def parse_subpropertyof(self, element):
        rdf_resource = self.expand_namespace("rdf", "resource")
        results = []
        try:
            results.append(PropertyReference(make_res_absolute(element, element.attrib[rdf_resource])))
        except KeyError:
            pass
        for child in element:
            prop = self.parse_property(child)
            if not prop:
                print_element(child)
                raise RuntimeError("Expected property!")
            results.append(prop)

        assert len(results) == 1

        return results[0]

    def parse_quantifier(self, element):
        # Input is a Restriction

        possible_quantifiers = {
            ("owl", "someValuesFrom"): self.parse_quantifier_exists,
            ("owl", "allValuesFrom"): self.parse_quantifier_all,
            ("owl", "cardinality"): self.parse_quantifier_card,
            ("owl", "minCardinality"): self.parse_quantifier_mincard,
            ("owl", "maxCardinality"): self.parse_quantifier_maxcard,
            ("owl", "hasValue"): self.parse_quantifier_has_value,
            (
                "owl",
                "qualifiedCardinality",
            ): self.parse_quantifier_qualified_cardinality,
            (
                "owl",
                "maxQualifiedCardinality",
            ): self.parse_quantifier_min_qualified_cardinality,
            (
                "owl",
                "minQualifiedCardinality",
            ): self.parse_quantifier_max_qualified_cardinality,
            ("owl", "hasSelf"): self.parse_quantifier_self,
        }

        quantifiers = []
        for name, func in possible_quantifiers.items():
            s = element.find(self.expand_namespace(*name))
            if s is not None:
                value = func(s)
                quantifiers.append(value)

        if len(quantifiers) != 1:
            print_element(element)
            raise RuntimeError("Expecting exactly one quantifier!")
        return quantifiers[0]

    def parse_quantifier_self(self, element):
        return HasSelf()

    def parse_quantifier_qualified_cardinality(self, element):
        print("QualifiedCardinality")
        return Cardinality(0)

    def parse_quantifier_min_qualified_cardinality(self, element):
        print("MinQualifiedCardinality")
        return Cardinality(0)

    def parse_quantifier_max_qualified_cardinality(self, element):
        print("MaxQualifiedCardinality")
        return Cardinality(0)

    def _parse_quantifier_from(self, element, cls):
        identifier = self._one_of(element, ("rdf", "resource"), ("rdf", "nodeID"))

        if identifier is None:
            options = []
            if element.text is not None and element.text.strip():
                options.append(element.text.strip())
            for child in element:
                obj = self.parse_element(child)
                if obj is not None:
                    options.append(obj)
            if len(options) != 1:
                print_element(element)
                print(options)
                raise RuntimeError("Value or class required!")
            identifier = options[0]
        else:
            identifier = ClassIdentifier(make_res_absolute(element, identifier))

        return cls(identifier)

    def parse_quantifier_exists(self, element):
        return self._parse_quantifier_from(element, SomeValues)

    def parse_quantifier_all(self, element):
        return self._parse_quantifier_from(element, AllValues)

    def _parse_quantifier_card(self, element, cls):
        count = int(element.text)
        return cls(count)

    def parse_quantifier_card(self, element):
        return self._parse_quantifier_card(element, Cardinality)

    def parse_quantifier_mincard(self, element):
        return self._parse_quantifier_card(element, MinCardinality)

    def parse_quantifier_maxcard(self, element):
        return self._parse_quantifier_card(element, MaxCardinality)

    def parse_quantifier_has_value(self, element):
        return self._parse_quantifier_from(element, HasValue)
    

    def _one_of(self, element, *attrs):
        """

        :param element:
        :type element: xml.etree.ElementTree.Element
        :param attrs:
        :return:
        """
        results = []
        for attr in attrs:
            attr_name = self.expand_namespace(*attr)
            try:
                results.append(element.attrib[attr_name])
            except KeyError:
                pass

        if len(results) == 0:
            return None
        elif len(results) > 1:
            raise RuntimeError("Only one match allowed: %s" % results)
        else:
            return results[0]
