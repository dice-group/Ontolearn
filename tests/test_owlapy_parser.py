import unittest

from owlapy.model import OWLObjectMinCardinality, OWLObjectSomeValuesFrom, OWLObjectUnionOf, \
    DoubleOWLDatatype, IntegerOWLDatatype, OWLClass, IRI, OWLDataAllValuesFrom, OWLDataIntersectionOf, \
    OWLDataOneOf, OWLDataProperty, OWLDataSomeValuesFrom, OWLDatatypeRestriction, OWLFacetRestriction, \
    OWLLiteral, OWLNamedIndividual, OWLObjectAllValuesFrom, OWLObjectComplementOf, OWLObjectExactCardinality, \
    OWLObjectHasSelf, OWLObjectHasValue, OWLObjectIntersectionOf, OWLObjectMaxCardinality, OWLObjectOneOf, \
    OWLObjectProperty
from owlapy.parser import ManchesterSyntaxParser
from owlapy.vocab import OWLFacet


class ManchesterSyntaxParserTest(unittest.TestCase):

    def setUp(self):
        self.namespace = "http://dl-learner.org/mutagenesis#"
        self.parser = ManchesterSyntaxParser(self.namespace)

        # Classes
        self.atom = OWLClass(IRI(self.namespace, 'Atom'))
        self.bond = OWLClass(IRI(self.namespace, 'Bond'))
        self.compound = OWLClass(IRI(self.namespace, 'Compound'))

        # Object Properties
        self.in_bond = OWLObjectProperty(IRI.create(self.namespace, 'inBond'))
        self.has_bond = OWLObjectProperty(IRI.create(self.namespace, 'hasBond'))

        # Data Properties
        self.charge = OWLDataProperty(IRI.create(self.namespace, 'charge'))
        self.act = OWLDataProperty(IRI.create(self.namespace, 'act'))
        self.has_fife_examples = OWLDataProperty(IRI.create(self.namespace, 'hasFifeExamplesOfAcenthrylenes'))

        # Individuals
        self.bond5225 = OWLNamedIndividual(IRI.create(self.namespace, 'bond5225'))
        self.d91_17 = OWLNamedIndividual(IRI.create(self.namespace, 'd91_17'))
        self.d91_32 = OWLNamedIndividual(IRI.create(self.namespace, 'd91_32'))

    def test_union_intersection(self):
        p = self.parser.parse_expression('Atom or Bond and Compound')
        c = OWLObjectUnionOf((self.atom, OWLObjectIntersectionOf((self.bond, self.compound))))
        self.assertEqual(p, c)

        p = self.parser.parse_expression('(Atom or Bond) and Compound')
        c = OWLObjectIntersectionOf((OWLObjectUnionOf((self.atom, self.bond)), self.compound))
        self.assertEqual(p, c)

        p = self.parser.parse_expression('((Atom or Bond) and Atom) and Compound or Bond')
        c = OWLObjectUnionOf((OWLObjectIntersectionOf((OWLObjectIntersectionOf((
                                                            OWLObjectUnionOf((self.atom, self.bond)),
                                                            self.atom)),
                                                       self.compound)),
                              self.bond))
        self.assertEqual(p, c)

    def test_object_properties(self):
        p = self.parser.parse_expression('inBond some Bond')
        c = OWLObjectSomeValuesFrom(self.in_bond, self.bond)
        self.assertEqual(p, c)

        p = self.parser.parse_expression('hasBond only Atom')
        c = OWLObjectAllValuesFrom(self.has_bond, self.atom)
        self.assertEqual(p, c)

        p = self.parser.parse_expression('inBond some (hasBond some (Bond and Atom))')
        c = OWLObjectSomeValuesFrom(self.in_bond,
                                    OWLObjectSomeValuesFrom(self.has_bond,
                                                            OWLObjectIntersectionOf((self.bond, self.atom))))
        self.assertEqual(p, c)

        p = self.parser.parse_expression('inBond max 5 Bond')
        c = OWLObjectMaxCardinality(5, self.in_bond, self.bond)
        self.assertEqual(p, c)

        p = self.parser.parse_expression('inBond min 124 Atom')
        c = OWLObjectMinCardinality(124, self.in_bond, self.atom)
        self.assertEqual(p, c)

        p = self.parser.parse_expression('inBond exactly 11 Bond')
        c = OWLObjectExactCardinality(11, self.in_bond, self.bond)
        self.assertEqual(p, c)

        p = self.parser.parse_expression('inBond value d91_32')
        c = OWLObjectHasValue(self.in_bond, self.d91_32)
        self.assertEqual(p, c)

        p = self.parser.parse_expression('inBond Self')
        c = OWLObjectHasSelf(self.in_bond)
        self.assertEqual(p, c)

        p = self.parser.parse_expression('hasBond only {d91_32, d91_17, bond5225}')
        c = OWLObjectAllValuesFrom(self.has_bond, OWLObjectOneOf((self.d91_32, self.d91_17, self.bond5225)))
        self.assertEqual(p, c)

        p = self.parser.parse_expression('(not (Atom or Bond) and Atom) and not Compound '
                                         'or (hasBond some (inBond max 4 Bond))')
        c1 = OWLObjectIntersectionOf((OWLObjectComplementOf(OWLObjectUnionOf((self.atom, self.bond))), self.atom))
        c2 = OWLObjectIntersectionOf((c1, OWLObjectComplementOf(self.compound)))
        c3 = OWLObjectSomeValuesFrom(self.has_bond, OWLObjectMaxCardinality(4, self.in_bond, self.bond))
        c = OWLObjectUnionOf((c2, c3))
        self.assertEqual(p, c)

    def test_whitespace(self):
        p = self.parser.parse_expression('    inBond   some    Bond')
        c = OWLObjectSomeValuesFrom(self.in_bond, self.bond)
        self.assertEqual(p, c)

        p = self.parser.parse_expression('( \n Atom or Bond\t)  and\nCompound  ')
        c = OWLObjectIntersectionOf((OWLObjectUnionOf((self.atom, self.bond)), self.compound))
        self.assertEqual(p, c)

        p = self.parser.parse_expression('hasBond only { \n\t d91_32,d91_17  ,    bond5225  }')
        c = OWLObjectAllValuesFrom(self.has_bond, OWLObjectOneOf((self.d91_32, self.d91_17, self.bond5225)))
        self.assertEqual(p, c)

        p = self.parser.parse_expression('act only { \n\t 1.2f  ,    3.2f  }')
        c = OWLDataAllValuesFrom(self.act, OWLDataOneOf((OWLLiteral(1.2), OWLLiteral(3.2))))
        self.assertEqual(p, c)

        p = self.parser.parse_expression('act some (  xsd:double[  > 5f,< 4.2f \n, <  -1.8e10f  ]\t and  integer )')
        f1 = OWLFacetRestriction(OWLFacet.MIN_EXCLUSIVE, OWLLiteral(5.0))
        f2 = OWLFacetRestriction(OWLFacet.MAX_EXCLUSIVE, OWLLiteral(4.2))
        f3 = OWLFacetRestriction(OWLFacet.MAX_EXCLUSIVE, OWLLiteral(-1.8e10))
        c = OWLDataSomeValuesFrom(self.act, OWLDataIntersectionOf(
                                    (OWLDatatypeRestriction(DoubleOWLDatatype, (f1, f2, f3)), IntegerOWLDatatype)))
        self.assertEqual(p, c)
