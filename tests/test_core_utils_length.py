import unittest

from owlapy.class_expression import OWLObjectUnionOf, OWLObjectComplementOf, OWLObjectIntersectionOf, OWLThing, \
    OWLObjectOneOf, OWLObjectHasValue, OWLObjectMinCardinality, OWLClass, OWLObjectSomeValuesFrom, OWLDataAllValuesFrom, \
    OWLDataExactCardinality, OWLDataHasValue, OWLDataMaxCardinality, OWLDataMinCardinality, OWLDataOneOf, \
    OWLDataSomeValuesFrom
from owlapy.iri import IRI
from owlapy.owl_data_ranges import OWLDataUnionOf, OWLDataComplementOf, OWLDataIntersectionOf
from owlapy.owl_individual import OWLNamedIndividual
from owlapy.owl_literal import OWLLiteral, DoubleOWLDatatype, IntegerOWLDatatype
from owlapy.owl_property import OWLObjectProperty, OWLDataProperty

from ontolearn.base.owl.utils import OWLClassExpressionLengthMetric
from ontolearn.utils import setup_logging
from owlapy.providers import owl_datatype_min_max_inclusive_restriction

setup_logging("ontolearn/logging_test.conf")


class Core_OWLClassExpressionLengthMetric_Test(unittest.TestCase):
    def test_ce_length(self):
        NS = "http://example.com/father#"

        cl = OWLClassExpressionLengthMetric.get_default()

        male = OWLClass(IRI.create(NS, 'male'))
        female = OWLClass(IRI.create(NS, 'female'))
        has_child = OWLObjectProperty(IRI(NS, 'hasChild'))
        has_age = OWLDataProperty(IRI(NS, 'hasAge'))

        ce = OWLObjectUnionOf((male, OWLObjectSomeValuesFrom(property=has_child, filler=female)))
        le = cl.length(ce)
        # male ⊔ (∃ hasChild.female)
        self.assertEqual(le, 5)
        ce = OWLObjectComplementOf(OWLObjectIntersectionOf((female,
                                                            OWLObjectSomeValuesFrom(property=has_child,
                                                                                    filler=OWLThing))))
        le = cl.length(ce)
        # ¬(female ⊓ (∃ hasChild.⊤))
        self.assertEqual(le, 6)
        ce = OWLObjectSomeValuesFrom(property=has_child,
                                     filler=OWLObjectSomeValuesFrom(property=has_child,
                                                                    filler=OWLObjectSomeValuesFrom(property=has_child,
                                                                                                   filler=OWLThing)))
        le = cl.length(ce)
        # ∃ hasChild.(∃ hasChild.(∃ hasChild.⊤))
        self.assertEqual(le, 7)

        i1 = OWLNamedIndividual(IRI.create(NS, 'heinz'))
        i2 = OWLNamedIndividual(IRI.create(NS, 'marie'))
        ce = OWLObjectOneOf((i1, i2))
        le = cl.length(ce)
        # {heinz ⊔ marie}
        self.assertEqual(le, 1)

        ce = OWLObjectHasValue(property=has_child, individual=i1)
        le = cl.length(ce)
        # ∃ hasChild.{heinz}
        self.assertEqual(le, 3)

        ce = OWLObjectMinCardinality(cardinality=2, property=has_child, filler=OWLThing)
        le = cl.length(ce)
        # >= 2 hasChild.⊤
        self.assertEqual(le, 4)

        ce = OWLDataSomeValuesFrom(property=has_age,
                                   filler=OWLDataComplementOf(DoubleOWLDatatype))
        le = cl.length(ce)
        # ∃ hasAge.¬xsd:double
        self.assertEqual(le, 4)

        datatype_restriction = owl_datatype_min_max_inclusive_restriction(40, 80)

        ce = OWLDataSomeValuesFrom(property=has_age, filler=OWLDataUnionOf([datatype_restriction, IntegerOWLDatatype]))
        le = cl.length(ce)
        # ∃ hasAge.(xsd:integer[>= 40 ,<= 80] ⊔ xsd:integer)
        self.assertEqual(le, 6)

        ce = OWLDataAllValuesFrom(property=has_age,
                                  filler=OWLDataIntersectionOf([OWLDataOneOf([OWLLiteral(32.5), OWLLiteral(4.5)]),
                                                                IntegerOWLDatatype]))
        le = cl.length(ce)
        # ∀ hasAge.({32.5 ⊔ 4.5} ⊓ xsd:integer)
        self.assertEqual(le, 5)

        ce = OWLDataHasValue(property=has_age, value=OWLLiteral(50))
        le = cl.length(ce)
        # ∃ hasAge.{50}
        self.assertEqual(le, 3)

        ce = OWLDataExactCardinality(cardinality=1, property=has_age, filler=IntegerOWLDatatype)
        le = cl.length(ce)
        # = 1 hasAge.xsd:integer
        self.assertEqual(le, 4)

        ce = OWLDataMinCardinality(cardinality=5, property=has_age, filler=IntegerOWLDatatype)
        le = cl.length(ce)
        # = 1 hasAge.xsd:integer
        self.assertEqual(le, 4)

        ce = OWLDataMaxCardinality(cardinality=5, property=has_age, filler=IntegerOWLDatatype)
        le = cl.length(ce)
        # = 1 hasAge.xsd:integer
        self.assertEqual(le, 4)


if __name__ == '__main__':
    unittest.main()
