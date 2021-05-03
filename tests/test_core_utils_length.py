import unittest

from ontolearn.core.owl.utils import OWLClassExpressionLengthMetric
from ontolearn.utils import setup_logging
from owlapy import IRI
from owlapy.model import OWLClass, OWLObjectProperty, OWLObjectUnionOf, OWLObjectSomeValuesFrom, \
    OWLObjectComplementOf, OWLObjectIntersectionOf, OWLThing, OWLNamedIndividual, OWLObjectOneOf, OWLObjectHasValue, \
    OWLObjectMinCardinality

setup_logging("logging_test.conf")


class Core_OWLClassExpressionLengthMetric_Test(unittest.TestCase):
    def test_ce_length(self):
        NS = "http://example.com/father#"

        cl = OWLClassExpressionLengthMetric.get_default()

        male = OWLClass(IRI.create(NS, 'male'))
        female = OWLClass(IRI.create(NS, 'female'))
        has_child = OWLObjectProperty(IRI(NS, 'hasChild'))

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

        ce = OWLObjectHasValue(property=has_child, value=i1)
        le = cl.length(ce)
        # ∃ hasChild.{heinz}
        self.assertEqual(le, 3)

        ce = OWLObjectMinCardinality(cardinality=2, property=has_child, filler=OWLThing)
        le = cl.length(ce)
        # ≥ 2 hasChild.⊤
        self.assertEqual(le, 4)


if __name__ == '__main__':
    unittest.main()
