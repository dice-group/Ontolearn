import unittest

from ontolearn.core.owl.utils import OWLClassExpressionLengthMetric
from ontolearn.owlapy import IRI
from ontolearn.owlapy.model import OWLClass, OWLObjectProperty, OWLObjectUnionOf, OWLObjectSomeValuesFrom, \
    OWLObjectComplementOf, OWLObjectIntersectionOf, OWLThing, OWLNamedIndividual, OWLObjectOneOf, OWLObjectHasValue, \
    OWLObjectMinCardinality


class Core_OWLClassExpressionLengthMetric_Test(unittest.TestCase):
    def test_ce_length(self):
        NS = "http://example.com/father#"

        cl = OWLClassExpressionLengthMetric.get_default()

        male = OWLClass(IRI.create(NS, 'male'))
        female = OWLClass(IRI.create(NS, 'female'))
        has_child = OWLObjectProperty(IRI(NS, 'hasChild'))

        c = OWLObjectUnionOf((male, OWLObjectSomeValuesFrom(property=has_child, filler=female)))
        l = cl.length(c)
        # male ⊔ (∃ hasChild.female)
        self.assertEqual(l, 5)
        c = OWLObjectComplementOf(OWLObjectIntersectionOf((female,
                                                           OWLObjectSomeValuesFrom(property=has_child,
                                                                                   filler=OWLThing))))
        l = cl.length(c)
        # ¬(female ⊓ (∃ hasChild.⊤))
        self.assertEqual(l, 6)
        c = OWLObjectSomeValuesFrom(property=has_child,
                                    filler=OWLObjectSomeValuesFrom(property=has_child,
                                                                   filler=OWLObjectSomeValuesFrom(property=has_child,
                                                                                                  filler=OWLThing)))
        l = cl.length(c)
        # ∃ hasChild.(∃ hasChild.(∃ hasChild.⊤))
        self.assertEqual(l, 7)

        i1 = OWLNamedIndividual(IRI.create(NS, 'heinz'))
        i2 = OWLNamedIndividual(IRI.create(NS, 'marie'))
        c = OWLObjectOneOf((i1, i2))
        # {heinz ⊔ marie}
        self.assertEqual(l, 1)

        c = OWLObjectHasValue(property=has_child, value=i1)
        l = cl.length(c)
        # ∃ hasChild.{heinz}
        self.assertEqual(l, 3)

        c = OWLObjectMinCardinality(property=has_child, cardinality=2, filler=OWLThing)
        l = cl.length(c)
        # ≥ 2 hasChild.⊤
        self.assertEqual(l, 4)


if __name__ == '__main__':
    unittest.main()
