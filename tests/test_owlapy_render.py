import unittest

from owlapy import IRI
from owlapy.model import OWLClass, OWLObjectProperty, OWLObjectIntersectionOf, OWLObjectSomeValuesFrom, \
    OWLThing, OWLObjectComplementOf, OWLObjectUnionOf, OWLNamedIndividual, OWLObjectOneOf, OWLObjectHasValue, \
    OWLObjectMinCardinality
from owlapy.render import DLSyntaxRenderer


class Owlapy_DLRenderer_Test(unittest.TestCase):
    def test_ce_render(self):
        renderer = DLSyntaxRenderer()
        NS = "http://example.com/father#"

        male = OWLClass(IRI.create(NS, 'male'))
        female = OWLClass(IRI.create(NS, 'female'))
        has_child = OWLObjectProperty(IRI(NS, 'hasChild'))

        c = OWLObjectUnionOf((male, OWLObjectSomeValuesFrom(property=has_child, filler=female)))
        r = renderer.render(c)
        print(r)
        self.assertEqual(r, "male ⊔ (∃ hasChild.female)")
        c = OWLObjectComplementOf(OWLObjectIntersectionOf((female,
                                                           OWLObjectSomeValuesFrom(property=has_child,
                                                                                   filler=OWLThing))))
        r = renderer.render(c)
        print(r)
        self.assertEqual(r, "¬(female ⊓ (∃ hasChild.⊤))")
        c = OWLObjectSomeValuesFrom(property=has_child,
                                    filler=OWLObjectSomeValuesFrom(property=has_child,
                                                                   filler=OWLObjectSomeValuesFrom(property=has_child,
                                                                                                  filler=OWLThing)))
        r = renderer.render(c)
        print(r)
        self.assertEqual(r, "∃ hasChild.(∃ hasChild.(∃ hasChild.⊤))")

        i1 = OWLNamedIndividual(IRI.create(NS, 'heinz'))
        i2 = OWLNamedIndividual(IRI.create(NS, 'marie'))
        oneof = OWLObjectOneOf((i1, i2))
        r = renderer.render(oneof)
        print(r)
        self.assertEqual(r, "{heinz ⊔ marie}")

        hasvalue = OWLObjectHasValue(property=has_child, value=i1)
        r = renderer.render(hasvalue)
        print(r)
        self.assertEqual(r, "∃ hasChild.{heinz}")

        mincard = OWLObjectMinCardinality(property=has_child, cardinality=2, filler=OWLThing)
        r = renderer.render(mincard)
        print(r)
        self.assertEqual(r, "≥ 2 hasChild.⊤")


if __name__ == '__main__':
    unittest.main()
