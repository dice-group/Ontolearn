import unittest

from ontolearn.owlapy import IRI
from ontolearn.owlapy.model import OWLClass, OWLObjectProperty, OWLObjectIntersectionOf, OWLObjectSomeValuesFrom, \
    OWLThing, OWLObjectComplementOf, OWLObjectUnionOf
from ontolearn.owlapy.render import DLSyntaxRenderer


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


if __name__ == '__main__':
    unittest.main()
