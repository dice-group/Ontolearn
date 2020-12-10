from ontolearn.owlapy import IRI, namespaces
from ontolearn.owlapy.namespaces import Namespaces
from ontolearn.owlapy.model import OWLClass, OWLObjectUnionOf

base = Namespaces("ex", "http://example.org/")


def test_class():
    c1 = OWLClass(IRI(base, "C1"))
    c2 = OWLClass(IRI(base, "C2"))
    c1x = OWLClass(IRI(base, "C1"))
    thing = OWLClass(IRI(namespaces.OWL, "Thing"))
    assert thing.is_owl_thing()
    assert c1 == c1x
    assert c2 != c1


def test_union():
    c1 = OWLClass(IRI(base, "C1"))
    c2 = OWLClass(IRI(base, "C2"))
    c3 = OWLObjectUnionOf((c1, c2))
    assert list(c3.operands()) == [c1, c2]


# test_class()
# test_union()
