from owlapy import IRI, namespaces
from owlapy.namespaces import Namespaces
from owlapy.model import OWLClass, OWLObjectUnionOf
from owlapy.utils import IRIFixedSet

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


def test_iri_fixed_set():
    fs = IRIFixedSet({IRI.create(base, "C1"), IRI.create(base, "C2")})
    assert IRI.create(base, "C1") in fs
    assert IRI.create(base, "C3") not in fs
    assert fs(IRI.create(base, "C2")) != fs(IRI.create(base, "C1"))
    assert fs(IRI.create(base, "C1")) == fs(IRI.create(base, "C1"))
    assert fs(IRI.create(base, "C3"), ignore_missing=True) == 0
    assert fs(set()) == 0
    assert list(fs(fs(IRI.create(base, "C1")))) == [IRI.create(base, "C1")]


if __name__ == '__main__':
    test_class()
    test_union()
    test_iri_fixed_set()
