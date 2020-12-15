import unittest

from ontolearn.core.owl import ClassHierarchy
from ontolearn.owlapy import IRI
from ontolearn.owlapy.model import OWLClass
from ontolearn.owlapy.owlready2 import OWLOntologyManager_Owlready2, OWLReasoner_Owlready2


class Owl_Core_ClassHierarchy_Test(unittest.TestCase):
    # TODO
    # def test_class_hierarchy_circle(self):
    #     NS = "http://example.org/circle#"
    #     a1 = OWLClass(IRI.create(NS, 'A1'))
    #     a3 = OWLClass(IRI.create(NS, 'A3'))
    #     a2 = OWLClass(IRI.create(NS, 'A2'))
    #     a0 = OWLClass(IRI.create(NS, 'A0'))
    #     a4 = OWLClass(IRI.create(NS, 'A4'))
    #     ch = ClassHierarchy({a1: [a2],
    #                          a2: [a3],
    #                          a3: [a1, a4],
    #                          a0: [a1],
    #                          a4: []}.items())
    #     for k in sorted(ch.roots()):
    #         _print_children(ch, k)

    def test_class_hierarchy_restrict(self):
        NS = "http://www.benchmark.org/family#"
        mgr = OWLOntologyManager_Owlready2()
        onto = mgr.load_ontology(IRI.create("file://data/family-benchmark_rich_background.owl"))
        reasoner = OWLReasoner_Owlready2(onto)

        ch = ClassHierarchy(reasoner).restrict_and_copy(remove=frozenset({OWLClass(IRI(NS, 'Grandchild'))}))

        target_cls = frozenset({OWLClass(IRI(NS, 'Daughter')),
                                OWLClass(IRI(NS, 'Granddaughter')),
                                OWLClass(IRI(NS, 'Grandson')),
                                OWLClass(IRI(NS, 'Son'))})
        self.assertEqual(frozenset(ch.children(OWLClass(IRI(NS, 'Child')))), target_cls)

    def test_class_hierarchy2(self):
        NS = "http://example.com/father#"
        mgr = OWLOntologyManager_Owlready2()
        onto = mgr.load_ontology(IRI.create("file://data/father.owl"))
        reasoner = OWLReasoner_Owlready2(onto)

        ch = ClassHierarchy(reasoner)
        for k in sorted(ch.roots()):
            _print_children(ch, k)

        target_cls = frozenset({OWLClass(IRI(NS, 'female')),
                                OWLClass(IRI(NS, 'male'))})
        self.assertEqual(frozenset(ch.children(OWLClass(IRI(NS, 'person')))), target_cls)

    def test_class_hierarchy(self):
        NS = "http://www.benchmark.org/family#"
        mgr = OWLOntologyManager_Owlready2()
        onto = mgr.load_ontology(IRI.create("file://data/family-benchmark_rich_background.owl"))
        reasoner = OWLReasoner_Owlready2(onto)

        ch = ClassHierarchy(reasoner)
        grandmother = OWLClass(IRI(NS, 'Grandmother'))
        _print_parents(ch, grandmother)

        target_cls = frozenset({OWLClass(IRI(NS, 'Female')),
                                OWLClass(IRI(NS, 'Grandparent'))})
        self.assertEqual(frozenset(ch.parents(grandmother)), target_cls)


def _print_children(ch: ClassHierarchy, c: OWLClass, level: int = 0) -> None:
    print(' ' * 2 * level, c, '=>')
    for d in sorted(ch.children(c)):
        _print_children(ch, d, level + 1)


def _print_parents(ch: ClassHierarchy, c: OWLClass, level: int = 0) -> None:
    print(' ' * 2 * level, c, '<=')
    for d in sorted(ch.parents(c)):
        _print_parents(ch, d, level + 1)


if __name__ == '__main__':
    unittest.main()
