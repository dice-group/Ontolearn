import unittest

from ontolearn.owlapy import IRI
from ontolearn.owlapy.fast_instance_checker import OWLReasoner_FastInstanceChecker
from ontolearn.owlapy.model import OWLClass, OWLObjectProperty, OWLNamedIndividual, OWLObjectIntersectionOf, \
    OWLObjectSomeValuesFrom, OWLThing, OWLTopObjectProperty
from ontolearn.owlapy.owlready2 import OWLOntologyManager_Owlready2, OWLReasoner_Owlready2


class Owlapy_FastInstanceChecker_Test(unittest.TestCase):
    def test_instances(self):
        NS = "http://example.com/father#"
        mgr = OWLOntologyManager_Owlready2()
        onto = mgr.load_ontology(IRI.create("file://data/father.owl"))

        male = OWLClass(IRI.create(NS, 'male'))
        female = OWLClass(IRI.create(NS, 'female'))
        has_child = OWLObjectProperty(IRI(NS, 'hasChild'))

        base_reasoner = OWLReasoner_Owlready2(onto)
        reasoner = OWLReasoner_FastInstanceChecker(onto, base_reasoner=base_reasoner)

        self.assertEqual([], list(reasoner.sub_object_properties(has_child, direct=True)))

        inst = frozenset(reasoner.instances(female))
        target_inst = frozenset({OWLNamedIndividual(IRI(NS, 'anna')),
                                 OWLNamedIndividual(IRI(NS, 'michelle'))})
        self.assertEqual(inst, target_inst)

        inst = frozenset(reasoner.instances(
            OWLObjectIntersectionOf((male, OWLObjectSomeValuesFrom(property=has_child, filler=female)))))
        target_inst = frozenset({OWLNamedIndividual(IRI(NS, 'markus'))})
        self.assertEqual(inst, target_inst)

        inst = frozenset(reasoner.instances(
            OWLObjectIntersectionOf((female, OWLObjectSomeValuesFrom(property=has_child, filler=OWLThing)))))
        target_inst = frozenset({OWLNamedIndividual(IRI(NS, 'anna'))})
        self.assertEqual(inst, target_inst)

        inst = frozenset(reasoner.instances(
            OWLObjectSomeValuesFrom(property=has_child,
                                    filler=OWLObjectSomeValuesFrom(property=has_child,
                                                                   filler=OWLObjectSomeValuesFrom(property=has_child,
                                                                                                  filler=OWLThing)))))
        target_inst = frozenset({OWLNamedIndividual(IRI(NS, 'stefan'))})
        self.assertEqual(inst, target_inst)


if __name__ == '__main__':
    unittest.main()
