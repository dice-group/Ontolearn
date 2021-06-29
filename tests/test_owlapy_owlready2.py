import unittest

import owlapy.owlready2.utils
from owlapy.model import OWLObjectProperty, OWLNamedIndividual, OWLThing, OWLClass, OWLObjectUnionOf, \
    OWLObjectIntersectionOf, OWLObjectSomeValuesFrom, OWLObjectComplementOf, IRI
from owlapy.owlready2 import OWLOntologyManager_Owlready2, OWLReasoner_Owlready2
from owlapy.owlready2.temp_classes import OWLReasoner_Owlready2_TempClasses


class Owlapy_Owlready2_Test(unittest.TestCase):
    def test_sub_classes(self):
        NS = "http://example.com/father#"
        mgr = OWLOntologyManager_Owlready2()
        onto = mgr.load_ontology(IRI.create("file://KGs/father.owl"))
        reasoner = OWLReasoner_Owlready2(onto)

        classes = frozenset(reasoner.sub_classes(OWLClass(IRI.create(NS, "person"))))
        target_classes = frozenset((OWLClass(IRI.create(NS, "male")), OWLClass(IRI.create(NS, "female"))))
        self.assertEqual(classes, target_classes)

    def test_sub_object_properties(self):
        NS = "http://www.biopax.org/examples/glycolysis#"
        mgr = OWLOntologyManager_Owlready2()
        onto = mgr.load_ontology(IRI.create("file://KGs/Biopax/biopax.owl"))
        reasoner = OWLReasoner_Owlready2(onto)

        participants = OWLObjectProperty(IRI.create(NS, 'PARTICIPANTS'))
        target_props = frozenset({OWLObjectProperty(IRI(NS, 'COFACTOR')),
                                  OWLObjectProperty(IRI(NS, 'CONTROLLED')),
                                  OWLObjectProperty(IRI(NS, 'CONTROLLER')),
                                  OWLObjectProperty(IRI(NS, 'LEFT')),
                                  OWLObjectProperty(IRI(NS, 'RIGHT'))})
        self.assertEqual(frozenset(reasoner.sub_object_properties(participants, direct=True)), target_props)
        self.assertEqual(frozenset(reasoner.sub_object_properties(participants, direct=False)), target_props)

    def test_instances(self):
        NS = "http://example.com/father#"
        mgr = OWLOntologyManager_Owlready2()
        onto = mgr.load_ontology(IRI.create("file://KGs/father.owl"))
        reasoner = OWLReasoner_Owlready2(onto)

        inst = frozenset(reasoner.instances(OWLThing))
        target_inst = frozenset({OWLNamedIndividual(IRI(NS, 'anna')),
                                 OWLNamedIndividual(IRI(NS, 'heinz')),
                                 OWLNamedIndividual(IRI(NS, 'markus')),
                                 OWLNamedIndividual(IRI(NS, 'martin')),
                                 OWLNamedIndividual(IRI(NS, 'michelle')),
                                 OWLNamedIndividual(IRI(NS, 'stefan'))})
        self.assertEqual(inst, target_inst)

        inst = frozenset(reasoner.instances(OWLClass(IRI.create(NS, "male"))))
        target_inst = frozenset({OWLNamedIndividual(IRI(NS, 'heinz')), OWLNamedIndividual(IRI(NS, 'martin')),
                                 OWLNamedIndividual(IRI(NS, 'markus')), OWLNamedIndividual(IRI(NS, 'stefan'))})
        self.assertEqual(inst, target_inst)

    def test_types(self):
        NS = "http://example.com/father#"
        mgr = OWLOntologyManager_Owlready2()
        onto = mgr.load_ontology(IRI.create("file://KGs/father.owl"))
        reasoner = OWLReasoner_Owlready2(onto)

        types = frozenset(reasoner.types(OWLNamedIndividual(IRI.create(NS, 'stefan'))))
        target_types = frozenset({OWLThing, OWLClass(IRI(NS, 'male')), OWLClass(IRI(NS, 'person'))})
        self.assertEqual(target_types, types)

    def test_object_values(self):
        NS = "http://example.com/father#"
        mgr = OWLOntologyManager_Owlready2()
        onto = mgr.load_ontology(IRI.create("file://KGs/father.owl"))
        reasoner = OWLReasoner_Owlready2(onto)

        stefan = OWLNamedIndividual(IRI.create(NS, 'stefan'))
        has_child = OWLObjectProperty(IRI.create(NS, 'hasChild'))

        kids = frozenset(reasoner.object_property_values(stefan, has_child))
        target_kids = frozenset({OWLNamedIndividual(IRI(NS, 'markus'))})
        self.assertEqual(target_kids, kids)

        heinz = OWLNamedIndividual(IRI(NS, 'heinz'))
        no_kids = frozenset(reasoner.object_property_values(heinz, has_child))
        self.assertEqual(frozenset(), no_kids)

    def test_mapping(self):
        NS = "http://example.com/father#"
        mgr = OWLOntologyManager_Owlready2()
        onto = mgr.load_ontology(IRI.create("file://KGs/father.owl"))

        male = OWLClass(IRI.create(NS, 'male'))
        female = OWLClass(IRI.create(NS, 'female'))
        has_child = OWLObjectProperty(IRI(NS, 'hasChild'))

        to_owlready = owlapy.owlready2.utils.ToOwlready2(world=onto._world)

        ce = male
        owlready_ce = onto._onto.male
        self.assertEqual(owlready_ce, to_owlready.map_concept(ce))

        ce = OWLObjectUnionOf((male, female))
        owlready_ce = onto._onto.male | onto._onto.female
        self.assertEqual(owlready_ce, to_owlready.map_concept(ce))

        ce = OWLObjectIntersectionOf((male, female))
        owlready_ce = onto._onto.male & onto._onto.female
        self.assertEqual(owlready_ce, to_owlready.map_concept(ce))

        ce = OWLObjectSomeValuesFrom(has_child, OWLThing)
        import owlready2
        owlready_ce = onto._onto.hasChild.some(owlready2.owl.Thing)
        self.assertEqual(owlready_ce, to_owlready.map_concept(ce))

        ce = OWLObjectSomeValuesFrom(has_child, male)
        owlready_ce = onto._onto.hasChild.some(onto._onto.male)
        self.assertEqual(owlready_ce, to_owlready.map_concept(ce))

        ce = male.get_object_complement_of()
        owlready_ce = owlready2.Not(onto._onto.male)
        self.assertEqual(owlready_ce, to_owlready.map_concept(ce))

    def test_mapping_rev(self):
        NS = "http://example.com/father#"
        mgr = OWLOntologyManager_Owlready2()
        onto = mgr.load_ontology(IRI.create("file://KGs/father.owl"))

        male = onto._onto.male
        female = onto._onto.female
        has_child = onto._onto.hasChild

        from_owlready = owlapy.owlready2.utils.FromOwlready2()

        import owlready2
        ce = male | female
        owl_ce = OWLObjectUnionOf((OWLClass(IRI.create(NS, 'male')), OWLClass(IRI.create(NS, 'female'))))
        self.assertEqual(owl_ce, from_owlready.map_concept(ce))

        ce = has_child.some(owlready2.owl.Thing)
        owl_ce = OWLObjectSomeValuesFrom(OWLObjectProperty(IRI(NS, 'hasChild')), OWLThing)
        self.assertEqual(owl_ce, from_owlready.map_concept(ce))

        ce = has_child.some(male)
        owl_ce = OWLObjectSomeValuesFrom(OWLObjectProperty(IRI(NS, 'hasChild')), OWLClass(IRI.create(NS, 'male')))
        self.assertEqual(owl_ce, from_owlready.map_concept(ce))

        ce = owlready2.Not(male)
        owl_ce = OWLObjectComplementOf(OWLClass(IRI(NS, 'male')))
        self.assertEqual(owl_ce, from_owlready.map_concept(ce))


class Owlapy_Owlready2_TempClasses_Test(unittest.TestCase):
    # noinspection DuplicatedCode
    def test_instances(self):
        NS = "http://example.com/father#"
        mgr = OWLOntologyManager_Owlready2()
        onto = mgr.load_ontology(IRI.create("file://KGs/father.owl"))

        male = OWLClass(IRI.create(NS, 'male'))
        female = OWLClass(IRI.create(NS, 'female'))
        has_child = OWLObjectProperty(IRI(NS, 'hasChild'))

        # reasoner = OWLReasoner_Owlready2(onto)
        reasoner = OWLReasoner_Owlready2_TempClasses(onto)

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


if __name__ == '__main__':
    unittest.main()
