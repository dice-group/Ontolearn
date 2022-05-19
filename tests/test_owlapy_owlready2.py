from datetime import date, datetime
import unittest

from pandas import Timedelta
from owlapy.model.providers import OWLDatatypeMaxInclusiveRestriction, OWLDatatypeMinInclusiveRestriction, \
    OWLDatatypeMinMaxExclusiveRestriction, OWLDatatypeMinMaxInclusiveRestriction

import owlready2
import owlapy.owlready2.utils
from owlapy.model import OWLObjectPropertyRangeAxiom, OWLSameIndividualAxiom, OWLSubClassOfAxiom, OWLClass, \
    OWLObjectIntersectionOf, OWLObjectSomeValuesFrom, OWLObjectComplementOf, IRI, OWLDataAllValuesFrom, \
    OWLDataComplementOf, OWLDataHasValue, OWLDataIntersectionOf, OWLDataProperty, OWLDataSomeValuesFrom, \
    OWLDataUnionOf, OWLLiteral, BooleanOWLDatatype, DoubleOWLDatatype, IntegerOWLDatatype, OWLDataOneOf, \
    OWLDataExactCardinality, OWLDataMaxCardinality, OWLDataMinCardinality, OWLObjectExactCardinality, \
    OWLObjectMaxCardinality, OWLObjectMinCardinality, OWLObjectHasValue, OWLObjectAllValuesFrom, \
    OWLObjectOneOf, DateOWLDatatype, DateTimeOWLDatatype, DurationOWLDatatype, OWLClassAssertionAxiom, \
    OWLNamedIndividual, OWLEquivalentClassesAxiom, OWLSubObjectPropertyOfAxiom, OWLThing, \
    OWLDifferentIndividualsAxiom, OWLDisjointClassesAxiom, OWLDisjointDataPropertiesAxiom, OWLObjectUnionOf, \
    OWLDisjointObjectPropertiesAxiom, OWLEquivalentDataPropertiesAxiom, OWLEquivalentObjectPropertiesAxiom, \
    OWLDataPropertyAssertionAxiom, OWLObjectProperty, OWLDataPropertyDomainAxiom, OWLDataPropertyRangeAxiom, \
    OWLObjectPropertyAssertionAxiom, OWLObjectPropertyDomainAxiom

from owlapy.owlready2 import OWLOntologyManager_Owlready2, OWLReasoner_Owlready2
from owlapy.owlready2.temp_classes import OWLReasoner_Owlready2_TempClasses


class Owlapy_Owlready2_Test(unittest.TestCase):
    def test_sub_classes(self):
        ns = "http://example.com/father#"
        mgr = OWLOntologyManager_Owlready2()
        onto = mgr.load_ontology(IRI.create("file://KGs/father.owl"))
        reasoner = OWLReasoner_Owlready2(onto)

        classes = frozenset(reasoner.sub_classes(OWLClass(IRI.create(ns, "person"))))
        target_classes = frozenset((OWLClass(IRI.create(ns, "male")), OWLClass(IRI.create(ns, "female"))))
        self.assertEqual(classes, target_classes)

    def test_sub_object_properties(self):
        ns = "http://www.biopax.org/examples/glycolysis#"
        mgr = OWLOntologyManager_Owlready2()
        onto = mgr.load_ontology(IRI.create("file://KGs/Biopax/biopax.owl"))
        reasoner = OWLReasoner_Owlready2(onto)

        participants = OWLObjectProperty(IRI.create(ns, 'PARTICIPANTS'))
        target_props = frozenset({OWLObjectProperty(IRI(ns, 'COFACTOR')),
                                  OWLObjectProperty(IRI(ns, 'CONTROLLED')),
                                  OWLObjectProperty(IRI(ns, 'CONTROLLER')),
                                  OWLObjectProperty(IRI(ns, 'LEFT')),
                                  OWLObjectProperty(IRI(ns, 'RIGHT'))})
        self.assertEqual(frozenset(reasoner.sub_object_properties(participants, direct=True)), target_props)
        self.assertEqual(frozenset(reasoner.sub_object_properties(participants, direct=False)), target_props)

    def test_instances(self):
        ns = "http://example.com/father#"
        mgr = OWLOntologyManager_Owlready2()
        onto = mgr.load_ontology(IRI.create("file://KGs/father.owl"))
        reasoner = OWLReasoner_Owlready2(onto)

        inst = frozenset(reasoner.instances(OWLThing))
        target_inst = frozenset({OWLNamedIndividual(IRI(ns, 'anna')),
                                 OWLNamedIndividual(IRI(ns, 'heinz')),
                                 OWLNamedIndividual(IRI(ns, 'markus')),
                                 OWLNamedIndividual(IRI(ns, 'martin')),
                                 OWLNamedIndividual(IRI(ns, 'michelle')),
                                 OWLNamedIndividual(IRI(ns, 'stefan'))})
        self.assertEqual(inst, target_inst)

        inst = frozenset(reasoner.instances(OWLClass(IRI.create(ns, "male"))))
        target_inst = frozenset({OWLNamedIndividual(IRI(ns, 'heinz')), OWLNamedIndividual(IRI(ns, 'martin')),
                                 OWLNamedIndividual(IRI(ns, 'markus')), OWLNamedIndividual(IRI(ns, 'stefan'))})
        self.assertEqual(inst, target_inst)

    def test_types(self):
        ns = "http://example.com/father#"
        mgr = OWLOntologyManager_Owlready2()
        onto = mgr.load_ontology(IRI.create("file://KGs/father.owl"))
        reasoner = OWLReasoner_Owlready2(onto)

        types = frozenset(reasoner.types(OWLNamedIndividual(IRI.create(ns, 'stefan'))))
        target_types = frozenset({OWLThing, OWLClass(IRI(ns, 'male')), OWLClass(IRI(ns, 'person'))})
        self.assertEqual(target_types, types)

    def test_object_values(self):
        ns = "http://example.com/father#"
        mgr = OWLOntologyManager_Owlready2()
        onto = mgr.load_ontology(IRI.create("file://KGs/father.owl"))
        reasoner = OWLReasoner_Owlready2(onto)

        stefan = OWLNamedIndividual(IRI.create(ns, 'stefan'))
        has_child = OWLObjectProperty(IRI.create(ns, 'hasChild'))

        kids = frozenset(reasoner.object_property_values(stefan, has_child))
        target_kids = frozenset({OWLNamedIndividual(IRI(ns, 'markus'))})
        self.assertEqual(target_kids, kids)

        heinz = OWLNamedIndividual(IRI(ns, 'heinz'))
        no_kids = frozenset(reasoner.object_property_values(heinz, has_child))
        self.assertEqual(frozenset(), no_kids)

    def test_add_remove_axiom(self):
        ns = "http://example.com/father#"
        mgr = OWLOntologyManager_Owlready2()
        onto = mgr.load_ontology(IRI.create("file://KGs/father.owl"))
        reasoner = OWLReasoner_Owlready2(onto)

        markus = OWLNamedIndividual(IRI.create(ns, 'markus'))
        michelle = OWLNamedIndividual(IRI.create(ns, 'michelle'))
        anna = OWLNamedIndividual(IRI.create(ns, 'anna'))
        marius = OWLNamedIndividual(IRI.create(ns, 'marius'))
        person1 = OWLNamedIndividual(IRI.create(ns, 'person1'))

        has_child = OWLObjectProperty(IRI.create(ns, 'hasChild'))
        has_sibling = OWLObjectProperty(IRI.create(ns, 'has_sibling'))
        age = OWLDataProperty(IRI.create(ns, 'age'))
        test1 = OWLDataProperty(IRI.create(ns, 'test1'))

        male = OWLClass(IRI(ns, 'male'))
        brother = OWLClass(IRI(ns, 'brother'))
        sister = OWLClass(IRI(ns, 'sister'))
        person = OWLClass(IRI(ns, 'person'))
        person_sibling = OWLClass(IRI(ns, 'PersonWithASibling'))

        self.assertNotIn(sister, list(onto.classes_in_signature()))
        mgr.add_axiom(onto, OWLClassAssertionAxiom(anna, sister))
        self.assertIn(sister, list(onto.classes_in_signature()))
        self.assertIn(anna, list(reasoner.instances(sister)))
        self.assertIn(sister, list(reasoner.types(anna)))
        mgr.remove_axiom(onto, OWLClassAssertionAxiom(anna, sister))
        self.assertNotIn(anna, list(reasoner.instances(sister)))
        self.assertNotIn(sister, list(reasoner.types(anna)))

        self.assertNotIn(michelle, list(reasoner.instances(sister)))
        mgr.add_axiom(onto, OWLClassAssertionAxiom(michelle, sister))
        self.assertIn(michelle, list(reasoner.instances(sister)))
        self.assertIn(sister, list(reasoner.types(michelle)))
        mgr.remove_axiom(onto, OWLClassAssertionAxiom(michelle, sister))
        self.assertNotIn(michelle, list(reasoner.instances(sister)))
        self.assertNotIn(sister, list(reasoner.types(michelle)))

        self.assertFalse(list(reasoner.object_property_values(michelle, has_child)))
        mgr.add_axiom(onto, OWLObjectPropertyAssertionAxiom(michelle, has_child, anna))
        self.assertIn(anna, list(reasoner.object_property_values(michelle, has_child)))
        mgr.remove_axiom(onto, OWLObjectPropertyAssertionAxiom(michelle, has_child, anna))
        self.assertNotIn(anna, list(reasoner.object_property_values(michelle, has_child)))
        mgr.remove_axiom(onto, OWLObjectPropertyAssertionAxiom(michelle, has_child, anna))

        self.assertNotIn(has_sibling, list(onto.object_properties_in_signature()))
        self.assertNotIn(marius, list(onto.individuals_in_signature()))
        mgr.add_axiom(onto, OWLObjectPropertyAssertionAxiom(marius, has_sibling, michelle))
        self.assertIn(has_sibling, list(onto.object_properties_in_signature()))
        self.assertIn(marius, list(onto.individuals_in_signature()))
        self.assertIn(michelle, list(reasoner.object_property_values(marius, has_sibling)))
        mgr.remove_axiom(onto, OWLObjectPropertyAssertionAxiom(marius, has_sibling, michelle))
        self.assertNotIn(michelle, list(reasoner.object_property_values(marius, has_sibling)))

        self.assertNotIn(age, list(onto.data_properties_in_signature()))
        mgr.add_axiom(onto, OWLDataPropertyAssertionAxiom(markus, age, OWLLiteral(30)))
        self.assertIn(age, list(onto.data_properties_in_signature()))
        self.assertIn(OWLLiteral(30), list(reasoner.data_property_values(markus, age)))
        mgr.remove_axiom(onto, OWLDataPropertyAssertionAxiom(markus, age, OWLLiteral(30)))
        self.assertNotIn(OWLLiteral(30), list(reasoner.data_property_values(markus, age)))

        self.assertNotIn(OWLLiteral(31), list(reasoner.data_property_values(anna, age)))
        mgr.add_axiom(onto, OWLDataPropertyAssertionAxiom(anna, age, OWLLiteral(31)))
        self.assertIn(OWLLiteral(31), list(reasoner.data_property_values(anna, age)))
        mgr.remove_axiom(onto, OWLDataPropertyAssertionAxiom(anna, age, OWLLiteral(31)))
        self.assertNotIn(OWLLiteral(31), list(reasoner.data_property_values(anna, age)))

        self.assertNotIn(brother, list(onto.classes_in_signature()))
        mgr.add_axiom(onto, OWLSubClassOfAxiom(brother, male))
        self.assertIn(brother, list(reasoner.sub_classes(male)))
        self.assertIn(male, list(reasoner.super_classes(brother)))
        mgr.remove_axiom(onto, OWLSubClassOfAxiom(brother, male))
        self.assertNotIn(brother, list(reasoner.sub_classes(male)))
        self.assertNotIn(male, list(reasoner.super_classes(brother)))

        self.assertNotIn(has_sibling, list(reasoner.sub_object_properties(has_child)))
        mgr.add_axiom(onto, OWLSubObjectPropertyOfAxiom(has_sibling, has_child))
        self.assertIn(has_sibling, list(reasoner.sub_object_properties(has_child)))
        mgr.remove_axiom(onto, OWLSubObjectPropertyOfAxiom(has_sibling, has_child))
        self.assertNotIn(has_sibling, list(reasoner.sub_object_properties(has_child)))

        self.assertNotIn(OWLObjectUnionOf([person, person_sibling]),
                         list(reasoner.object_property_domains(has_sibling)))
        mgr.add_axiom(onto, OWLObjectPropertyDomainAxiom(has_sibling, OWLObjectUnionOf([person, person_sibling])))
        self.assertIn(OWLObjectUnionOf([person, person_sibling]),
                      list(reasoner.object_property_domains(has_sibling, direct=True)))
        mgr.remove_axiom(onto, OWLObjectPropertyDomainAxiom(has_sibling, OWLObjectUnionOf([person, person_sibling])))
        self.assertNotIn(OWLObjectUnionOf([person, person_sibling]),
                         list(reasoner.object_property_domains(has_sibling, direct=True)))

        self.assertNotIn(sister, list(reasoner.object_property_ranges(has_sibling)))
        mgr.add_axiom(onto, OWLObjectPropertyRangeAxiom(has_sibling, sister))
        self.assertIn(sister, list(reasoner.object_property_ranges(has_sibling)))
        mgr.remove_axiom(onto, OWLObjectPropertyRangeAxiom(has_sibling, sister))
        self.assertNotIn(sister, list(reasoner.object_property_ranges(has_sibling)))

        self.assertNotIn(person, list(reasoner.data_property_domains(age)))
        mgr.add_axiom(onto, OWLDataPropertyDomainAxiom(age, person))
        self.assertIn(person, list(reasoner.data_property_domains(age)))
        mgr.remove_axiom(onto, OWLDataPropertyDomainAxiom(age, person))
        self.assertNotIn(person, list(reasoner.data_property_domains(age)))

        self.assertFalse(list(reasoner.data_property_ranges(age)))
        mgr.add_axiom(onto, OWLDataPropertyRangeAxiom(age, OWLDataUnionOf([IntegerOWLDatatype, DateOWLDatatype])))
        self.assertIn(OWLDataUnionOf([IntegerOWLDatatype, DateOWLDatatype]), list(reasoner.data_property_ranges(age)))
        mgr.remove_axiom(onto, OWLDataPropertyRangeAxiom(age, OWLDataUnionOf([IntegerOWLDatatype, DateOWLDatatype])))
        self.assertNotIn(OWLDataUnionOf([IntegerOWLDatatype, DateOWLDatatype]),
                         list(reasoner.data_property_ranges(age)))

        self.assertFalse(list(reasoner.equivalent_classes(brother)))
        mgr.add_axiom(onto, OWLEquivalentClassesAxiom(brother, male))
        self.assertIn(male, list(reasoner.equivalent_classes(brother)))
        mgr.remove_axiom(onto, OWLEquivalentClassesAxiom(brother, male))
        self.assertNotIn(male, list(reasoner.equivalent_classes(brother)))

        self.assertFalse(list(reasoner.equivalent_object_properties(has_child)))
        mgr.add_axiom(onto, OWLEquivalentObjectPropertiesAxiom([has_child, has_sibling]))
        self.assertIn(has_sibling, list(reasoner.equivalent_object_properties(has_child)))
        mgr.remove_axiom(onto, OWLEquivalentObjectPropertiesAxiom([has_child, has_sibling]))
        self.assertNotIn(has_sibling, list(reasoner.equivalent_object_properties(has_child)))

        self.assertFalse(list(reasoner.equivalent_data_properties(age)))
        mgr.add_axiom(onto, OWLEquivalentDataPropertiesAxiom([age, test1]))
        self.assertIn(test1, list(reasoner.equivalent_data_properties(age)))
        mgr.remove_axiom(onto, OWLEquivalentDataPropertiesAxiom([age, test1]))
        self.assertNotIn(test1, list(reasoner.equivalent_data_properties(age)))

        self.assertFalse(list(reasoner.same_individuals(markus)))
        mgr.add_axiom(onto, OWLSameIndividualAxiom([markus, anna, person1]))
        self.assertEqual(set([anna, person1]), set(reasoner.same_individuals(markus)))
        mgr.remove_axiom(onto, OWLSameIndividualAxiom([markus, anna, person1]))
        self.assertFalse(set(reasoner.same_individuals(markus)))

        self.assertFalse(list(reasoner.disjoint_classes(brother)))
        self.assertFalse(list(reasoner.disjoint_classes(sister)))
        mgr.add_axiom(onto, OWLDisjointClassesAxiom([brother, male]))
        mgr.add_axiom(onto, OWLDisjointClassesAxiom([brother, sister, person]))
        self.assertEqual(set([male, sister, person]), set(reasoner.disjoint_classes(brother)))
        self.assertEqual(set([brother, person]), set(reasoner.disjoint_classes(sister)))
        mgr.remove_axiom(onto, OWLDisjointClassesAxiom([brother, male]))
        mgr.remove_axiom(onto, OWLDisjointClassesAxiom([brother, sister, person]))
        self.assertFalse(set(reasoner.disjoint_classes(brother)))
        self.assertFalse(set(reasoner.disjoint_classes(sister)))

        self.assertFalse(list(reasoner.disjoint_object_properties(has_sibling)))
        self.assertFalse(list(reasoner.disjoint_object_properties(has_child)))
        mgr.add_axiom(onto, OWLDisjointObjectPropertiesAxiom([has_child, has_sibling]))
        self.assertIn(has_sibling, set(reasoner.disjoint_object_properties(has_child)))
        self.assertIn(has_child, set(reasoner.disjoint_object_properties(has_sibling)))
        mgr.remove_axiom(onto, OWLDisjointObjectPropertiesAxiom([has_child, has_sibling]))
        self.assertNotIn(has_sibling, set(reasoner.disjoint_object_properties(has_child)))
        self.assertNotIn(has_child, set(reasoner.disjoint_object_properties(has_sibling)))

        self.assertFalse(list(reasoner.disjoint_data_properties(age)))
        self.assertFalse(list(reasoner.disjoint_data_properties(test1)))
        mgr.add_axiom(onto, OWLDisjointDataPropertiesAxiom([age, test1]))
        self.assertIn(test1, set(reasoner.disjoint_data_properties(age)))
        self.assertIn(age, set(reasoner.disjoint_data_properties(test1)))
        mgr.remove_axiom(onto, OWLDisjointDataPropertiesAxiom([age, test1]))
        self.assertNotIn(test1, set(reasoner.disjoint_data_properties(age)))
        self.assertNotIn(age, set(reasoner.disjoint_data_properties(test1)))

        self.assertFalse(list(reasoner.different_individuals(markus)))
        self.assertFalse(list(reasoner.different_individuals(michelle)))
        mgr.add_axiom(onto, OWLDifferentIndividualsAxiom([markus, michelle]))
        mgr.add_axiom(onto, OWLDifferentIndividualsAxiom([markus, anna, marius]))
        self.assertEqual(set([michelle, anna, marius]), set(reasoner.different_individuals(markus)))
        self.assertEqual(set([markus]), set(reasoner.different_individuals(michelle)))
        mgr.remove_axiom(onto, OWLDifferentIndividualsAxiom([markus, michelle]))
        mgr.remove_axiom(onto, OWLDifferentIndividualsAxiom([markus, anna, marius]))
        self.assertFalse(set(reasoner.different_individuals(markus)))
        self.assertFalse(set(reasoner.different_individuals(michelle)))

    def test_mapping(self):
        ns = "http://example.com/father#"
        mgr = OWLOntologyManager_Owlready2()
        onto = mgr.load_ontology(IRI.create("file://KGs/father.owl"))

        male = OWLClass(IRI.create(ns, 'male'))
        female = OWLClass(IRI.create(ns, 'female'))
        has_child = OWLObjectProperty(IRI(ns, 'hasChild'))

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
        owlready_ce = onto._onto.hasChild.some(owlready2.owl.Thing)
        self.assertEqual(owlready_ce, to_owlready.map_concept(ce))

        ce = OWLObjectAllValuesFrom(has_child, male)
        owlready_ce = onto._onto.hasChild.only(onto._onto.male)
        self.assertEqual(owlready_ce, to_owlready.map_concept(ce))

        ce = male.get_object_complement_of()
        owlready_ce = owlready2.Not(onto._onto.male)
        self.assertEqual(owlready_ce, to_owlready.map_concept(ce))

        ce = OWLObjectOneOf([OWLNamedIndividual(IRI.create(ns, 'martin')),
                             OWLNamedIndividual(IRI.create(ns, 'michelle'))])
        owlready_ce = owlready2.OneOf([onto._onto.martin, onto._onto.michelle])
        self.assertEqual(owlready_ce, to_owlready.map_concept(ce))

        ce = OWLObjectMinCardinality(2, has_child, male)
        owlready_ce = onto._onto.hasChild.min(2, onto._onto.male)
        self.assertEqual(owlready_ce, to_owlready.map_concept(ce))

        ce = OWLObjectMaxCardinality(5, has_child, female)
        owlready_ce = onto._onto.hasChild.max(5, onto._onto.female)
        self.assertEqual(owlready_ce, to_owlready.map_concept(ce))

        ce = OWLObjectExactCardinality(3, has_child, OWLThing)
        owlready_ce = onto._onto.hasChild.exactly(3, owlready2.owl.Thing)
        self.assertEqual(owlready_ce, to_owlready.map_concept(ce))

        ce = OWLObjectHasValue(has_child, OWLNamedIndividual(IRI.create(ns, 'markus')))
        owlready_ce = onto._onto.hasChild.value(onto._onto.markus)
        self.assertEqual(owlready_ce, to_owlready.map_concept(ce))

    def test_mapping_data_properties(self):
        ns = "http://dl-learner.org/mutagenesis#"
        mgr = OWLOntologyManager_Owlready2()
        onto = mgr.load_ontology(IRI.create("file://KGs/Mutagenesis/mutagenesis.owl"))

        act = OWLDataProperty(IRI(ns, 'act'))
        charge = OWLDataProperty(IRI(ns, 'charge'))

        to_owlready = owlapy.owlready2.utils.ToOwlready2(world=onto._world)

        # owlready2 defines no equal or hash method for ConstrainedDatatype, just using the __dict__ attribute
        # should be sufficient for the purpose of these tests
        def constraint_datatype_eq(self, other):
            return isinstance(other, owlready2.ConstrainedDatatype) and self.__dict__ == other.__dict__
        setattr(owlready2.ConstrainedDatatype, '__eq__', constraint_datatype_eq)
        setattr(owlready2.ConstrainedDatatype, '__hash__', lambda self: hash(frozenset(self.__dict__.items())))

        ce = OWLDataSomeValuesFrom(act, DateOWLDatatype)
        owlready_ce = onto._onto.act.some(date)
        self.assertEqual(owlready_ce, to_owlready.map_concept(ce))

        res = OWLDatatypeMinInclusiveRestriction(20)
        ce = OWLDataAllValuesFrom(charge, OWLDataComplementOf(res))
        owlready_ce = onto._onto.charge.only(owlready2.Not(owlready2.ConstrainedDatatype(int, min_inclusive=20)))
        self.assertEqual(owlready_ce, to_owlready.map_concept(ce))

        res_both = OWLDatatypeMinMaxExclusiveRestriction(0.5, 1)
        ce = OWLDataAllValuesFrom(charge, OWLDataUnionOf([res, res_both]))
        owlready_ce = onto._onto.charge.only(
            owlready2.Or([owlready2.ConstrainedDatatype(int, min_inclusive=20),
                          owlready2.ConstrainedDatatype(float, min_exclusive=0.5, max_exclusive=1.0)]))
        self.assertEqual(owlready_ce, to_owlready.map_concept(ce))

        res = OWLDatatypeMaxInclusiveRestriction(1.2)
        oneof = OWLDataOneOf([OWLLiteral(2.3), OWLLiteral(5.9), OWLLiteral(7.2)])
        ce = OWLDataAllValuesFrom(charge, OWLDataIntersectionOf([res, oneof]))
        owlready_ce = onto._onto.charge.only(owlready2.ConstrainedDatatype(float, max_inclusive=1.2) &
                                             owlready2.OneOf([2.3, 5.9, 7.2]))
        self.assertEqual(owlready_ce, to_owlready.map_concept(ce))

        ce = OWLDataSomeValuesFrom(charge, OWLDataIntersectionOf([res, BooleanOWLDatatype]))
        owlready_ce = onto._onto.charge.some(
            owlready2.And([owlready2.ConstrainedDatatype(float, max_inclusive=1.2), bool]))
        self.assertEqual(owlready_ce, to_owlready.map_concept(ce))

        ce = OWLDataMinCardinality(2, act, res)
        owlready_ce = onto._onto.act.min(2, owlready2.ConstrainedDatatype(float, max_inclusive=1.2))
        self.assertEqual(owlready_ce, to_owlready.map_concept(ce))

        ce = OWLDataMaxCardinality(4, charge, DurationOWLDatatype)
        owlready_ce = onto._onto.charge.max(4, Timedelta)
        self.assertEqual(owlready_ce, to_owlready.map_concept(ce))

        ce = OWLDataExactCardinality(3, charge, OWLDataComplementOf(IntegerOWLDatatype))
        owlready_ce = onto._onto.charge.exactly(3, owlready2.Not(int))
        self.assertEqual(owlready_ce, to_owlready.map_concept(ce))

    def test_mapping_rev(self):
        ns = "http://example.com/father#"
        mgr = OWLOntologyManager_Owlready2()
        onto = mgr.load_ontology(IRI.create("file://KGs/father.owl"))

        male = onto._onto.male
        female = onto._onto.female
        has_child = onto._onto.hasChild

        from_owlready = owlapy.owlready2.utils.FromOwlready2()

        ce = male | female
        owl_ce = OWLObjectUnionOf((OWLClass(IRI.create(ns, 'male')), OWLClass(IRI.create(ns, 'female'))))
        self.assertEqual(owl_ce, from_owlready.map_concept(ce))

        ce = male & female
        owl_ce = OWLObjectIntersectionOf((OWLClass(IRI.create(ns, 'male')), OWLClass(IRI.create(ns, 'female'))))
        self.assertEqual(owl_ce, from_owlready.map_concept(ce))

        ce = has_child.some(owlready2.owl.Thing)
        owl_ce = OWLObjectSomeValuesFrom(OWLObjectProperty(IRI(ns, 'hasChild')), OWLThing)
        self.assertEqual(owl_ce, from_owlready.map_concept(ce))

        ce = has_child.only(male)
        owl_ce = OWLObjectAllValuesFrom(OWLObjectProperty(IRI(ns, 'hasChild')), OWLClass(IRI.create(ns, 'male')))
        self.assertEqual(owl_ce, from_owlready.map_concept(ce))

        ce = owlready2.Not(male)
        owl_ce = OWLObjectComplementOf(OWLClass(IRI(ns, 'male')))
        self.assertEqual(owl_ce, from_owlready.map_concept(ce))

        ce = owlready2.OneOf([onto._onto.markus, onto._onto.anna])
        owl_ce = OWLObjectOneOf([OWLNamedIndividual(IRI.create(ns, 'markus')),
                                 OWLNamedIndividual(IRI.create(ns, 'anna'))])
        self.assertEqual(owl_ce, from_owlready.map_concept(ce))

        ce = onto._onto.hasChild.min(2, onto._onto.male)
        owl_ce = OWLObjectMinCardinality(2, OWLObjectProperty(IRI(ns, 'hasChild')), OWLClass(IRI(ns, 'male')))
        self.assertEqual(owl_ce, from_owlready.map_concept(ce))

        ce = onto._onto.hasChild.max(5, onto._onto.female)
        owl_ce = OWLObjectMaxCardinality(5, OWLObjectProperty(IRI(ns, 'hasChild')), OWLClass(IRI(ns, 'female')))
        self.assertEqual(owl_ce, from_owlready.map_concept(ce))

        ce = onto._onto.hasChild.exactly(3, owlready2.owl.Thing)
        owl_ce = OWLObjectExactCardinality(3, OWLObjectProperty(IRI(ns, 'hasChild')), OWLThing)
        self.assertEqual(owl_ce, from_owlready.map_concept(ce))

        ce = has_child.value(onto._onto.markus)
        owl_ce = OWLObjectHasValue(OWLObjectProperty(IRI(ns, 'hasChild')), OWLNamedIndividual(IRI.create(ns, 'markus')))
        self.assertEqual(owl_ce, from_owlready.map_concept(ce))

    def test_mapping_rev_data_properties(self):
        ns = "http://dl-learner.org/mutagenesis#"
        mgr = OWLOntologyManager_Owlready2()
        onto = mgr.load_ontology(IRI.create("file://KGs/Mutagenesis/mutagenesis.owl"))

        act = onto._onto.act
        charge = onto._onto.charge

        from_owlready = owlapy.owlready2.utils.FromOwlready2()

        ce = act.some(float)
        owl_ce = OWLDataSomeValuesFrom(OWLDataProperty(IRI(ns, 'act')), DoubleOWLDatatype)
        self.assertEqual(owl_ce, from_owlready.map_concept(ce))

        ce = charge.only(owlready2.Not(owlready2.ConstrainedDatatype(int, max_inclusive=2)))
        res = OWLDatatypeMaxInclusiveRestriction(2)
        owl_ce = OWLDataAllValuesFrom(OWLDataProperty(IRI(ns, 'charge')), OWLDataComplementOf(res))
        self.assertEqual(owl_ce, from_owlready.map_concept(ce))

        ce = charge.some(owlready2.Not(owlready2.ConstrainedDatatype(int, max_inclusive=2)) |
                         owlready2.ConstrainedDatatype(float, min_inclusive=2.1, max_inclusive=2.2))
        res = OWLDatatypeMaxInclusiveRestriction(2)
        res2 = OWLDatatypeMinMaxInclusiveRestriction(2.1, 2.2)
        owl_ce = OWLDataSomeValuesFrom(OWLDataProperty(IRI(ns, 'charge')),
                                       OWLDataUnionOf([OWLDataComplementOf(res), res2]))
        self.assertEqual(owl_ce, from_owlready.map_concept(ce))

        ce = act.only(owlready2.Not(owlready2.ConstrainedDatatype(int, max_inclusive=2)) & datetime)
        owl_ce = OWLDataAllValuesFrom(OWLDataProperty(IRI(ns, 'act')),
                                      OWLDataIntersectionOf([OWLDataComplementOf(res), DateTimeOWLDatatype]))
        self.assertEqual(owl_ce, from_owlready.map_concept(ce))

        ce = act.some(owlready2.Not(owlready2.OneOf([1, 2, 3])) & Timedelta)
        values = OWLDataOneOf([OWLLiteral(1), OWLLiteral(2), OWLLiteral(3)])
        owl_ce = OWLDataSomeValuesFrom(OWLDataProperty(IRI(ns, 'act')),
                                       OWLDataIntersectionOf([OWLDataComplementOf(values), DurationOWLDatatype]))
        self.assertEqual(owl_ce, from_owlready.map_concept(ce))

        ce = onto._onto.act.value(19.5)
        owl_ce = OWLDataHasValue(OWLDataProperty(IRI(ns, 'act')), OWLLiteral(19.5))
        self.assertEqual(owl_ce, from_owlready.map_concept(ce))

        ce = onto._onto.act.min(2, owlready2.ConstrainedDatatype(int, max_inclusive=2))
        owl_ce = OWLDataMinCardinality(2, OWLDataProperty(IRI(ns, 'act')), res)
        self.assertEqual(owl_ce, from_owlready.map_concept(ce))

        ce = onto._onto.charge.max(5, date)
        owl_ce = OWLDataMaxCardinality(5, OWLDataProperty(IRI(ns, 'charge')), DateOWLDatatype)
        self.assertEqual(owl_ce, from_owlready.map_concept(ce))

        ce = onto._onto.charge.exactly(3, owlready2.Not(float))
        owl_ce = OWLDataExactCardinality(3, OWLDataProperty(IRI(ns, 'charge')), OWLDataComplementOf(DoubleOWLDatatype))
        self.assertEqual(owl_ce, from_owlready.map_concept(ce))


class Owlapy_Owlready2_TempClasses_Test(unittest.TestCase):
    # noinspection DuplicatedCode
    def test_instances(self):
        ns = "http://example.com/father#"
        mgr = OWLOntologyManager_Owlready2()
        onto = mgr.load_ontology(IRI.create("file://KGs/father.owl"))

        male = OWLClass(IRI.create(ns, 'male'))
        female = OWLClass(IRI.create(ns, 'female'))
        has_child = OWLObjectProperty(IRI(ns, 'hasChild'))

        # reasoner = OWLReasoner_Owlready2(onto)
        reasoner = OWLReasoner_Owlready2_TempClasses(onto)

        inst = frozenset(reasoner.instances(female))
        target_inst = frozenset({OWLNamedIndividual(IRI(ns, 'anna')),
                                 OWLNamedIndividual(IRI(ns, 'michelle'))})
        self.assertEqual(inst, target_inst)

        inst = frozenset(reasoner.instances(
            OWLObjectIntersectionOf((male, OWLObjectSomeValuesFrom(property=has_child, filler=female)))))
        target_inst = frozenset({OWLNamedIndividual(IRI(ns, 'markus'))})
        self.assertEqual(inst, target_inst)

        inst = frozenset(reasoner.instances(
            OWLObjectIntersectionOf((female, OWLObjectSomeValuesFrom(property=has_child, filler=OWLThing)))))
        target_inst = frozenset({OWLNamedIndividual(IRI(ns, 'anna'))})
        self.assertEqual(inst, target_inst)


if __name__ == '__main__':
    unittest.main()
