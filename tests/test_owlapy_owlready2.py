from datetime import date, datetime
import unittest

from pandas import Timedelta
from owlapy.model.providers import OWLDatatypeMaxInclusiveRestriction, OWLDatatypeMinInclusiveRestriction, \
    OWLDatatypeMinMaxExclusiveRestriction, OWLDatatypeMinMaxInclusiveRestriction

import owlready2
import owlapy.owlready2.utils
from owlapy.model import OWLObjectProperty, OWLNamedIndividual, OWLThing, OWLClass, OWLObjectUnionOf, \
    OWLObjectIntersectionOf, OWLObjectSomeValuesFrom, OWLObjectComplementOf, IRI, OWLDataAllValuesFrom, \
    OWLDataComplementOf, OWLDataHasValue, OWLDataIntersectionOf, OWLDataProperty, OWLDataSomeValuesFrom, \
    OWLDataUnionOf, OWLLiteral, BooleanOWLDatatype, DoubleOWLDatatype, IntegerOWLDatatype, OWLDataOneOf, \
    OWLDataExactCardinality, OWLDataMaxCardinality, OWLDataMinCardinality, OWLObjectExactCardinality, \
    OWLObjectMaxCardinality, OWLObjectMinCardinality, OWLObjectHasValue, OWLObjectAllValuesFrom, \
    OWLObjectOneOf, DateOWLDatatype, DateTimeOWLDatatype, DurationOWLDatatype

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
        owlready_ce = onto._onto.hasChild.some(owlready2.owl.Thing)
        self.assertEqual(owlready_ce, to_owlready.map_concept(ce))

        ce = OWLObjectAllValuesFrom(has_child, male)
        owlready_ce = onto._onto.hasChild.only(onto._onto.male)
        self.assertEqual(owlready_ce, to_owlready.map_concept(ce))

        ce = male.get_object_complement_of()
        owlready_ce = owlready2.Not(onto._onto.male)
        self.assertEqual(owlready_ce, to_owlready.map_concept(ce))

        ce = OWLObjectOneOf([OWLNamedIndividual(IRI.create(NS, 'martin')),
                             OWLNamedIndividual(IRI.create(NS, 'michelle'))])
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

        ce = OWLObjectHasValue(has_child, OWLNamedIndividual(IRI.create(NS, 'markus')))
        owlready_ce = onto._onto.hasChild.value(onto._onto.markus)
        self.assertEqual(owlready_ce, to_owlready.map_concept(ce))

    def test_mapping_data_properties(self):
        NS = "http://dl-learner.org/mutagenesis#"
        mgr = OWLOntologyManager_Owlready2()
        onto = mgr.load_ontology(IRI.create("file://KGs/Mutagenesis/mutagenesis.owl"))

        act = OWLDataProperty(IRI(NS, 'act'))
        charge = OWLDataProperty(IRI(NS, 'charge'))

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
        NS = "http://example.com/father#"
        mgr = OWLOntologyManager_Owlready2()
        onto = mgr.load_ontology(IRI.create("file://KGs/father.owl"))

        male = onto._onto.male
        female = onto._onto.female
        has_child = onto._onto.hasChild

        from_owlready = owlapy.owlready2.utils.FromOwlready2()

        ce = male | female
        owl_ce = OWLObjectUnionOf((OWLClass(IRI.create(NS, 'male')), OWLClass(IRI.create(NS, 'female'))))
        self.assertEqual(owl_ce, from_owlready.map_concept(ce))

        ce = male & female
        owl_ce = OWLObjectIntersectionOf((OWLClass(IRI.create(NS, 'male')), OWLClass(IRI.create(NS, 'female'))))
        self.assertEqual(owl_ce, from_owlready.map_concept(ce))

        ce = has_child.some(owlready2.owl.Thing)
        owl_ce = OWLObjectSomeValuesFrom(OWLObjectProperty(IRI(NS, 'hasChild')), OWLThing)
        self.assertEqual(owl_ce, from_owlready.map_concept(ce))

        ce = has_child.only(male)
        owl_ce = OWLObjectAllValuesFrom(OWLObjectProperty(IRI(NS, 'hasChild')), OWLClass(IRI.create(NS, 'male')))
        self.assertEqual(owl_ce, from_owlready.map_concept(ce))

        ce = owlready2.Not(male)
        owl_ce = OWLObjectComplementOf(OWLClass(IRI(NS, 'male')))
        self.assertEqual(owl_ce, from_owlready.map_concept(ce))

        ce = owlready2.OneOf([onto._onto.markus, onto._onto.anna])
        owl_ce = OWLObjectOneOf([OWLNamedIndividual(IRI.create(NS, 'markus')),
                                 OWLNamedIndividual(IRI.create(NS, 'anna'))])
        self.assertEqual(owl_ce, from_owlready.map_concept(ce))

        ce = onto._onto.hasChild.min(2, onto._onto.male)
        owl_ce = OWLObjectMinCardinality(2, OWLObjectProperty(IRI(NS, 'hasChild')), OWLClass(IRI(NS, 'male')))
        self.assertEqual(owl_ce, from_owlready.map_concept(ce))

        ce = onto._onto.hasChild.max(5, onto._onto.female)
        owl_ce = OWLObjectMaxCardinality(5, OWLObjectProperty(IRI(NS, 'hasChild')), OWLClass(IRI(NS, 'female')))
        self.assertEqual(owl_ce, from_owlready.map_concept(ce))

        ce = onto._onto.hasChild.exactly(3, owlready2.owl.Thing)
        owl_ce = OWLObjectExactCardinality(3, OWLObjectProperty(IRI(NS, 'hasChild')), OWLThing)
        self.assertEqual(owl_ce, from_owlready.map_concept(ce))

        ce = has_child.value(onto._onto.markus)
        owl_ce = OWLObjectHasValue(OWLObjectProperty(IRI(NS, 'hasChild')), OWLNamedIndividual(IRI.create(NS, 'markus')))
        self.assertEqual(owl_ce, from_owlready.map_concept(ce))

    def test_mapping_rev_data_properties(self):
        NS = "http://dl-learner.org/mutagenesis#"
        mgr = OWLOntologyManager_Owlready2()
        onto = mgr.load_ontology(IRI.create("file://KGs/Mutagenesis/mutagenesis.owl"))

        act = onto._onto.act
        charge = onto._onto.charge

        from_owlready = owlapy.owlready2.utils.FromOwlready2()

        ce = act.some(float)
        owl_ce = OWLDataSomeValuesFrom(OWLDataProperty(IRI(NS, 'act')), DoubleOWLDatatype)
        self.assertEqual(owl_ce, from_owlready.map_concept(ce))

        ce = charge.only(owlready2.Not(owlready2.ConstrainedDatatype(int, max_inclusive=2)))
        res = OWLDatatypeMaxInclusiveRestriction(2)
        owl_ce = OWLDataAllValuesFrom(OWLDataProperty(IRI(NS, 'charge')), OWLDataComplementOf(res))
        self.assertEqual(owl_ce, from_owlready.map_concept(ce))

        ce = charge.some(owlready2.Not(owlready2.ConstrainedDatatype(int, max_inclusive=2)) |
                         owlready2.ConstrainedDatatype(float, min_inclusive=2.1, max_inclusive=2.2))
        res = OWLDatatypeMaxInclusiveRestriction(2)
        res2 = OWLDatatypeMinMaxInclusiveRestriction(2.1, 2.2)
        owl_ce = OWLDataSomeValuesFrom(OWLDataProperty(IRI(NS, 'charge')),
                                       OWLDataUnionOf([OWLDataComplementOf(res), res2]))
        self.assertEqual(owl_ce, from_owlready.map_concept(ce))

        ce = act.only(owlready2.Not(owlready2.ConstrainedDatatype(int, max_inclusive=2)) & datetime)
        owl_ce = OWLDataAllValuesFrom(OWLDataProperty(IRI(NS, 'act')),
                                      OWLDataIntersectionOf([OWLDataComplementOf(res), DateTimeOWLDatatype]))
        self.assertEqual(owl_ce, from_owlready.map_concept(ce))

        ce = act.some(owlready2.Not(owlready2.OneOf([1, 2, 3])) & Timedelta)
        values = OWLDataOneOf([OWLLiteral(1), OWLLiteral(2), OWLLiteral(3)])
        owl_ce = OWLDataSomeValuesFrom(OWLDataProperty(IRI(NS, 'act')),
                                       OWLDataIntersectionOf([OWLDataComplementOf(values), DurationOWLDatatype]))
        self.assertEqual(owl_ce, from_owlready.map_concept(ce))

        ce = onto._onto.act.value(19.5)
        owl_ce = OWLDataHasValue(OWLDataProperty(IRI(NS, 'act')), OWLLiteral(19.5))
        self.assertEqual(owl_ce, from_owlready.map_concept(ce))

        ce = onto._onto.act.min(2, owlready2.ConstrainedDatatype(int, max_inclusive=2))
        owl_ce = OWLDataMinCardinality(2, OWLDataProperty(IRI(NS, 'act')), res)
        self.assertEqual(owl_ce, from_owlready.map_concept(ce))

        ce = onto._onto.charge.max(5, date)
        owl_ce = OWLDataMaxCardinality(5, OWLDataProperty(IRI(NS, 'charge')), DateOWLDatatype)
        self.assertEqual(owl_ce, from_owlready.map_concept(ce))

        ce = onto._onto.charge.exactly(3, owlready2.Not(float))
        owl_ce = OWLDataExactCardinality(3, OWLDataProperty(IRI(NS, 'charge')), OWLDataComplementOf(DoubleOWLDatatype))
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
