from datetime import date, datetime
import unittest

from owlready2.prop import DataProperty
from pandas import Timedelta

from owlapy.fast_instance_checker import OWLReasoner_FastInstanceChecker
from owlapy.model import DurationOWLDatatype, OWLObjectOneOf, OWLObjectProperty, OWLNamedIndividual, \
    OWLObjectSomeValuesFrom, OWLThing, OWLObjectComplementOf, IRI, OWLObjectAllValuesFrom, OWLNothing, \
    OWLObjectHasValue, DoubleOWLDatatype, OWLClass, OWLDataAllValuesFrom, OWLDataComplementOf, \
    OWLDataHasValue, OWLDataIntersectionOf, OWLDataOneOf, OWLDataProperty, OWLDataSomeValuesFrom, \
    OWLDataUnionOf, OWLLiteral, OWLObjectExactCardinality, OWLObjectMaxCardinality, OWLObjectMinCardinality, \
    OWLObjectIntersectionOf
from owlapy.model.providers import OWLDatatypeMinExclusiveRestriction, OWLDatatypeMinMaxInclusiveRestriction, \
    OWLDatatypeMinMaxExclusiveRestriction, OWLDatatypeMaxExclusiveRestriction, OWLDatatypeMaxInclusiveRestriction
from owlapy.owlready2 import OWLOntologyManager_Owlready2, OWLReasoner_Owlready2
from pytest import mark


class Owlapy_FastInstanceChecker_Test(unittest.TestCase):
    # noinspection DuplicatedCode
    def test_instances(self):
        NS = "http://example.com/father#"
        mgr = OWLOntologyManager_Owlready2()
        onto = mgr.load_ontology(IRI.create("file://KGs/father.owl"))

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

        inst = frozenset(reasoner.instances(OWLObjectHasValue(property=has_child,
                                                              individual=OWLNamedIndividual(IRI(NS, 'heinz')))))
        target_inst = frozenset({OWLNamedIndividual(IRI(NS, 'anna')),
                                 OWLNamedIndividual(IRI(NS, 'martin'))})
        self.assertEqual(inst, target_inst)

        inst = frozenset(reasoner.instances(OWLObjectOneOf((OWLNamedIndividual(IRI(NS, 'anna')),
                                                            OWLNamedIndividual(IRI(NS, 'michelle')),
                                                            OWLNamedIndividual(IRI(NS, 'markus'))))))
        target_inst = frozenset({OWLNamedIndividual(IRI(NS, 'anna')),
                                 OWLNamedIndividual(IRI(NS, 'michelle')),
                                 OWLNamedIndividual(IRI(NS, 'markus'))})
        self.assertEqual(inst, target_inst)

    def test_complement(self):
        NS = "http://example.com/father#"
        mgr = OWLOntologyManager_Owlready2()
        onto = mgr.load_ontology(IRI.create("file://KGs/father.owl"))

        male = OWLClass(IRI.create(NS, 'male'))
        female = OWLClass(IRI.create(NS, 'female'))
        has_child = OWLObjectProperty(IRI(NS, 'hasChild'))

        base_reasoner = OWLReasoner_Owlready2(onto)
        reasoner_nd = OWLReasoner_FastInstanceChecker(onto, base_reasoner=base_reasoner, negation_default=True)
        reasoner_open = OWLReasoner_FastInstanceChecker(onto, base_reasoner=base_reasoner, negation_default=False)

        self.assertEqual(set(reasoner_nd.instances(male)), set(reasoner_nd.instances(OWLObjectComplementOf(female))))
        self.assertEqual(set(reasoner_nd.instances(female)), set(reasoner_nd.instances(OWLObjectComplementOf(male))))

        self.assertEqual(set(), set(reasoner_open.instances(
            OWLObjectComplementOf(
                OWLObjectSomeValuesFrom(property=has_child, filler=OWLThing)))))

        all_inds = set(onto.individuals_in_signature())
        unknown_child = set(reasoner_nd.instances(
            OWLObjectComplementOf(
                OWLObjectSomeValuesFrom(property=has_child, filler=OWLThing))))
        with_child = set(reasoner_open.instances(
            OWLObjectSomeValuesFrom(property=has_child, filler=OWLThing)))
        self.assertEqual(all_inds - unknown_child, with_child)

    def test_all_values(self):
        NS = "http://example.com/father#"
        mgr = OWLOntologyManager_Owlready2()
        onto = mgr.load_ontology(IRI.create("file://KGs/father.owl"))

        male = OWLClass(IRI.create(NS, 'male'))
        female = OWLClass(IRI.create(NS, 'female'))
        has_child = OWLObjectProperty(IRI(NS, 'hasChild'))

        base_reasoner = OWLReasoner_Owlready2(onto)
        reasoner_nd = OWLReasoner_FastInstanceChecker(onto, base_reasoner=base_reasoner, negation_default=True)

        # note, these answers are all wrong under OWA
        only_male_child = frozenset(reasoner_nd.instances(OWLObjectAllValuesFrom(property=has_child, filler=male)))
        only_female_child = frozenset(reasoner_nd.instances(OWLObjectAllValuesFrom(property=has_child, filler=female)))
        no_child = frozenset(reasoner_nd.instances(OWLObjectAllValuesFrom(property=has_child, filler=OWLNothing)))
        target_inst = frozenset({OWLNamedIndividual(IRI('http://example.com/father#', 'michelle')),
                                 OWLNamedIndividual(IRI('http://example.com/father#', 'heinz'))})
        self.assertEqual(no_child, target_inst)
        print(no_child)

    @mark.xfail
    def test_complement2(self):
        NS = "http://example.com/father#"
        mgr = OWLOntologyManager_Owlready2()
        onto = mgr.load_ontology(IRI.create("file://KGs/father.owl"))

        male = OWLClass(IRI.create(NS, 'male'))
        female = OWLClass(IRI.create(NS, 'female'))
        has_child = OWLObjectProperty(IRI(NS, 'hasChild'))

        base_reasoner = OWLReasoner_Owlready2(onto)
        reasoner_open = OWLReasoner_FastInstanceChecker(onto, base_reasoner=base_reasoner, negation_default=False)

        self.assertEqual(set(reasoner_open.instances(male)),
                         set(reasoner_open.instances(OWLObjectComplementOf(female))))
        self.assertEqual(set(reasoner_open.instances(female)),
                         set(reasoner_open.instances(OWLObjectComplementOf(male))))

    def test_cardinality_restrictions(self):
        NS = "http://dl-learner.org/mutagenesis#"
        mgr = OWLOntologyManager_Owlready2()
        onto = mgr.load_ontology(IRI.create("file://KGs/Mutagenesis/mutagenesis.owl"))

        hydrogen_3 = OWLClass(IRI.create(NS, 'Hydrogen-3'))
        atom = OWLClass(IRI.create(NS, 'Atom'))
        has_atom = OWLObjectProperty(IRI(NS, 'hasAtom'))

        base_reasoner = OWLReasoner_Owlready2(onto)
        reasoner = OWLReasoner_FastInstanceChecker(onto, base_reasoner=base_reasoner, negation_default=True)

        inst = frozenset(reasoner.instances(OWLObjectExactCardinality(cardinality=2,
                                                                      property=has_atom,
                                                                      filler=hydrogen_3)))
        target_inst = frozenset({OWLNamedIndividual(IRI(NS, 'd160')),
                                 OWLNamedIndividual(IRI(NS, 'd195')),
                                 OWLNamedIndividual(IRI(NS, 'd175'))})
        self.assertEqual(inst, target_inst)

        inst = frozenset(reasoner.instances(OWLObjectMinCardinality(cardinality=40,
                                                                    property=has_atom,
                                                                    filler=atom)))
        target_inst_min = frozenset({OWLNamedIndividual(IRI(NS, 'd52')),
                                     OWLNamedIndividual(IRI(NS, 'd91')),
                                     OWLNamedIndividual(IRI(NS, 'd71')),
                                     OWLNamedIndividual(IRI(NS, 'd51'))})
        self.assertEqual(inst, target_inst_min)

        all_inds = set(onto.individuals_in_signature())
        inst = frozenset(reasoner.instances(OWLObjectMaxCardinality(cardinality=39,
                                                                    property=has_atom,
                                                                    filler=atom)))
        self.assertEqual(all_inds - target_inst_min, inst)

    def test_data_properties(self):
        NS = "http://dl-learner.org/mutagenesis#"
        mgr = OWLOntologyManager_Owlready2()
        onto = mgr.load_ontology(IRI.create("file://KGs/Mutagenesis/mutagenesis.owl"))

        act = OWLDataProperty(IRI(NS, 'act'))
        fused_rings = OWLDataProperty(IRI(NS, 'hasThreeOrMoreFusedRings'))
        lumo = OWLDataProperty(IRI(NS, 'lumo'))
        logp = OWLDataProperty(IRI(NS, 'logp'))
        charge = OWLDataProperty(IRI(NS, 'charge'))

        base_reasoner = OWLReasoner_Owlready2(onto)
        reasoner = OWLReasoner_FastInstanceChecker(onto, base_reasoner=base_reasoner, negation_default=True)

        self.assertEqual([], list(reasoner.sub_data_properties(act, direct=True)))

        # OWLDataHasValue
        inst = frozenset(reasoner.instances(
            OWLObjectIntersectionOf((OWLDataHasValue(property=fused_rings, value=OWLLiteral(True)),
                                     OWLDataHasValue(property=act, value=OWLLiteral(2.11))))))
        target_inst = frozenset({OWLNamedIndividual(IRI(NS, 'd1'))})
        self.assertEqual(inst, target_inst)

        # OWLDatatypeRestriction
        restriction = OWLDatatypeMinMaxInclusiveRestriction(-3.0, -2.8)
        inst = frozenset(reasoner.instances(OWLDataSomeValuesFrom(property=lumo, filler=restriction)))
        target_inst = frozenset({OWLNamedIndividual(IRI(NS, 'd149')),
                                 OWLNamedIndividual(IRI(NS, 'd29')),
                                 OWLNamedIndividual(IRI(NS, 'd49')),
                                 OWLNamedIndividual(IRI(NS, 'd96'))})
        self.assertEqual(inst, target_inst)

        # OWLDataAllValuesFrom
        inst2 = frozenset(reasoner.instances(
            OWLObjectComplementOf(OWLDataAllValuesFrom(property=lumo, filler=OWLDataComplementOf(restriction)))))
        self.assertEqual(inst, inst2)

        # OWLDataComplementOf
        restriction = OWLDatatypeMinMaxExclusiveRestriction(-2.0, 0.88)
        inst = frozenset(reasoner.instances(OWLDataSomeValuesFrom(property=charge,
                                                                  filler=OWLDataComplementOf(restriction))))
        target_inst = frozenset({OWLNamedIndividual(IRI(NS, 'd195_12')),
                                 OWLNamedIndividual(IRI(NS, 'd33_27'))})
        self.assertEqual(inst, target_inst)

        # OWLDataOneOf, OWLDatatype, OWLDataIntersectionOf
        inst = frozenset(reasoner.instances(
            OWLDataSomeValuesFrom(property=logp,
                                  filler=OWLDataIntersectionOf((
                                      OWLDataOneOf((OWLLiteral(6.26), OWLLiteral(6.07))),
                                      DoubleOWLDatatype
                                  )))))
        target_inst = frozenset({OWLNamedIndividual(IRI(NS, 'd101')),
                                 OWLNamedIndividual(IRI(NS, 'd109')),
                                 OWLNamedIndividual(IRI(NS, 'd104')),
                                 OWLNamedIndividual(IRI(NS, 'd180'))})
        self.assertEqual(inst, target_inst)

        # OWLDataUnionOf
        restriction = OWLDatatypeMinMaxExclusiveRestriction(5.07, 5.3)
        inst = frozenset(reasoner.instances(
            OWLDataSomeValuesFrom(property=logp,
                                  filler=OWLDataUnionOf((
                                      OWLDataOneOf((OWLLiteral(6.26), OWLLiteral(6.07))), restriction)))))
        target_inst = frozenset({OWLNamedIndividual(IRI(NS, 'd101')),
                                 OWLNamedIndividual(IRI(NS, 'd109')),
                                 OWLNamedIndividual(IRI(NS, 'd92')),
                                 OWLNamedIndividual(IRI(NS, 'd22')),
                                 OWLNamedIndividual(IRI(NS, 'd104')),
                                 OWLNamedIndividual(IRI(NS, 'd180'))})
        self.assertEqual(inst, target_inst)

    def test_data_properties_time(self):
        NS = "http://example.com/father#"
        mgr = OWLOntologyManager_Owlready2()
        onto = mgr.load_ontology(IRI.create("file://KGs/father.owl"))

        with onto._onto:
            class birthDate(DataProperty):
                range = [date]

            class birthDateTime(DataProperty):
                range = [datetime]

            class age(DataProperty):
                range = [Timedelta]

        onto._onto.markus.birthDate = [date(year=1990, month=10, day=2)]
        onto._onto.markus.birthDateTime = [datetime(year=1990, month=10, day=2, hour=10, minute=20, second=5)]
        onto._onto.markus.age = [Timedelta(days=11315, hours=10, minutes=2)]

        onto._onto.anna.birthDate = [date(year=1995, month=6, day=10)]
        onto._onto.anna.birthDateTime = [datetime(year=1995, month=6, day=10, hour=2, minute=10)]
        onto._onto.anna.age = [Timedelta(days=9490, hours=4)]

        onto._onto.heinz.birthDate = [date(year=1986, month=6, day=10)]
        onto._onto.heinz.birthDateTime = [datetime(year=1986, month=6, day=10, hour=10, second=10)]
        onto._onto.heinz.age = [Timedelta(days=12775, hours=14, seconds=40)]

        onto._onto.michelle.birthDate = [date(year=2000, month=1, day=4)]
        onto._onto.michelle.birthDateTime = [datetime(year=2000, month=1, day=4, minute=4, second=10)]
        onto._onto.michelle.age = [Timedelta(days=7665, hours=1, milliseconds=11)]

        onto._onto.martin.birthDate = [date(year=1999, month=3, day=1)]
        onto._onto.martin.birthDateTime = [datetime(year=1999, month=3, day=2, hour=20, minute=2, second=30)]
        onto._onto.martin.age = [Timedelta(days=8030, minutes=2)]

        birth_date = OWLDataProperty(IRI(NS, 'birthDate'))
        birth_date_time = OWLDataProperty(IRI(NS, 'birthDateTime'))
        age_ = OWLDataProperty(IRI(NS, 'age'))
        markus = OWLNamedIndividual(IRI(NS, 'markus'))
        anna = OWLNamedIndividual(IRI(NS, 'anna'))
        heinz = OWLNamedIndividual(IRI(NS, 'heinz'))
        michelle = OWLNamedIndividual(IRI(NS, 'michelle'))
        martin = OWLNamedIndividual(IRI(NS, 'martin'))

        base_reasoner = OWLReasoner_Owlready2(onto)
        reasoner = OWLReasoner_FastInstanceChecker(onto, base_reasoner=base_reasoner)

        restriction = OWLDatatypeMinMaxExclusiveRestriction(date(year=1995, month=6, day=12),
                                                            date(year=1999, month=3, day=2))
        inst = frozenset(reasoner.instances(OWLDataSomeValuesFrom(property=birth_date,
                                                                  filler=restriction)))
        target_inst = frozenset({martin})
        self.assertEqual(inst, target_inst)

        inst = frozenset(reasoner.instances(OWLDataSomeValuesFrom(property=birth_date,
                                                                  filler=OWLDataComplementOf(restriction))))
        target_inst = frozenset({michelle, anna, heinz, markus})
        self.assertEqual(inst, target_inst)

        restriction = OWLDatatypeMaxInclusiveRestriction(datetime(year=1990, month=10, day=2, hour=10,
                                                                  minute=20, second=5))
        inst = frozenset(reasoner.instances(OWLDataSomeValuesFrom(property=birth_date_time,
                                                                  filler=restriction)))
        target_inst = frozenset({markus, heinz})
        self.assertEqual(inst, target_inst)

        restriction_min = OWLDatatypeMinExclusiveRestriction(Timedelta(days=8030, minutes=1))
        restriction_max = OWLDatatypeMaxExclusiveRestriction(Timedelta(days=9490, hours=4, nanoseconds=1))
        filler = OWLDataIntersectionOf([restriction_min, restriction_max, DurationOWLDatatype])
        inst = frozenset(reasoner.instances(OWLDataSomeValuesFrom(property=age_, filler=filler)))
        target_inst = frozenset({anna, martin})
        self.assertEqual(inst, target_inst)


if __name__ == '__main__':
    unittest.main()
