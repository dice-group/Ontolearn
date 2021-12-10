import unittest
from itertools import repeat
from ontolearn.concept_generator import ConceptGenerator

from owlapy.fast_instance_checker import OWLReasoner_FastInstanceChecker
from owlapy.model import OWLObjectProperty, IRI, OWLObjectSomeValuesFrom, OWLObjectUnionOf, OWLThing, \
    BooleanOWLDatatype, DoubleOWLDatatype, IntegerOWLDatatype, OWLClass, OWLDataAllValuesFrom, \
    OWLDataHasValue, OWLDataProperty, OWLDataSomeValuesFrom, OWLLiteral, OWLNamedIndividual, \
    OWLNothing, OWLObjectAllValuesFrom, OWLObjectComplementOf, OWLObjectExactCardinality, \
    OWLObjectHasValue, OWLObjectIntersectionOf, OWLObjectInverseOf, OWLObjectMaxCardinality, \
    OWLObjectMinCardinality

from owlapy.owlready2 import OWLOntologyManager_Owlready2, OWLReasoner_Owlready2


class ConceptGeneratorTest(unittest.TestCase):

    def setUp(self):
        self.namespace = "http://dl-learner.org/mutagenesis#"
        mgr = OWLOntologyManager_Owlready2()
        onto = mgr.load_ontology(IRI.create("file://KGs/Mutagenesis/mutagenesis.owl"))

        base_reasoner = OWLReasoner_Owlready2(onto)
        reasoner = OWLReasoner_FastInstanceChecker(onto, base_reasoner=base_reasoner, negation_default=True)
        self.generator = ConceptGenerator(reasoner)

        # Classes
        self.atom = OWLClass(IRI.create(self.namespace, 'Atom'))
        self.bond = OWLClass(IRI.create(self.namespace, 'Bond'))
        self.compound = OWLClass(IRI(self.namespace, 'Compound'))
        self.ring_structure = OWLClass(IRI(self.namespace, 'RingStructure'))
        self.bond1 = OWLClass(IRI.create(self.namespace, 'Bond-1'))
        self.bond2 = OWLClass(IRI.create(self.namespace, 'Bond-2'))
        self.bond3 = OWLClass(IRI.create(self.namespace, 'Bond-3'))
        self.bond4 = OWLClass(IRI.create(self.namespace, 'Bond-4'))
        self.bond5 = OWLClass(IRI.create(self.namespace, 'Bond-5'))
        self.bond7 = OWLClass(IRI.create(self.namespace, 'Bond-7'))

        # Object Properties
        self.in_bond = OWLObjectProperty(IRI.create(self.namespace, 'inBond'))
        self.has_bond = OWLObjectProperty(IRI.create(self.namespace, 'hasBond'))
        self.has_atom = OWLObjectProperty(IRI.create(self.namespace, 'hasAtom'))
        self.in_structure = OWLObjectProperty(IRI.create(self.namespace, 'inStructure'))
        self.has_structure = OWLObjectProperty(IRI.create(self.namespace, 'hasStructure'))
        self.object_properties = {self.in_bond, self.has_bond, self.has_atom, self.in_structure, self.has_structure}

        # Data Properties
        self.charge = OWLDataProperty(IRI.create(self.namespace, 'charge'))
        self.act = OWLDataProperty(IRI.create(self.namespace, 'act'))
        self.logp = OWLDataProperty(IRI.create(self.namespace, 'logp'))
        self.lumo = OWLDataProperty(IRI.create(self.namespace, 'lumo'))
        self.has_fife_examples = OWLDataProperty(IRI.create(self.namespace, 'hasFifeExamplesOfAcenthrylenes'))
        self.has_three = OWLDataProperty(IRI.create(self.namespace, 'hasThreeOrMoreFusedRings'))
        self.data_properties = {self.charge, self.act, self.logp, self.lumo, self.has_fife_examples,
                                self.has_three}
        self.boolean_data_properties = {self.has_fife_examples, self.has_three}
        self.numeric_data_properties = {self.charge, self.act, self.logp, self.lumo}

        # Individuals
        self.bond5225 = OWLNamedIndividual(IRI.create(self.namespace, 'bond5225'))
        self.d91_17 = OWLNamedIndividual(IRI.create(self.namespace, 'd91_17'))
        self.d91_32 = OWLNamedIndividual(IRI.create(self.namespace, 'd91_32'))

    def test_classes_retrieval(self):
        # get concepts
        self.assertEqual(86, len(list(self.generator.get_concepts())))

        # direct sub concepts
        classes = set(self.generator.get_direct_sub_concepts(OWLThing))
        true_classes = {self.atom, self.bond, self.compound, self.ring_structure}
        self.assertEqual(true_classes, classes)

        # all sub concepts
        classes = set(self.generator.get_all_sub_concepts(self.bond))
        true_classes = {self.bond1, self.bond2, self.bond3, self.bond4, self.bond5, self.bond7}
        self.assertEqual(true_classes, classes)

        # all leaf concepts
        classes = set(self.generator.get_leaf_concepts(self.bond))
        self.assertEqual(true_classes, classes)

        # get direct parents
        classes = set(self.generator.get_direct_parents(self.bond1))
        true_classes = {self.bond}
        self.assertEqual(true_classes, classes)

        # types of an individual
        classes = set(self.generator.get_types(self.bond5225, direct=True))
        true_classes = {self.bond1}
        self.assertEqual(true_classes, classes)

        classes = set(self.generator.get_types(self.bond5225))
        true_classes = {self.bond, self.bond1, OWLThing}
        self.assertEqual(true_classes, classes)

    def test_property_retrieval(self):
        self.assertEqual(self.object_properties, set(self.generator.get_object_properties()))
        self.assertEqual(self.data_properties, set(self.generator.get_data_properties()))
        self.assertEqual(self.boolean_data_properties, set(self.generator.get_boolean_data_properties()))
        self.assertEqual(self.numeric_data_properties, set(self.generator.get_numeric_data_properties()))
        self.assertFalse(set(self.generator.get_time_data_properties()))

        # most general data properties
        self.assertEqual(self.boolean_data_properties,
                         set(self.generator.most_general_boolean_data_properties(domain=self.compound)))
        self.assertFalse(set(self.generator.most_general_boolean_data_properties(domain=self.bond)))

        self.assertEqual({self.charge}, set(self.generator.most_general_numeric_data_properties(domain=self.atom)))
        self.assertFalse(set(self.generator.most_general_numeric_data_properties(domain=self.bond)))

        self.data_properties.remove(self.charge)
        self.assertEqual(self.data_properties,
                         set(self.generator.most_general_data_properties(domain=self.compound)))
        self.assertFalse(set(self.generator.most_general_data_properties(domain=self.bond)))

        self.assertFalse(set(self.generator.most_general_time_data_properties(domain=OWLThing)))

        # object property values of an individual
        inds = set(self.generator.get_object_property_values(self.bond5225, self.in_bond))
        true_inds = {self.d91_32, self.d91_17}
        self.assertEqual(true_inds, inds)

        # data property values of an individual
        values = set(self.generator.get_data_property_values(self.d91_32, self.charge))
        true_values = {OWLLiteral(0.146)}
        self.assertEqual(true_values, values)

    def test_ignore(self):
        concepts_to_ignore = {self.bond1, self.compound}
        object_properties_to_ignore = {self.in_bond, self.has_structure}
        data_properties_to_ignore = {self.act, self.has_fife_examples}
        self.generator._class_hierarchy = self.generator._class_hierarchy.restrict_and_copy(remove=concepts_to_ignore)
        self.generator._object_property_hierarchy = (
            self.generator._object_property_hierarchy.restrict_and_copy(remove=object_properties_to_ignore)
        )
        self.generator._data_property_hierarchy = (
            self.generator._data_property_hierarchy.restrict_and_copy(remove=data_properties_to_ignore)
        )

        # get concepts
        concepts = set(self.generator.get_concepts())
        self.assertEqual(84, len(concepts))
        self.assertTrue(self.bond1 not in concepts)
        self.assertTrue(self.compound not in concepts)

        # direct sub concepts
        classes = set(self.generator.get_direct_sub_concepts(OWLThing))
        true_classes = {self.atom, self.bond, self.ring_structure}
        self.assertEqual(true_classes, classes)

        # all sub concepts
        classes = set(self.generator.get_all_sub_concepts(self.bond))
        true_classes = {self.bond2, self.bond3, self.bond4, self.bond5, self.bond7}
        self.assertEqual(true_classes, classes)

        # all leaf concepts
        classes = set(self.generator.get_leaf_concepts(self.bond))
        self.assertEqual(true_classes, classes)

        # types of an individual
        classes = set(self.generator.get_types(self.bond5225, direct=True))
        self.assertFalse(classes)

        classes = set(self.generator.get_types(self.bond5225))
        true_classes = {self.bond, OWLThing}
        self.assertEqual(true_classes, classes)

        # properties
        object_properties = {self.has_bond, self.has_atom, self.in_structure}
        self.assertEqual(object_properties, set(self.generator.get_object_properties()))

        data_properties = {self.charge, self.logp, self.lumo, self.has_three}
        self.assertEqual(data_properties, set(self.generator.get_data_properties()))

        boolean_data_properties = {self.has_three}
        self.assertEqual(boolean_data_properties, set(self.generator.get_boolean_data_properties()))

        numeric_data_properties = {self.charge, self.logp, self.lumo}
        self.assertEqual(numeric_data_properties, set(self.generator.get_numeric_data_properties()))

        true_res = set(map(OWLObjectSomeValuesFrom, object_properties, repeat(OWLThing)))
        res = set(self.generator.most_general_existential_restrictions(domain=OWLThing))
        self.assertEqual(true_res, res)

    def test_domain_range_retrieval(self):
        # object properties
        self.assertEqual(self.compound, self.generator.get_object_property_domains(self.has_atom))
        self.assertEqual(self.bond, self.generator.get_object_property_domains(self.in_bond))

        self.assertEqual(self.ring_structure, self.generator.get_object_property_ranges(self.in_structure))
        self.assertEqual(OWLThing, self.generator.get_object_property_domains(self.in_structure))
        self.assertEqual(self.atom, self.generator.get_object_property_ranges(self.in_bond))

        # data properties
        self.assertEqual(self.atom, self.generator.get_data_property_domains(self.charge))
        self.assertEqual(self.compound, self.generator.get_data_property_domains(self.act))

        self.assertEqual({DoubleOWLDatatype}, self.generator.get_data_property_ranges(self.charge))
        self.assertEqual({BooleanOWLDatatype}, self.generator.get_data_property_ranges(self.has_fife_examples))

    def test_concept_building(self):
        # negation from iterables
        true_ces = {OWLObjectComplementOf(self.bond), self.atom}
        ces = set(self.generator.negation_from_iterables([self.bond, OWLObjectComplementOf(self.atom)]))
        self.assertEqual(true_ces, ces)

        # intersection from iterables
        true_ces = {OWLObjectIntersectionOf([OWLThing, self.bond]), OWLObjectIntersectionOf([OWLThing, self.atom]),
                    OWLObjectIntersectionOf([OWLDataSomeValuesFrom(self.charge, IntegerOWLDatatype), self.bond]),
                    OWLObjectIntersectionOf([OWLDataSomeValuesFrom(self.charge, IntegerOWLDatatype), self.atom])}

        iter1 = [OWLThing, OWLDataSomeValuesFrom(self.charge, IntegerOWLDatatype)]
        iter2 = [self.bond, self.atom]
        ces = set(self.generator.intersect_from_iterables(iter1, iter2))
        self.assertEqual(true_ces, ces)

        # union from iterables
        true_ces = {OWLObjectUnionOf([OWLThing, self.bond]), OWLObjectUnionOf([OWLThing, self.atom]),
                    OWLObjectUnionOf([OWLDataSomeValuesFrom(self.charge, IntegerOWLDatatype), self.bond]),
                    OWLObjectUnionOf([OWLDataSomeValuesFrom(self.charge, IntegerOWLDatatype), self.atom])}
        iter1 = [OWLThing, OWLDataSomeValuesFrom(self.charge, IntegerOWLDatatype)]
        iter2 = [self.bond, self.atom]
        ces = set(self.generator.union_from_iterables(iter1, iter2))
        self.assertEqual(true_ces, ces)

        # most general existential/universal restrictions
        true_res = set(map(OWLObjectSomeValuesFrom, self.object_properties, repeat(OWLThing)))
        res = set(self.generator.most_general_existential_restrictions(domain=OWLThing))
        self.assertEqual(true_res, res)

        true_res = {OWLObjectAllValuesFrom(filler=OWLThing, property=self.in_bond)}
        res = set(self.generator.most_general_universal_restrictions(domain=self.bond))
        self.assertEqual(true_res, res)

        true_res = {OWLObjectSomeValuesFrom(filler=OWLThing, property=self.has_bond.get_inverse_property())}
        res = set(self.generator.most_general_existential_restrictions_inverse(domain=self.bond))
        self.assertEqual(true_res, res)

        true_res = set(map(OWLObjectAllValuesFrom, map(OWLObjectInverseOf, self.object_properties), repeat(OWLThing)))
        res = set(self.generator.most_general_universal_restrictions_inverse(domain=OWLThing))
        self.assertEqual(true_res, res)

        # general functions for concept building
        ce = self.generator.intersection([self.atom, self.compound, OWLObjectAllValuesFrom(self.in_bond, self.atom)])
        true_ce = OWLObjectIntersectionOf([self.atom, self.compound, OWLObjectAllValuesFrom(self.in_bond, self.atom)])
        self.assertEqual(true_res, res)

        ce = self.generator.union([self.atom, self.compound, OWLObjectAllValuesFrom(self.in_bond, self.atom)])
        true_ce = OWLObjectUnionOf([self.atom, self.compound, OWLObjectAllValuesFrom(self.in_bond, self.atom)])
        self.assertEqual(true_res, res)

        ce = self.generator.existential_restriction(self.atom, self.has_atom)
        true_ce = OWLObjectSomeValuesFrom(self.has_atom, self.atom)
        self.assertEqual(true_ce, ce)

        ce = self.generator.universal_restriction(self.atom, self.has_atom)
        true_ce = OWLObjectAllValuesFrom(self.has_atom, self.atom)
        self.assertEqual(true_ce, ce)

        ce = self.generator.has_value_restriction(self.bond5225, self.has_atom)
        true_ce = OWLObjectHasValue(self.has_atom, self.bond5225)
        self.assertEqual(true_ce, ce)

        ce = self.generator.min_cardinality_restriction(self.atom, self.has_atom, 5)
        true_ce = OWLObjectMinCardinality(5, self.has_atom, self.atom)
        self.assertEqual(true_ce, ce)

        ce = self.generator.max_cardinality_restriction(self.atom, self.has_atom, 20)
        true_ce = OWLObjectMaxCardinality(20, self.has_atom, self.atom)
        self.assertEqual(true_ce, ce)

        ce = self.generator.exact_cardinality_restriction(self.atom, self.has_atom, 4)
        true_ce = OWLObjectExactCardinality(4, self.has_atom, self.atom)
        self.assertEqual(true_ce, ce)

        ce = self.generator.data_existential_restriction(DoubleOWLDatatype, self.act)
        true_ce = OWLDataSomeValuesFrom(self.act, DoubleOWLDatatype)
        self.assertEqual(true_ce, ce)

        ce = self.generator.data_universal_restriction(BooleanOWLDatatype, self.has_fife_examples)
        true_ce = OWLDataAllValuesFrom(self.has_fife_examples, BooleanOWLDatatype)
        self.assertEqual(true_ce, ce)

        ce = self.generator.data_has_value_restriction(OWLLiteral(True), self.has_three)
        true_ce = OWLDataHasValue(self.has_three, OWLLiteral(True))
        self.assertEqual(true_ce, ce)

        ce = self.generator.negation(OWLThing)
        self.assertEqual(OWLNothing, ce)

        ce = self.generator.negation(OWLDataHasValue(self.has_three, OWLLiteral(True)))
        self.assertEqual(OWLObjectComplementOf(OWLDataHasValue(self.has_three, OWLLiteral(True))), ce)
