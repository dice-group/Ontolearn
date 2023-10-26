import unittest
from itertools import repeat

from ontolearn.concept_generator import ConceptGenerator
from ontolearn.knowledge_base import KnowledgeBase

from ontolearn.owlapy.model import OWLObjectUnionOf, OWLSubDataPropertyOfAxiom, OWLSubObjectPropertyOfAxiom, OWLThing, \
    BooleanOWLDatatype, DoubleOWLDatatype, IntegerOWLDatatype, OWLClass, OWLDataAllValuesFrom, \
    OWLDataHasValue, OWLDataProperty, OWLDataSomeValuesFrom, OWLLiteral, OWLNamedIndividual, \
    OWLNothing, OWLObjectAllValuesFrom, OWLObjectComplementOf, OWLObjectExactCardinality, \
    OWLObjectHasValue, OWLObjectIntersectionOf, OWLObjectInverseOf, OWLObjectMaxCardinality, \
    OWLObjectMinCardinality, OWLObjectProperty, IRI, OWLObjectSomeValuesFrom


class KnowledgeBaseTest(unittest.TestCase):

    def setUp(self):
        self.namespace = "http://dl-learner.org/mutagenesis#"
        self.kb = KnowledgeBase(path="KGs/Mutagenesis/mutagenesis.owl")
        self.onto = self.kb.ontology()
        self.mgr = self.onto.get_owl_ontology_manager()
        self.generator = ConceptGenerator()
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

        self.anthracene = OWLClass(IRI.create(self.namespace, 'Anthracene'))
        self.ball3 = OWLClass(IRI.create(self.namespace, 'Ball3'))
        self.benzene = OWLClass(IRI.create(self.namespace, 'Benzene'))
        self.carbon_5_aromatic_ring = OWLClass(IRI.create(self.namespace, 'Carbon_5_aromatic_ring'))
        self.carbon_6_ring = OWLClass(IRI.create(self.namespace, 'Carbon_6_ring'))
        self.hetero_aromatic_5_ring = OWLClass(IRI.create(self.namespace, 'Hetero_aromatic_5_ring'))
        self.hetero_aromatic_6_ring = OWLClass(IRI.create(self.namespace, 'Hetero_aromatic_6_ring'))
        self.methyl = OWLClass(IRI.create(self.namespace, 'Methyl'))
        self.nitro = OWLClass(IRI.create(self.namespace, 'Nitro'))
        self.phenanthrene = OWLClass(IRI.create(self.namespace, 'Phenanthrene'))
        self.ring_size_5 = OWLClass(IRI.create(self.namespace, 'Ring_size_5'))
        self.ring_size_6 = OWLClass(IRI.create(self.namespace, 'Ring_size_6'))

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
        self.assertEqual(86, len(list(self.kb.get_concepts())))

        # direct sub concepts
        classes = frozenset(self.kb.get_direct_sub_concepts(OWLThing))
        true_classes = {self.atom, self.bond, self.compound, self.ring_structure}
        self.assertEqual(true_classes, classes)

        # all sub concepts
        classes = frozenset(self.kb.get_all_sub_concepts(self.bond))
        true_classes = {self.bond1, self.bond2, self.bond3, self.bond4, self.bond5, self.bond7}
        self.assertEqual(true_classes, classes)

        # all leaf concepts
        classes = frozenset(self.kb.get_leaf_concepts(self.bond))
        self.assertEqual(true_classes, classes)

        # get direct parents
        classes = frozenset(self.kb.get_direct_parents(self.bond1))
        true_classes = {self.bond}
        self.assertEqual(true_classes, classes)

        # types of an individual
        classes = frozenset(self.kb.get_types(self.bond5225, direct=True))
        true_classes = {self.bond1}
        self.assertEqual(true_classes, classes)

        classes = frozenset(self.kb.get_types(self.bond5225))
        true_classes = {self.bond, self.bond1, OWLThing}
        self.assertEqual(true_classes, classes)

        # direct sub-concepts
        classes = frozenset(self.kb.get_all_direct_sub_concepts(self.ring_structure))
        true_classes = {self.anthracene, self.ball3, self.benzene, self.carbon_5_aromatic_ring, self.carbon_6_ring,
                        self.hetero_aromatic_5_ring, self.hetero_aromatic_6_ring, self.methyl, self.nitro,
                        self.phenanthrene, self.ring_size_5, self.ring_size_6}
        self.assertEqual(true_classes, classes)

    def test_property_retrieval(self):
        self.assertEqual(self.object_properties, frozenset(self.kb.get_object_properties()))
        self.assertEqual(self.data_properties, frozenset(self.kb.get_data_properties()))
        self.assertEqual(self.boolean_data_properties, frozenset(self.kb.get_boolean_data_properties()))
        self.assertEqual(self.numeric_data_properties, frozenset(self.kb.get_numeric_data_properties()))
        self.assertFalse(frozenset(self.kb.get_time_data_properties()))

        # most general data properties
        self.assertEqual(self.boolean_data_properties,
                         frozenset(self.kb.most_general_boolean_data_properties(domain=self.compound)))
        self.assertFalse(frozenset(self.kb.most_general_boolean_data_properties(domain=self.bond)))

        self.assertEqual({self.charge},
                         frozenset(self.kb.most_general_numeric_data_properties(domain=self.atom)))
        self.assertFalse(frozenset(self.kb.most_general_numeric_data_properties(domain=self.bond)))

        self.data_properties.remove(self.charge)
        self.assertEqual(self.data_properties,
                         frozenset(self.kb.most_general_data_properties(domain=self.compound)))
        self.assertFalse(frozenset(self.kb.most_general_data_properties(domain=self.bond)))

        self.assertFalse(frozenset(self.kb.most_general_time_data_properties(domain=OWLThing)))

        # object property values of an individual
        inds = frozenset(self.kb.get_object_property_values(self.bond5225, self.in_bond, direct=True))
        true_inds = {self.d91_32, self.d91_17}
        self.assertEqual(true_inds, inds)

        # indirect object property values of an individual
        super_in_bond = OWLObjectProperty(IRI.create(self.namespace, 'super_inBond'))
        self.mgr.add_axiom(self.onto, OWLSubObjectPropertyOfAxiom(self.in_bond, super_in_bond))
        inds = frozenset(self.kb.get_object_property_values(self.bond5225, super_in_bond, direct=False))
        true_inds = {self.d91_32, self.d91_17}
        self.assertEqual(true_inds, inds)
        inds = frozenset(self.kb.get_object_property_values(self.bond5225, super_in_bond, direct=True))
        self.assertEqual(frozenset(), inds)
        self.mgr.remove_axiom(self.onto, OWLSubObjectPropertyOfAxiom(self.in_bond, super_in_bond))

        # data property values of an individual
        values = frozenset(self.kb.get_data_property_values(self.d91_32, self.charge, direct=True))
        true_values = {OWLLiteral(0.146)}
        self.assertEqual(true_values, values)

        # indirect data property values of an individual
        super_charge = OWLDataProperty(IRI.create(self.namespace, 'super_charge'))
        self.mgr.add_axiom(self.onto, OWLSubDataPropertyOfAxiom(self.charge, super_charge))
        values = frozenset(self.kb.get_data_property_values(self.d91_32, super_charge, direct=False))
        true_values = {OWLLiteral(0.146)}
        self.assertEqual(true_values, values)
        values = frozenset(self.kb.get_data_property_values(self.d91_32, super_charge, direct=True))
        self.assertEqual(frozenset(), values)
        self.mgr.remove_axiom(self.onto, OWLSubDataPropertyOfAxiom(self.charge, super_charge))

        # object properties of an individual
        properties = frozenset(self.kb.get_object_properties_for_ind(self.bond5225, direct=True))
        true_properties = {self.in_bond}
        self.assertEqual(true_properties, properties)

        # indirect object properties of an individual
        self.mgr.add_axiom(self.onto, OWLSubObjectPropertyOfAxiom(self.in_bond, self.has_bond))
        properties = frozenset(self.kb.get_object_properties_for_ind(self.bond5225, direct=False))
        true_properties = {self.in_bond, self.has_bond}
        self.assertEqual(true_properties, properties)
        properties = frozenset(self.kb.get_object_properties_for_ind(self.bond5225, direct=True))
        true_properties = {self.in_bond}
        self.assertEqual(true_properties, properties)
        self.mgr.remove_axiom(self.onto, OWLSubObjectPropertyOfAxiom(self.in_bond, self.has_bond))

        # data properties of an individual
        properties = frozenset(self.kb.get_data_properties_for_ind(self.d91_32, direct=True))
        true_properties = {self.charge}
        self.assertEqual(true_properties, properties)

        # indirect data properties of an individual
        self.mgr.add_axiom(self.onto, OWLSubDataPropertyOfAxiom(self.charge, self.act))
        properties = frozenset(self.kb.get_data_properties_for_ind(self.d91_32, direct=False))
        true_properties = {self.charge, self.act}
        self.assertEqual(true_properties, properties)
        properties = frozenset(self.kb.get_data_properties_for_ind(self.d91_32, direct=True))
        true_properties = {self.charge}
        self.assertEqual(true_properties, properties)
        self.mgr.remove_axiom(self.onto, OWLSubDataPropertyOfAxiom(self.charge, self.act))

    def test_ignore(self):
        concepts_to_ignore = {self.bond1, self.compound}
        object_properties_to_ignore = {self.in_bond, self.has_structure}
        data_properties_to_ignore = {self.act, self.has_fife_examples}
        self.kb._class_hierarchy = self.kb._class_hierarchy.restrict_and_copy(remove=concepts_to_ignore)
        self.kb._object_property_hierarchy = (
            self.kb._object_property_hierarchy.restrict_and_copy(remove=object_properties_to_ignore)
        )
        self.kb._data_property_hierarchy = (
            self.kb._data_property_hierarchy.restrict_and_copy(remove=data_properties_to_ignore)
        )

        # get concepts
        concepts = frozenset(self.kb.get_concepts())
        self.assertEqual(84, len(concepts))
        self.assertTrue(self.bond1 not in concepts)
        self.assertTrue(self.compound not in concepts)

        # direct sub concepts
        classes = frozenset(self.kb.get_direct_sub_concepts(OWLThing))
        true_classes = {self.atom, self.bond, self.ring_structure}
        self.assertEqual(true_classes, classes)

        # all sub concepts
        classes = frozenset(self.kb.get_all_sub_concepts(self.bond))
        true_classes = {self.bond2, self.bond3, self.bond4, self.bond5, self.bond7}
        self.assertEqual(true_classes, classes)

        # all leaf concepts
        classes = frozenset(self.kb.get_leaf_concepts(self.bond))
        self.assertEqual(true_classes, classes)

        # types of an individual
        classes = frozenset(self.kb.get_types(self.bond5225, direct=True))
        self.assertFalse(classes)

        classes = frozenset(self.kb.get_types(self.bond5225))
        true_classes = {self.bond, OWLThing}
        self.assertEqual(true_classes, classes)

        # properties
        object_properties = {self.has_bond, self.has_atom, self.in_structure}
        self.assertEqual(object_properties, frozenset(self.kb.get_object_properties()))

        data_properties = {self.charge, self.logp, self.lumo, self.has_three}
        self.assertEqual(data_properties, frozenset(self.kb.get_data_properties()))

        boolean_data_properties = {self.has_three}
        self.assertEqual(boolean_data_properties, frozenset(self.kb.get_boolean_data_properties()))

        numeric_data_properties = {self.charge, self.logp, self.lumo}
        self.assertEqual(numeric_data_properties, frozenset(self.kb.get_numeric_data_properties()))

        true_res = frozenset(map(OWLObjectSomeValuesFrom, object_properties, repeat(OWLThing)))
        res = frozenset(self.kb.most_general_existential_restrictions(domain=OWLThing))
        self.assertEqual(true_res, res)

    def test_domain_range_retrieval(self):
        # object properties
        self.assertEqual(self.compound, self.kb.get_object_property_domains(self.has_atom))
        self.assertEqual(self.bond, self.kb.get_object_property_domains(self.in_bond))

        self.assertEqual(self.ring_structure, self.kb.get_object_property_ranges(self.in_structure))
        self.assertEqual(OWLThing, self.kb.get_object_property_domains(self.in_structure))
        self.assertEqual(self.atom, self.kb.get_object_property_ranges(self.in_bond))

        # data properties
        self.assertEqual(self.atom, self.kb.get_data_property_domains(self.charge))
        self.assertEqual(self.compound, self.kb.get_data_property_domains(self.act))

        self.assertEqual({DoubleOWLDatatype}, self.kb.get_data_property_ranges(self.charge))
        self.assertEqual({BooleanOWLDatatype}, self.kb.get_data_property_ranges(self.has_fife_examples))

    def test_concept_generation(self):
        # negation from iterables
        true_ces = {OWLObjectComplementOf(self.bond), self.atom}
        ces = frozenset(self.generator.negation_from_iterables([self.bond, OWLObjectComplementOf(self.atom)]))
        self.assertEqual(true_ces, ces)

        # intersection from iterables
        true_ces = {OWLObjectIntersectionOf([OWLThing, self.bond]), OWLObjectIntersectionOf([OWLThing, self.atom]),
                    OWLObjectIntersectionOf([OWLDataSomeValuesFrom(self.charge, IntegerOWLDatatype), self.bond]),
                    OWLObjectIntersectionOf([OWLDataSomeValuesFrom(self.charge, IntegerOWLDatatype), self.atom])}

        iter1 = [OWLThing, OWLDataSomeValuesFrom(self.charge, IntegerOWLDatatype)]
        iter2 = [self.bond, self.atom]
        ces = frozenset(self.generator.intersect_from_iterables(iter1, iter2))
        self.assertEqual(true_ces, ces)

        # union from iterables
        true_ces = {OWLObjectUnionOf([OWLThing, self.bond]), OWLObjectUnionOf([OWLThing, self.atom]),
                    OWLObjectUnionOf([OWLDataSomeValuesFrom(self.charge, IntegerOWLDatatype), self.bond]),
                    OWLObjectUnionOf([OWLDataSomeValuesFrom(self.charge, IntegerOWLDatatype), self.atom])}
        iter1 = [OWLThing, OWLDataSomeValuesFrom(self.charge, IntegerOWLDatatype)]
        iter2 = [self.bond, self.atom]
        ces = frozenset(self.generator.union_from_iterables(iter1, iter2))
        self.assertEqual(true_ces, ces)

        # most general existential/universal restrictions
        true_res = frozenset(map(OWLObjectSomeValuesFrom, self.object_properties, repeat(OWLThing)))
        res = frozenset(self.kb.most_general_existential_restrictions(domain=OWLThing))
        self.assertEqual(true_res, res)

        true_res = {OWLObjectAllValuesFrom(filler=OWLThing, property=self.in_bond),
                    OWLObjectAllValuesFrom(filler=OWLThing, property=self.in_structure)}
        res = frozenset(self.kb.most_general_universal_restrictions(domain=self.bond))
        self.assertEqual(true_res, res)

        true_res = {OWLObjectSomeValuesFrom(filler=OWLThing, property=self.has_bond.get_inverse_property()),
                    OWLObjectSomeValuesFrom(filler=OWLThing, property=self.has_structure.get_inverse_property())}
        res = frozenset(self.kb.most_general_existential_restrictions_inverse(domain=self.bond))
        self.assertEqual(true_res, res)

        true_res = frozenset(map(OWLObjectAllValuesFrom,
                                 map(OWLObjectInverseOf, self.object_properties), repeat(OWLThing)))
        res = frozenset(self.kb.most_general_universal_restrictions_inverse(domain=OWLThing))
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

    def test_repr(self):
        representation = repr(self.kb)
        self.assertEqual("KnowledgeBase(path='KGs/Mutagenesis/mutagenesis.owl' <86 classes, 11 properties,"
                         " 14145 individuals)", representation)
