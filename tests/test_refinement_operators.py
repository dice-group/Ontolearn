""" Test for refinement_operators.py"""
from functools import partial
from itertools import repeat
from pytest import mark
import unittest

import json

from ontolearn import KnowledgeBase
from ontolearn.core.owl.utils import ConceptOperandSorter
from ontolearn.utils import setup_logging
from owlapy.model.providers import OWLDatatypeMaxInclusiveRestriction, OWLDatatypeMinInclusiveRestriction
from owlapy.render import DLSyntaxObjectRenderer
from owlapy.model import OWLObjectMinCardinality, OWLObjectProperty, OWLObjectSomeValuesFrom, OWLObjectUnionOf, \
    OWLClass, IRI, OWLDataHasValue, OWLDataProperty, OWLDataSomeValuesFrom, OWLLiteral, OWLObjectAllValuesFrom, \
    OWLObjectCardinalityRestriction, OWLObjectComplementOf, OWLObjectIntersectionOf, OWLObjectMaxCardinality
from ontolearn.refinement_operators import CustomRefinementOperator, ModifiedCELOERefinement, LengthBasedRefinement, \
    ExpressRefinement


setup_logging("logging_test.conf")


class ModifiedCELOERefinementTest(unittest.TestCase):

    def setUp(self):
        self.kb = KnowledgeBase(path='KGs/Mutagenesis/mutagenesis.owl')
        namespace_ = "http://dl-learner.org/mutagenesis#"

        # Classes
        self.atom = OWLClass(IRI.create(namespace_, 'Atom'))
        self.bond = OWLClass(IRI.create(namespace_, 'Bond'))
        self.compound = OWLClass(IRI(namespace_, 'Compound'))
        self.ring_structure = OWLClass(IRI(namespace_, 'RingStructure'))
        self.bond1 = OWLClass(IRI.create(namespace_, 'Bond-1'))
        self.bond2 = OWLClass(IRI.create(namespace_, 'Bond-2'))
        self.bond3 = OWLClass(IRI.create(namespace_, 'Bond-3'))
        self.bond4 = OWLClass(IRI.create(namespace_, 'Bond-4'))
        self.bond5 = OWLClass(IRI.create(namespace_, 'Bond-5'))
        self.bond7 = OWLClass(IRI.create(namespace_, 'Bond-7'))
        self.ball3 = OWLClass(IRI.create(namespace_, 'Ball3'))

        self.all_bond_classes = {self.bond1, self.bond2, self.bond3, self.bond4, self.bond5, self.bond7}

        # Object Properties
        self.in_bond = OWLObjectProperty(IRI.create(namespace_, 'inBond'))
        self.has_bond = OWLObjectProperty(IRI.create(namespace_, 'hasBond'))
        self.has_atom = OWLObjectProperty(IRI.create(namespace_, 'hasAtom'))
        self.in_structure = OWLObjectProperty(IRI.create(namespace_, 'inStructure'))
        self.has_structure = OWLObjectProperty(IRI.create(namespace_, 'hasStructure'))

        # Data Properties
        self.charge = OWLDataProperty(IRI.create(namespace_, 'charge'))
        self.act = OWLDataProperty(IRI.create(namespace_, 'act'))

        self.has_fife_examples = OWLDataProperty(IRI.create(namespace_, 'hasFifeExamplesOfAcenthrylenes'))
        self.has_three = OWLDataProperty(IRI.create(namespace_, 'hasThreeOrMoreFusedRings'))

    def test_atomic_refinements_classes(self):
        thing_true_refs = {self.atom, self.bond, self.compound, self.ring_structure,
                           OWLObjectComplementOf(self.bond1), OWLObjectComplementOf(self.ball3),
                           OWLObjectComplementOf(self.bond4)}
        thing_false_refs = {self.bond1, self.bond2, self.bond4, self.ball3, OWLObjectComplementOf(self.atom)}

        bond_true_refs = {self.bond1, self.bond2, self.bond4, OWLObjectComplementOf(self.bond1),
                          OWLObjectComplementOf(self.bond2)}
        bond_false_refs = {self.atom, self.bond, self.compound, self.ring_structure,
                           OWLObjectComplementOf(self.ball3)}

        rho = ModifiedCELOERefinement(self.kb, use_negation=True)
        thing_refs = set(rho.refine(self.kb.thing, max_length=2, current_domain=self.kb.thing))
        bond_refs = set(rho.refine(self.bond, max_length=2, current_domain=self.kb.thing))

        self.assertLessEqual(thing_true_refs, thing_refs)
        self.assertLessEqual(bond_true_refs, bond_refs)
        self.assertFalse(thing_refs & thing_false_refs)
        self.assertFalse(bond_refs & bond_false_refs)

    def test_atomic_refinements_existential_universal(self):
        thing_true_refs = {OWLObjectSomeValuesFrom(self.in_bond, self.kb.thing),
                           OWLObjectAllValuesFrom(self.has_bond, self.kb.thing),
                           OWLObjectSomeValuesFrom(self.has_atom.get_inverse_property(), self.kb.thing),
                           OWLObjectAllValuesFrom(self.in_structure.get_inverse_property(), self.kb.thing)}
        bond_true_refs = {OWLObjectSomeValuesFrom(self.in_bond, self.kb.thing),
                          OWLObjectAllValuesFrom(self.in_bond, self.kb.thing),
                          OWLObjectSomeValuesFrom(self.has_bond.get_inverse_property(), self.kb.thing),
                          OWLObjectAllValuesFrom(self.has_bond.get_inverse_property(), self.kb.thing)}
        bond_false_refs = {OWLObjectSomeValuesFrom(self.has_bond, self.kb.thing),
                           OWLObjectAllValuesFrom(self.has_bond, self.kb.thing)}

        rho = ModifiedCELOERefinement(self.kb, use_negation=True, use_all_constructor=True, use_inverse=True)
        thing_refs = set(rho.refine(self.kb.thing, max_length=3, current_domain=self.kb.thing))
        bond_refs = set(rho.refine(self.bond, max_length=3, current_domain=self.kb.thing))
        self.assertLessEqual(thing_true_refs, thing_refs)
        self.assertLessEqual(bond_true_refs, bond_refs)
        self.assertFalse(bond_refs & bond_false_refs)

        # max_length = 2 so property refinements should not be generated
        for i in rho.refine(self.kb.thing, max_length=2, current_domain=self.kb.thing):
            self.assertFalse(isinstance(i, OWLObjectSomeValuesFrom))
            self.assertFalse(isinstance(i, OWLObjectAllValuesFrom))

        for i in rho.refine(self.bond, max_length=2, current_domain=self.kb.thing):
            self.assertFalse(isinstance(i, OWLObjectSomeValuesFrom))
            self.assertFalse(isinstance(i, OWLObjectAllValuesFrom))

    def test_atomic_refinements_union_intersection(self):
        rho = ModifiedCELOERefinement(self.kb)
        true_refs = {OWLObjectUnionOf([self.bond, self.atom]), OWLObjectUnionOf([self.bond, self.compound]),
                     OWLObjectUnionOf([self.ring_structure, self.atom]),
                     OWLObjectUnionOf([self.bond, self.ring_structure]),
                     OWLObjectUnionOf([self.ring_structure, self.compound]),
                     OWLObjectUnionOf([self.atom, self.compound])}
        sorter = ConceptOperandSorter()
        true_refs = {sorter.sort(i) for i in true_refs}
        thing_refs = set(rho.refine(self.kb.thing, max_length=3, current_domain=self.kb.thing))
        thing_refs = {sorter.sort(i) for i in thing_refs}
        self.assertLessEqual(true_refs, thing_refs)

        # max_length = 2 so union or intersection refinements should not be generated
        for i in rho.refine(self.kb.thing, max_length=2, current_domain=self.kb.thing):
            self.assertFalse(isinstance(i, OWLObjectIntersectionOf))
            self.assertFalse(isinstance(i, OWLObjectUnionOf))

    def test_atomic_refinements_data_properties(self):
        rho = ModifiedCELOERefinement(self.kb, use_numeric_datatypes=True, use_boolean_datatype=True)
        # Just set some static splits
        splits = list(map(OWLLiteral, range(1, 10)))
        rho.dp_splits = {p: splits for p in rho.dp_splits}

        # numeric
        true_act = {OWLDataSomeValuesFrom(self.act, OWLDatatypeMinInclusiveRestriction(1)),
                    OWLDataSomeValuesFrom(self.act, OWLDatatypeMaxInclusiveRestriction(9))}
        true_charge = {OWLDataSomeValuesFrom(self.charge, OWLDatatypeMinInclusiveRestriction(1)),
                       OWLDataSomeValuesFrom(self.charge, OWLDatatypeMaxInclusiveRestriction(9))}
        thing_refs = set(rho.refine(self.kb.thing, max_length=3, current_domain=self.kb.thing))
        compound_refs = set(rho.refine(self.compound, max_length=3, current_domain=self.kb.thing))
        bond_refs = set(rho.refine(self.bond, max_length=3, current_domain=self.kb.thing))
        self.assertLessEqual(true_act, thing_refs)
        self.assertLessEqual(true_act, compound_refs)
        self.assertFalse(true_act & bond_refs)
        self.assertLessEqual(true_charge, thing_refs)
        self.assertFalse(true_charge & compound_refs)
        self.assertFalse(true_charge & bond_refs)

        # boolean
        true_boolean = {OWLDataHasValue(self.has_three, OWLLiteral(True)),
                        OWLDataHasValue(self.has_three, OWLLiteral(False)),
                        OWLDataHasValue(self.has_fife_examples, OWLLiteral(True)),
                        OWLDataHasValue(self.has_fife_examples, OWLLiteral(False))}
        self.assertLessEqual(true_boolean, thing_refs)
        self.assertLessEqual(true_boolean, compound_refs)
        self.assertFalse(true_boolean & bond_refs)

        # max_length = 2 so data property refinements should not be generated
        for i in rho.refine(self.kb.thing, max_length=2, current_domain=self.kb.thing):
            self.assertFalse(isinstance(i, OWLDataSomeValuesFrom))
            self.assertFalse(isinstance(i, OWLDataHasValue))

    def test_atomic_refinements_cardinality(self):
        rho = ModifiedCELOERefinement(self.kb, card_limit=10, use_card_restrictions=True)
        thing_true_refs = {OWLObjectMaxCardinality(9, self.has_bond, self.kb.thing),
                           OWLObjectMaxCardinality(9, self.has_atom, self.kb.thing),
                           OWLObjectMaxCardinality(1, self.in_bond, self.kb.thing),
                           OWLObjectMaxCardinality(9, self.has_structure, self.kb.thing)}
        thing_refs = set(rho.refine(self.kb.thing, max_length=4, current_domain=self.kb.thing))
        bond_refs = set(rho.refine(self.bond, max_length=4, current_domain=self.kb.thing))
        self.assertLessEqual(thing_true_refs, thing_refs)
        self.assertIn(OWLObjectMaxCardinality(1, self.in_bond, self.kb.thing), bond_refs)

        # max_length = 3 so cardinality refinements should not be generated
        thing_refs = set(rho.refine(self.kb.thing, max_length=3, current_domain=self.kb.thing))
        bond_refs = set(rho.refine(self.bond, max_length=3, current_domain=self.kb.thing))
        self.assertFalse(thing_true_refs & thing_refs)
        self.assertNotIn(OWLObjectMaxCardinality(1, self.in_bond, self.kb.thing), bond_refs)

    def test_atomic_use_flags(self):
        rho = ModifiedCELOERefinement(self.kb, use_negation=False, use_all_constructor=False,
                                      use_numeric_datatypes=False, use_boolean_datatype=False,
                                      use_card_restrictions=False)

        for i in rho.refine(self.kb.thing, max_length=4, current_domain=self.kb.thing):
            self.assertFalse(isinstance(i, OWLObjectAllValuesFrom))
            self.assertFalse(isinstance(i, OWLDataSomeValuesFrom))
            self.assertFalse(isinstance(i, OWLDataHasValue))
            self.assertFalse(isinstance(i, OWLObjectCardinalityRestriction))
            self.assertFalse(isinstance(i, OWLObjectComplementOf))

    def test_complement_of_refinements(self):
        rho = ModifiedCELOERefinement(self.kb, use_negation=True)
        bond_refs = set(rho.refine(OWLObjectComplementOf(self.bond1), max_length=3, current_domain=self.kb.thing))
        self.assertEqual({OWLObjectComplementOf(self.bond)}, bond_refs)

        ball3_refs = set(rho.refine(OWLObjectComplementOf(self.ball3), max_length=3, current_domain=self.kb.thing))
        self.assertEqual({OWLObjectComplementOf(self.ring_structure)}, ball3_refs)

    def test_object_some_values_from_refinements(self):
        rho = ModifiedCELOERefinement(self.kb, use_all_constructor=True, use_card_restrictions=True, card_limit=10)
        true_refs = set(map(partial(OWLObjectSomeValuesFrom, self.in_bond), self.all_bond_classes))
        true_refs.add(OWLObjectAllValuesFrom(self.in_bond, self.bond))
        refs = set(rho.refine(OWLObjectSomeValuesFrom(self.in_bond, self.bond),
                              max_length=3, current_domain=self.kb.thing))
        self.assertEqual(refs, true_refs)

        refs = set(rho.refine(OWLObjectSomeValuesFrom(self.in_bond, self.bond),
                              max_length=4, current_domain=self.kb.thing))
        self.assertIn(OWLObjectMinCardinality(2, self.in_bond, self.bond), refs)

    def test_object_all_values_from_refinements(self):
        rho = ModifiedCELOERefinement(self.kb, use_all_constructor=True)
        true_refs = set(map(partial(OWLObjectAllValuesFrom, self.in_bond), self.all_bond_classes))
        refs = set(rho.refine(OWLObjectAllValuesFrom(self.in_bond, self.bond),
                              max_length=3, current_domain=self.kb.thing))
        self.assertEqual(refs, true_refs)

    def test_intersection_refinements(self):
        rho = ModifiedCELOERefinement(self.kb)
        true_refs = set(map(OWLObjectIntersectionOf, zip(self.all_bond_classes, repeat(self.ball3))))
        refs = set(rho.refine(OWLObjectIntersectionOf([self.bond, self.ball3]),
                              max_length=3, current_domain=self.kb.thing))
        self.assertEqual(refs, true_refs)

    def test_union_refinements(self):
        rho = ModifiedCELOERefinement(self.kb)
        true_refs = set(map(OWLObjectUnionOf, zip(self.all_bond_classes, repeat(self.ball3))))
        refs = set(rho.refine(OWLObjectUnionOf([self.bond, self.ball3]), max_length=3, current_domain=self.kb.thing))
        self.assertEqual(refs, true_refs)

    def test_data_some_values_from_refinements(self):
        rho = ModifiedCELOERefinement(self.kb, use_numeric_datatypes=True)
        # Just set some static splits
        splits = list(map(OWLLiteral, range(1, 10)))
        rho.dp_splits = {p: splits for p in rho.dp_splits}

        # min inclusive
        refs = set(rho.refine(OWLDataSomeValuesFrom(self.charge, OWLDatatypeMinInclusiveRestriction(4)),
                              max_length=0, current_domain=self.kb.thing))
        true_refs = {OWLDataSomeValuesFrom(self.charge, OWLDatatypeMinInclusiveRestriction(5))}
        self.assertEqual(refs, true_refs)

        # test empty
        refs = set(rho.refine(OWLDataSomeValuesFrom(self.act, OWLDatatypeMinInclusiveRestriction(9)),
                              max_length=0, current_domain=self.kb.thing))
        self.assertFalse(refs)

        # max inclusive
        refs = set(rho.refine(OWLDataSomeValuesFrom(self.charge, OWLDatatypeMaxInclusiveRestriction(8)),
                              max_length=0, current_domain=self.kb.thing))
        true_refs = {OWLDataSomeValuesFrom(self.charge, OWLDatatypeMaxInclusiveRestriction(7))}
        self.assertEqual(refs, true_refs)

        # test empty
        refs = set(rho.refine(OWLDataSomeValuesFrom(self.act, OWLDatatypeMaxInclusiveRestriction(1)),
                              max_length=0, current_domain=self.kb.thing))
        self.assertFalse(refs)

    def test_cardinality_refinements(self):
        rho = ModifiedCELOERefinement(self.kb, card_limit=10, use_card_restrictions=True)

        # min cardinality
        refs = set(rho.refine(OWLObjectMinCardinality(4, self.has_atom, self.bond1),
                              max_length=0, current_domain=self.kb.thing))
        true_refs = {OWLObjectMinCardinality(5, self.has_atom, self.bond1)}
        self.assertEqual(true_refs, refs)

        # test empty
        refs = set(rho.refine(OWLObjectMinCardinality(10, self.has_atom, self.bond1),
                              max_length=0, current_domain=self.kb.thing))
        self.assertFalse(refs)

        # max cardinality
        refs = set(rho.refine(OWLObjectMaxCardinality(7, self.has_atom, self.bond1),
                              max_length=0, current_domain=self.kb.thing))
        true_refs = {OWLObjectMaxCardinality(6, self.has_atom, self.bond1)}
        self.assertEqual(true_refs, refs)

        # test empty
        refs = set(rho.refine(OWLObjectMaxCardinality(0, self.has_atom, self.bond1),
                              max_length=0, current_domain=self.kb.thing))
        self.assertFalse(refs)

    def test_max_nr_fillers(self):
        rho = ModifiedCELOERefinement(self.kb, card_limit=10, use_inverse=True, use_card_restrictions=True)
        self.assertEqual(rho.max_nr_fillers[self.in_bond], 2)
        self.assertEqual(rho.max_nr_fillers[self.has_bond], rho.card_limit)
        self.assertEqual(rho.max_nr_fillers[self.has_atom], rho.card_limit)
        self.assertEqual(rho.max_nr_fillers[self.in_structure], 0)
        self.assertEqual(rho.max_nr_fillers[self.has_structure], rho.card_limit)
        self.assertEqual(rho.max_nr_fillers[self.in_bond.get_inverse_property()], 4)
        self.assertEqual(rho.max_nr_fillers[self.has_bond.get_inverse_property()], 1)
        self.assertEqual(rho.max_nr_fillers[self.has_atom.get_inverse_property()], 1)
        self.assertEqual(rho.max_nr_fillers[self.in_structure.get_inverse_property()], 0)
        self.assertEqual(rho.max_nr_fillers[self.has_structure.get_inverse_property()], 1)


with open('examples/synthetic_problems.json') as json_file:
    settings = json.load(json_file)
# because '../KGs/Family/family-benchmark_rich_background.owl'
kb = KnowledgeBase(path=settings['data_path'][3:])


class LengthBasedRefinementTest(unittest.TestCase):

    def test_length_refinement_operator(self):
        r = DLSyntaxObjectRenderer()
        rho = LengthBasedRefinement(kb)
        for _ in enumerate(rho.refine(kb.thing)):
            print(r.render(_[1]))
            pass


class ExpressRefinementTest(unittest.TestCase):

    def test_express_refinement_operator(self):
        r = DLSyntaxObjectRenderer()
        rho = ExpressRefinement(kb)
        for _ in enumerate(rho.refine(kb.thing)):
            print(r.render(_[1]))
            pass


@mark.xfail
class CustomRefinementTest(unittest.TestCase):

    def test_custom_refinement_operator(self):
        r = DLSyntaxObjectRenderer()
        rho = CustomRefinementOperator(kb)
        for _ in enumerate(rho.refine(kb.thing)):
            print(r.render(_[1]))
            pass


if __name__ == '__main__':
    unittest.main()
