from typing import Optional
import unittest
from unittest.mock import MagicMock
from ontolearn.concept_abstract_syntax_tree import ConceptAbstractSyntaxTreeBuilder
from ontolearn.knowledge_base import KnowledgeBase
from owlapy.class_expression import  OWLClass, OWLObjectSomeValuesFrom,  OWLObjectAllValuesFrom, OWLObjectUnionOf, OWLObjectIntersectionOf, OWLClassExpression
from owlapy.iri import IRI
from owlapy.owl_property import OWLObjectProperty
from owlapy.parser import DLSyntaxParser
from ontolearn.utils.static_funcs import assert_class_expression_type


class TestConceptAbstractSyntaxTreeBuilder(unittest.TestCase):
    """
    Unit tests for ConceptAbstractSyntaxTreeBuilder, validating parsing and grammar strategies for DL expressions.
    """

    def setUp(self):
        """
        Set up mock ontology, roles, concepts, and parser for test cases.
        """
        def iri_for(name: str) -> IRI:
            return IRI.create(f"http://example.org/{name}")
        
        self.mock_concepts = [
            'Brother', 'Daughter', 'Father', 'Mother', 'Female', 'Granddaughter', 'Grandfather',
            'Grandmother', 'Grandparent', 'Person', 'PersonWithASibling', 'Sister',
            'Thing'
        ]
        self.mock_roles = ['hasChild', 'hasParent', 'hasSibling', 'married']

        mock_ontology = MagicMock()

        mock_ontology.classes_in_signature.return_value = [
            OWLClass(iri_for(c)) for c in self.mock_concepts
        ]

        mock_ontology.object_properties_in_signature.return_value = [
            OWLObjectProperty(iri_for(r)) for r in self.mock_roles
        ]

        self.mock_kb = MagicMock(spec=KnowledgeBase)
        self.mock_kb.ontology = mock_ontology

        kb_namespace = mock_ontology.classes_in_signature.return_value[0].iri.get_namespace()
        self.dl_parser = DLSyntaxParser(kb_namespace)

        self.concept_abst_builder = ConceptAbstractSyntaxTreeBuilder(knowledge_base=self.mock_kb, max_length=20)
        self.concept_abst_builder.unique_atom_concept_names = set(self.mock_concepts)
        self.concept_abst_builder.unique_roles = set(self.mock_roles)
    
    def assert_dl_parseable(self, expr_str: str, expected_type: Optional[type] = None, instances_check: bool = False):
        """
        Assert that a DL expression string is parseable and matches the expected type.
        """

        try:
            parsed = self.dl_parser.parse(expr_str)
            self.assertIsNotNone(parsed, f"Parser returned None for: {expr_str}")

            if instances_check:
                assert_class_expression_type(expr_str)
            else:
                self.assertIsInstance(parsed, expected_type,
                                    f"Expected type {expected_type}, got {type(parsed)} for: {expr_str}")

        except Exception as e:
            self.fail(f"DLSyntaxParser failed to parse expression: {expr_str}\nError: {e}")

    def test_valid_atomic_expression(self):
        """
        Test parsing of a valid atomic DL expression.
        """
        tokens = ['Person']
        expr_str, expr_dict = self.concept_abst_builder.parse(tokens)
        self.assertIn('Person', expr_str)
        self.assertEqual(expr_dict['type'], 'OWLClassExpression')
        self.assert_dl_parseable(expr_str, OWLClass)

    def test_valid_existential_expression(self):
        """
        Test parsing of a valid existential DL expression.
        """
        tokens = ['∃', 'hasSibling', '.', 'Person']
        expr_str, expr_dict = self.concept_abst_builder.parse(tokens)
        self.assertIn('∃hasSibling.Person', expr_str)
        self.assertEqual(expr_dict['type'], 'OWLClassExpression')
        self.assert_dl_parseable(expr_str, OWLObjectSomeValuesFrom)

    def test_valid_universal_expression(self):
        """
        Test parsing of a valid universal DL expression.
        """
        tokens = ['∀', 'married', '.', 'Thing']
        expr_str, expr_dict = self.concept_abst_builder.parse(tokens)
        self.assertIn('∀married.Thing', expr_str)
        self.assertEqual(expr_dict['type'], 'OWLClassExpression')
        self.assert_dl_parseable(expr_str, OWLObjectAllValuesFrom)

    def test_valid_conjuction_expression(self):
        """
        Test parsing of a valid disjunction DL expression.
        """
        tokens = ['(', 'Brother', '⊔', 'Sister', ')']
        expr_str, expr_dict = self.concept_abst_builder.parse(tokens)
        self.assertIn('Brother ⊔ Sister', expr_str)
        self.assertEqual(expr_dict['type'], 'OWLClassExpression')
        self.assert_dl_parseable(expr_str, OWLObjectUnionOf)

    def test_valid_disjunction_expression(self):
        """
        Test parsing of a valid conjunction DL expression.
        """
        tokens = ['(', 'Female', '⊓', 'Person', ')']
        expr_str, expr_dict = self.concept_abst_builder.parse(tokens)
        self.assertIn('Female ⊓ Person', expr_str)
        self.assertEqual(expr_dict['type'], 'OWLClassExpression')
        self.assert_dl_parseable(expr_str, OWLObjectIntersectionOf)

    def test_invalid_missing_role_enforced(self):
        """
        Test parsing with missing role and enforced validity.
        """
        tokens = ['∃', '.', 'Person']
        expr_str, expr_dict = self.concept_abst_builder.parse(tokens, enforce_validity=True)
        self.assertEqual(expr_dict['type'], 'OWLClassExpression')

    def test_balanced_parentheses(self):
        """
        Test balancing of parentheses in token sequences.
        """
        tokens = ['(', '(', 'Female', '⊓', 'Person', ')']
        balanced = self.concept_abst_builder.balance_flatten_parentheses(tokens, max_length=10)
        self.assertEqual(balanced.count('('), balanced.count(')'))

    def test_postprocess_tail(self):
        """
        Test postprocessing of incomplete token sequences.
        """
        tokens = ['∃', 'hasSibling']
        fixed = self.concept_abst_builder._postprocess_tail_fix(tokens.copy(), max_length=6)
        self.assertGreaterEqual(len(fixed), len(tokens))
        self.assertNotIn(fixed[-1], self.concept_abst_builder.quantifiers | self.concept_abst_builder.binary_ops)
        self.assert_dl_parseable(''.join(fixed), OWLObjectSomeValuesFrom)


    def test_grammar_strategy_after_quantifier(self):
        """
        Test grammar strategy lookahead after a quantifier token.
        """
        context = ['∃']
        next_tokens = self.concept_abst_builder._lookahead_grammar_strategy(context)
        self.assertTrue('hasSibling' in next_tokens)
        self.assertFalse('Person' in next_tokens)

    def test_enforce_on_malformed_tokens(self):
        """
        Test parsing and repair of malformed DL token sequences.
        """
        token_sequences = [
            ['∀', 'hasChild', '(', '¬', 'Father'],
            ['Person', '⊔', '(', '∃', 'hasSibling', '.', '(', '∃', '⊓', '(', ')', 'hasChild', ')', 'Father', ')', ')', ')'],
            ['Person', '⊓', '(', 'Mother', '⊔', '(', '∀', 'hasSibling', '.', '(', '¬', 'Mother', ')', ')', ')', '(', ')', ')', ')'],
            ['Person', '⊓', '(', '∀', 'married', '(', '∃', 'hasSibling', '.', 'Parent', ')', ')'],
            ['Person', '⊓', '(', '∃', 'married', '.', '(', '∀', 'hasChild', '.', '(', ')', ')', ')'],
            ['(', '⊓', '∀', 'hasSibling', '(', '(', ')', '.', 'Daughter', ')', ')', '⊔', '(', '∃', 'hasParent', '.', ')'],
            ['(',  '⊓',  '(', 'Brother',  '⊔',  '(', '(',  ')', ')', 'Sister', ')', ')'],
            ['∃', '⊔', '(', '∀', 'hasChild', '.', '(', 'Brother', ')'],
            ['Person', 'Mother', 'Thing'],
            ['(', '⊓', '∃', 'Person', ')'],
        ]

        for tokens in token_sequences:
            with self.subTest(tokens=tokens):
                self.concept_abst_builder.tokens = tokens
                repaired_expr_str, expr_dict = self.concept_abst_builder.parse(token_sequence=tokens, enforce_validity=True)
                self.assertEqual(expr_dict['type'], 'OWLClassExpression')
                self.assert_dl_parseable(repaired_expr_str, instances_check=True)
        
if __name__ == "__main__":
    unittest.main()


