import unittest
from ontolearn.learners import TDL
from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.learning_problem import PosNegLPStandard
from owlapy.iri import IRI
from owlapy.owl_individual import OWLNamedIndividual
from owlapy.converter import owl_expression_to_sparql
from ontolearn.utils.static_funcs import compute_f1_score, save_owl_class_expressions
import json
import rdflib


class TestConceptLearnerReg(unittest.TestCase):

    def test_regression_family(self):
        path = "KGs/Family/family-benchmark_rich_background.owl"
        kb = KnowledgeBase(path=path)
        with open("LPs/Family/lps.json") as json_file:
            settings = json.load(json_file)
        model = TDL(knowledge_base=kb, kwargs_classifier={"random_state": 1})
        for str_target_concept, examples in settings['problems'].items():
            p = set(examples['positive_examples'])
            n = set(examples['negative_examples'])
            typed_pos = set(map(OWLNamedIndividual, map(IRI.create, p)))
            typed_neg = set(map(OWLNamedIndividual, map(IRI.create, n)))
            lp = PosNegLPStandard(pos=typed_pos, neg=typed_neg)
            h = model.fit(learning_problem=lp).best_hypotheses()
            q = compute_f1_score(individuals=frozenset({i for i in kb.individuals(h)}), pos=lp.pos, neg=lp.neg)
            if str_target_concept == "Grandgrandmother":
                assert q >= 0.866
            elif str_target_concept == "Cousin":
                assert q >= 0.992
            else:
                assert q == 1.00
            # If not a valid SPARQL query, it should throw an error
            rdflib.Graph().query(owl_expression_to_sparql(root_variable="?x", expression=h))
            # Save the prediction
            save_owl_class_expressions(h)
            # (Load the prediction) and check the number of owl class definitions
            g = rdflib.Graph().parse("./Predictions.owl")
            # rdflib.Graph() parses named OWL Classes by the order of their definition
            named_owl_classes = [s for s, p, o in
                                 g.triples((None, rdflib.namespace.RDF.type, rdflib.namespace.OWL.Class)) if
                                 isinstance(s, rdflib.term.URIRef)]
            assert len(named_owl_classes) >= 1
            assert named_owl_classes.pop(0).n3() == "<https://dice-research.org/predictions#0>"

    def test_regression_mutagenesis(self):
        path = "KGs/Mutagenesis/mutagenesis.owl"
        # (1) Load a knowledge graph.
        kb = KnowledgeBase(path=path)
        with open("LPs/Mutagenesis/lps.json") as json_file:
            settings = json.load(json_file)
        model = TDL(knowledge_base=kb, report_classification=True, kwargs_classifier={"random_state": 1})
        for str_target_concept, examples in settings['problems'].items():
            p = set(examples['positive_examples'])
            n = set(examples['negative_examples'])
            typed_pos = set(map(OWLNamedIndividual, map(IRI.create, p)))
            typed_neg = set(map(OWLNamedIndividual, map(IRI.create, n)))
            lp = PosNegLPStandard(pos=typed_pos, neg=typed_neg)
            h = model.fit(learning_problem=lp).best_hypotheses()
            q = compute_f1_score(individuals=frozenset({i for i in kb.individuals(h)}), pos=lp.pos, neg=lp.neg)
            assert q >= 0.80

    def test_regression_carcinogenesis(self):
        path = "KGs/Carcinogenesis/carcinogenesis.owl"
        # (1) Load a knowledge graph.
        kb = KnowledgeBase(path=path)
        with open("LPs/Carcinogenesis/lps.json") as json_file:
            settings = json.load(json_file)
        model = TDL(knowledge_base=kb, report_classification=True, kwargs_classifier={"random_state": 1})
        for str_target_concept, examples in settings['problems'].items():
            p = set(examples['positive_examples'])
            n = set(examples['negative_examples'])
            typed_pos = set(map(OWLNamedIndividual, map(IRI.create, p)))
            typed_neg = set(map(OWLNamedIndividual, map(IRI.create, n)))
            lp = PosNegLPStandard(pos=typed_pos, neg=typed_neg)
            h = model.fit(learning_problem=lp).best_hypotheses()
            q = compute_f1_score(individuals=frozenset({i for i in kb.individuals(h)}), pos=lp.pos, neg=lp.neg)
            assert q >= 0.75

    def test_regression_family_triple_store(self):
        pass
        """
        # @TODO: CD: Removed because rdflib does not produce correct results
        path = "KGs/Family/family-benchmark_rich_background.owl"
        # (1) Load a knowledge graph.
        kb = TripleStore(path=path)
        with open("LPs/Family/lps.json") as json_file:
            settings = json.load(json_file)
        model = TDL(knowledge_base=kb, report_classification=False, kwargs_classifier={"random_state": 1})
        for str_target_concept, examples in settings['problems'].items():
            # CD: Other problems take too much time due to long SPARQL Query.
            if str_target_concept not in ["Brother", "Sister"
                                          "Daughter", "Son"
                                          "Father", "Mother",
                                          "Grandfather"]:
                continue
            p = set(examples['positive_examples'])
            n = set(examples['negative_examples'])
            typed_pos = set(map(OWLNamedIndividual, map(IRI.create, p)))
            typed_neg = set(map(OWLNamedIndividual, map(IRI.create, n)))
            lp = PosNegLPStandard(pos=typed_pos, neg=typed_neg)
            predicted_expression = model.fit(learning_problem=lp).best_hypotheses()
            predicted_expression = frozenset({i for i in kb.individuals(predicted_expression)})
            assert predicted_expression
            q = compute_f1_score(individuals=predicted_expression, pos=lp.pos, neg=lp.neg)
            assert q == 1.0
        """

    def test_regression_mutagenesis_triple_store(self):
        pass


class TestTDLConfigurationComparison(unittest.TestCase):
    """Test TDL performance with different configuration combinations.
    
    The hypothesis is that enabling all features (use_inverse, use_data_properties, 
    use_nominals, use_card_restrictions) should provide the best or comparable performance.
    """

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that are reused across tests."""
        cls.path_family = "KGs/Family/family-benchmark_rich_background.owl"
        cls.kb_family = KnowledgeBase(path=cls.path_family)
        
        with open("LPs/Family/lps.json") as json_file:
            cls.family_lps = json.load(json_file)

    def _evaluate_configuration(self, use_inverse, use_data_properties, use_nominals, 
                                use_card_restrictions, concept_name, max_examples=None):
        """Helper method to evaluate a TDL configuration on a specific concept.
        
        Args:
            use_inverse: Whether to use inverse properties
            use_data_properties: Whether to use data properties
            use_nominals: Whether to use nominals
            use_card_restrictions: Whether to use cardinality restrictions
            concept_name: Name of the concept to learn
            max_examples: Optional limit on number of examples (for faster testing)
            
        Returns:
            F1 score achieved by the configuration
        """
        model = TDL(
            knowledge_base=self.kb_family,
            use_inverse=use_inverse,
            use_data_properties=use_data_properties,
            use_nominals=use_nominals,
            use_card_restrictions=use_card_restrictions,
            verbose=0,
            report_classification=False,
            kwargs_classifier={"random_state": 42}
        )
        
        examples = self.family_lps['problems'][concept_name]
        p = set(examples['positive_examples'])
        n = set(examples['negative_examples'])
        
        if max_examples:
            p = set(list(p)[:max_examples])
            n = set(list(n)[:max_examples])
        
        typed_pos = set(map(OWLNamedIndividual, map(IRI.create, p)))
        typed_neg = set(map(OWLNamedIndividual, map(IRI.create, n)))
        lp = PosNegLPStandard(pos=typed_pos, neg=typed_neg)
        
        h = model.fit(learning_problem=lp).best_hypotheses()
        f1 = compute_f1_score(
            individuals=frozenset({i for i in self.kb_family.individuals(h)}),
            pos=lp.pos,
            neg=lp.neg
        )
        
        return f1

    def test_all_features_enabled_vs_default(self):
        """Test that enabling all features performs at least as well as default configuration."""
        concept = "Brother"
        
        # Default configuration (only use_nominals=True)
        f1_default = self._evaluate_configuration(
            use_inverse=False,
            use_data_properties=False,
            use_nominals=True,
            use_card_restrictions=False,
            concept_name=concept
        )
        
        # All features enabled
        f1_all_features = self._evaluate_configuration(
            use_inverse=True,
            use_data_properties=True,
            use_nominals=True,
            use_card_restrictions=True,
            concept_name=concept
        )
        
        # All features should perform at least as well (within a small tolerance)
        self.assertGreaterEqual(f1_all_features, f1_default - 0.05, 
                               f"All features F1={f1_all_features:.3f} should be >= default F1={f1_default:.3f}")
        
        # Both should achieve high performance on Brother
        self.assertGreaterEqual(f1_default, 0.95)
        self.assertGreaterEqual(f1_all_features, 0.95)

    def test_nominals_impact_on_performance(self):
        """Test the impact of nominals on learning performance."""
        concept = "Sister"
        
        # With nominals (default)
        f1_with_nominals = self._evaluate_configuration(
            use_inverse=False,
            use_data_properties=False,
            use_nominals=True,
            use_card_restrictions=False,
            concept_name=concept
        )
        
        # Without nominals
        f1_without_nominals = self._evaluate_configuration(
            use_inverse=False,
            use_data_properties=False,
            use_nominals=False,
            use_card_restrictions=False,
            concept_name=concept
        )
        
        # With nominals should generally perform better or equal
        # (nominals can provide very specific discriminative features)
        self.assertGreaterEqual(f1_with_nominals, f1_without_nominals - 0.1,
                               f"Nominals F1={f1_with_nominals:.3f} vs No nominals F1={f1_without_nominals:.3f}")

    def test_configuration_combinations_on_daughter(self):
        """Test various configuration combinations on Daughter concept."""
        concept = "Daughter"
        results = {}
        
        # Test different combinations
        configs = [
            ("default", False, False, True, False),
            ("all_enabled", True, True, True, True),
            ("no_nominals", False, False, False, False),
            ("with_cardinality", False, False, True, True),
        ]
        
        for name, use_inv, use_data, use_nom, use_card in configs:
            f1 = self._evaluate_configuration(
                use_inverse=use_inv,
                use_data_properties=use_data,
                use_nominals=use_nom,
                use_card_restrictions=use_card,
                concept_name=concept
            )
            results[name] = f1
        
        # All configurations should achieve reasonable performance
        for config_name, f1_score in results.items():
            self.assertGreaterEqual(f1_score, 0.85, 
                                   f"Config '{config_name}' should achieve F1 >= 0.85, got {f1_score:.3f}")
        
        # All features enabled should be among the best performers
        # (within 5% of the best configuration)
        best_f1 = max(results.values())
        self.assertGreaterEqual(results["all_enabled"], best_f1 - 0.05,
                               f"All enabled F1={results['all_enabled']:.3f} should be near best F1={best_f1:.3f}")

    def test_feature_richness_comparison(self):
        """Compare feature extraction richness across configurations."""
        concept = "Father"
        
        # Test how many features are extracted with different configurations
        examples = self.family_lps['problems'][concept]
        p = set(list(examples['positive_examples'])[:10])
        n = set(list(examples['negative_examples'])[:10])
        typed_pos = set(map(OWLNamedIndividual, map(IRI.create, p)))
        typed_neg = set(map(OWLNamedIndividual, map(IRI.create, n)))
        lp = PosNegLPStandard(pos=typed_pos, neg=typed_neg)
        
        # Minimal configuration
        model_minimal = TDL(
            knowledge_base=self.kb_family,
            use_inverse=False,
            use_data_properties=False,
            use_nominals=False,
            use_card_restrictions=False,
            verbose=0
        )
        model_minimal.fit(learning_problem=lp)
        features_minimal = len(model_minimal.features)
        
        # All features configuration
        model_all = TDL(
            knowledge_base=self.kb_family,
            use_inverse=True,
            use_data_properties=True,
            use_nominals=True,
            use_card_restrictions=True,
            verbose=0
        )
        model_all.fit(learning_problem=lp)
        features_all = len(model_all.features)
        
        # All features should extract at least as many features as minimal
        self.assertGreaterEqual(features_all, features_minimal,
                               f"All features config should extract >= features ({features_all} vs {features_minimal})")
        
        # Both should extract some features
        self.assertGreater(features_minimal, 0)
        self.assertGreater(features_all, 0)

    def test_consistency_across_problems(self):
        """Test that all features configuration maintains high performance across problems."""
        # Test on multiple concepts
        concepts = ["Brother", "Sister", "Father", "Mother"]
        
        for concept in concepts:
            f1 = self._evaluate_configuration(
                use_inverse=True,
                use_data_properties=True,
                use_nominals=True,
                use_card_restrictions=True,
                concept_name=concept
            )
            
            # Should achieve very high F1 on these straightforward concepts
            self.assertGreaterEqual(f1, 0.95, 
                                   f"All features should achieve F1 >= 0.95 on {concept}, got {f1:.3f}")

    def test_inverse_properties_benefit(self):
        """Test that inverse properties can improve learning on certain concepts."""
        # Concepts that might benefit from inverse properties
        concept = "Grandson"
        
        # Without inverse
        f1_no_inverse = self._evaluate_configuration(
            use_inverse=False,
            use_data_properties=False,
            use_nominals=True,
            use_card_restrictions=False,
            concept_name=concept
        )
        
        # With inverse
        f1_with_inverse = self._evaluate_configuration(
            use_inverse=True,
            use_data_properties=False,
            use_nominals=True,
            use_card_restrictions=False,
            concept_name=concept
        )
        
        # Both should achieve reasonable performance
        self.assertGreaterEqual(f1_no_inverse, 0.80)
        self.assertGreaterEqual(f1_with_inverse, 0.80)
        
        # With inverse should be competitive or better
        self.assertGreaterEqual(f1_with_inverse, f1_no_inverse - 0.1,
                               f"Inverse F1={f1_with_inverse:.3f} vs No inverse F1={f1_no_inverse:.3f}")


class TestTDLConfigurationMutagenesis(unittest.TestCase):
    """Test TDL configurations on Mutagenesis dataset (more complex)."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.path = "KGs/Mutagenesis/mutagenesis.owl"
        cls.kb = KnowledgeBase(path=cls.path)
        
        with open("LPs/Mutagenesis/lps.json") as json_file:
            cls.lps = json.load(json_file)

    def test_all_features_mutagenesis(self):
        """Test that all features configuration works well on Mutagenesis."""
        # Test on the first problem
        concept_name = list(self.lps['problems'].keys())[0]
        examples = self.lps['problems'][concept_name]
        
        p = set(examples['positive_examples'])
        n = set(examples['negative_examples'])
        typed_pos = set(map(OWLNamedIndividual, map(IRI.create, p)))
        typed_neg = set(map(OWLNamedIndividual, map(IRI.create, n)))
        lp = PosNegLPStandard(pos=typed_pos, neg=typed_neg)
        
        # All features enabled
        model_all = TDL(
            knowledge_base=self.kb,
            use_inverse=True,
            use_data_properties=True,
            use_nominals=True,
            use_card_restrictions=True,
            verbose=0,
            kwargs_classifier={"random_state": 42}
        )
        model_all.fit(learning_problem=lp)
        h_all = model_all.best_hypotheses()
        f1_all = compute_f1_score(
            individuals=frozenset({i for i in self.kb.individuals(h_all)}),
            pos=lp.pos,
            neg=lp.neg
        )
        
        # Default configuration
        model_default = TDL(
            knowledge_base=self.kb,
            use_inverse=False,
            use_data_properties=False,
            use_nominals=True,
            use_card_restrictions=False,
            verbose=0,
            kwargs_classifier={"random_state": 42}
        )
        model_default.fit(learning_problem=lp)
        h_default = model_default.best_hypotheses()
        f1_default = compute_f1_score(
            individuals=frozenset({i for i in self.kb.individuals(h_default)}),
            pos=lp.pos,
            neg=lp.neg
        )
        
        # Both should achieve reasonable performance on Mutagenesis
        self.assertGreaterEqual(f1_all, 0.75, f"All features F1={f1_all:.3f} should be >= 0.75")
        self.assertGreaterEqual(f1_default, 0.75, f"Default F1={f1_default:.3f} should be >= 0.75")
        
        # Note: More features doesn't always mean better performance
        # Sometimes simpler configurations (default) can outperform richer feature sets
        # due to reduced overfitting. Both configurations should achieve good results.
        # We verify that all features configuration is still competitive (within 20%)
        self.assertGreaterEqual(f1_all, f1_default - 0.20,
                               f"All features F1={f1_all:.3f} should be reasonably competitive with default F1={f1_default:.3f}")
