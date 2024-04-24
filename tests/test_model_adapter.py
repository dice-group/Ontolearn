import json
import unittest
from typing import cast

from ontolearn.concept_learner import CELOE, EvoLearner
from ontolearn.heuristics import CELOEHeuristic
from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.metrics import Accuracy
from ontolearn.model_adapter import ModelAdapter
from ontolearn.refinement_operators import ModifiedCELOERefinement
from owlapy.owl_individual import OWLNamedIndividual
from owlapy.iri import IRI
from ontolearn.base import OWLOntology_Owlready2, BaseReasoner_Owlready2
from ontolearn.base import OWLReasoner_Owlready2_ComplexCEInstances


class TestModelAdapter(unittest.TestCase):

    def test_celoe_quality_variant_1(self):
        with open('examples/synthetic_problems.json') as json_file:
            settings = json.load(json_file)
        kb_path = "KGs/Family/family-benchmark_rich_background.owl"
        kb = KnowledgeBase(path=kb_path)
        reasoner = OWLReasoner_Owlready2_ComplexCEInstances(cast(OWLOntology_Owlready2, kb.ontology),
                                                            BaseReasoner_Owlready2.HERMIT)
        op = ModifiedCELOERefinement(knowledge_base=kb, use_negation=False, use_all_constructor=False)
        p = set(settings['problems']['Uncle']['positive_examples'])
        n = set(settings['problems']['Uncle']['negative_examples'])
        typed_pos = set(map(OWLNamedIndividual, map(IRI.create, p)))
        typed_neg = set(map(OWLNamedIndividual, map(IRI.create, n)))

        model = ModelAdapter(learner_type=CELOE,
                             path=kb_path,
                             reasoner=reasoner,
                             quality_type=Accuracy,
                             max_runtime=5,
                             max_num_of_concepts_tested=10_000_000_000,
                             iter_bound=10_000_000_000,
                             refinement_operator=op)

        model = model.fit(pos=typed_pos, neg=typed_neg)
        hypothesis = model.best_hypotheses(n=1, return_node=True)
        assert hypothesis.quality >= 0.86

    def test_celoe_quality_variant_2(self):
        with open('examples/synthetic_problems.json') as json_file:
            settings = json.load(json_file)
        kb_path = "KGs/Family/family-benchmark_rich_background.owl"
        kb = KnowledgeBase(path=kb_path)
        reasoner = OWLReasoner_Owlready2_ComplexCEInstances(cast(OWLOntology_Owlready2, kb.ontology),
                                                            BaseReasoner_Owlready2.PELLET)
        op = ModifiedCELOERefinement(knowledge_base=kb, use_negation=False, use_all_constructor=False)
        p = set(settings['problems']['Uncle']['positive_examples'])
        n = set(settings['problems']['Uncle']['negative_examples'])
        typed_pos = set(map(OWLNamedIndividual, map(IRI.create, p)))
        typed_neg = set(map(OWLNamedIndividual, map(IRI.create, n)))

        model = ModelAdapter(learner_type=CELOE,
                             path=kb_path,
                             reasoner=reasoner,
                             quality_type=Accuracy,
                             max_runtime=5,
                             max_num_of_concepts_tested=10_000_000_000,
                             iter_bound=10_000_000_000,
                             refinement_operator=op,
                             heuristic_type=CELOEHeuristic,
                             expansionPenaltyFactor=0.05,
                             startNodeBonus=1.0,
                             nodeRefinementPenalty=0.01
                             )

        model = model.fit(pos=typed_pos, neg=typed_neg)
        hypothesis = model.best_hypotheses(n=1, return_node=True)
        assert hypothesis.quality >= 0.59

    def test_evolearner_quality(self):
        with open('examples/synthetic_problems.json') as json_file:
            settings = json.load(json_file)
        kb = KnowledgeBase(path="KGs/Family/family-benchmark_rich_background.owl")
        reasoner = OWLReasoner_Owlready2_ComplexCEInstances(cast(OWLOntology_Owlready2, kb.ontology),
                                                            BaseReasoner_Owlready2.HERMIT)
        p = set(settings['problems']['Uncle']['positive_examples'])
        n = set(settings['problems']['Uncle']['negative_examples'])
        typed_pos = set(map(OWLNamedIndividual, map(IRI.create, p)))
        typed_neg = set(map(OWLNamedIndividual, map(IRI.create, n)))

        model = ModelAdapter(learner_type=EvoLearner,
                             knowledge_base=kb,
                             reasoner=reasoner)

        model = model.fit(pos=typed_pos, neg=typed_neg)
        hypothesis = model.best_hypotheses(n=1,return_node=True)
        assert hypothesis.quality >= 0.9
