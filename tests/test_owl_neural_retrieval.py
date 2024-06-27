from ontolearn.triple_store import NeuralReasoner
from owlapy.owl_property import OWLObjectProperty, OWLProperty
from owlapy.class_expression import OWLClass
from owlapy.owl_individual import OWLNamedIndividual
from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.utils import jaccard_similarity


class Test_Neural_Retrieval:

    def test_retrieval_single_individual_father_owl(self):
        neural_owl_reasoner = NeuralReasoner(path_of_kb="KGs/Family/father.owl", gamma=0.8)
        triples_about_anna = {(s, p, o) for s, p, o in neural_owl_reasoner.abox("http://example.com/father#anna")}

        sanity_checking = {(OWLNamedIndividual('http://example.com/father#anna'),
                            OWLObjectProperty('http://example.com/father#hasChild'),
                            OWLNamedIndividual('http://example.com/father#heinz')),
                           (OWLNamedIndividual('http://example.com/father#anna'),
                            OWLProperty('http://www.w3.org/1999/02/22-rdf-syntax-ns#type'),
                            OWLClass('http://example.com/father#female')
                            ),
                           (OWLNamedIndividual('http://example.com/father#anna'),
                            OWLProperty('http://www.w3.org/1999/02/22-rdf-syntax-ns#type'),
                            OWLClass('http://www.w3.org/2002/07/owl#NamedIndividual')
                            )}

        assert triples_about_anna == sanity_checking

    def test_regression_named_concepts_owl(self):
        symbolic_kb = KnowledgeBase(path="KGs/Family/father.owl")
        benchmark_dataset = [(concept, {individual.str for individual in symbolic_kb.individuals(concept)}) for concept
                             in
                             symbolic_kb.get_concepts()]
        
        neural_owl_reasoner = NeuralReasoner(path_of_kb="KGs/Family/father.owl", gamma=0.8)

        avg_jaccard_index = 0
        for (concept, symbolic_retrieval) in benchmark_dataset:
            neural_retrieval = {i.str for i in neural_owl_reasoner.instances(concept)}
            v = jaccard_similarity(symbolic_retrieval, neural_retrieval)
            avg_jaccard_index += v
        assert avg_jaccard_index / len(benchmark_dataset) >= 0.66