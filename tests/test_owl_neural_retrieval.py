
from ontolearn.triple_store import TripleStoreNeuralReasoner
from owlapy.owl_property import OWLObjectProperty,OWLProperty
from owlapy.class_expression import OWLClass
from owlapy.owl_individual import OWLNamedIndividual
class Test_Neural_Retrieval:

    def test_retrieval_single_individual_father_owl(self):
        neural_owl_reasoner = TripleStoreNeuralReasoner(path_of_kb="KGs/Family/father.owl", gamma=0.8)
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
