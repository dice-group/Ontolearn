from ontolearn.triple_store import TripleStore, TripleStoreReasonerOntology
from dicee import KGE
from owlapy.class_expression import OWLClass, OWLClassExpression
from owlapy.owl_individual import OWLNamedIndividual
from owlapy.owl_property import OWLDataProperty
from typing import Generator


class DemoNeuralReasoner:
    model: KGE

    def __init__(self, KGE_path: str):
        self.model = KGE(path=KGE_path)

    def get_type_individuals(
        self, individual: str, confidence_threshold: float = 0
    ) -> Generator[OWLClass, None, None]:
        predictions = self.model.predict_topk(
            h=[individual],
            r=["http://www.w3.org/1999/02/22-rdf-syntax-ns#type"],
            t=None,
            topk=10,
        )
        for prediction in predictions:
            confidence = prediction[1]
            predicted_iri_str = prediction[0]
            try:
                owl_class = OWLClass(prediction[0])
                if confidence >= confidence_threshold:
                    yield owl_class
            except Exception as e:
                # Log the invalid IRI
                print(f"Invalid IRI detected: {predicted_iri_str}, error: {e}")
                continue

    def instances(
        self, owl_class: OWLClassExpression, confidence_threshold: float = 0
    ) -> Generator[OWLNamedIndividual, None, None]:
        predictions = self.model.predict_topk(
            h=None,
            r=["http://www.w3.org/1999/02/22-rdf-syntax-ns#type"],
            t=[owl_class.str],
            topk=10,
        )
        for prediction in predictions:
            confidence = prediction[1]
            predicted_iri_str = prediction[0]
            try:
                owl_named_individual = OWLNamedIndividual(predicted_iri_str)
                if confidence >= confidence_threshold:
                    print("confidence: ", confidence)
                    yield owl_named_individual
            except Exception as e:
                # Log the invalid IRI
                print(f"Invalid IRI detected: {predicted_iri_str}, error: {e}")
                continue

    def individuals_in_signature(self) -> Generator[OWLNamedIndividual, None, None]:
        pass

    def data_properties_in_signature(
        self, confidence_threshold: float = 0
    ) -> Generator[
        OWLDataProperty, None, None
    ]:  # Why is this in triple_store.py a Iterable and not a Generator?
        predictions = self.model.predict_topk(
            h=None,
            r=["http://www.w3.org/1999/02/22-rdf-syntax-ns#type"],
            t=["http://www.w3.org/2002/07/owl#DatatypeProperty"],
            topk=10,
        )
        for prediction in predictions:
            predicted_iri_str = prediction[0]
            try:
                owl_data_property = OWLDataProperty(predicted_iri_str)

                yield owl_data_property
            except Exception as e:
                # Log the invalid IRI
                print(f"Invalid IRI detected: {predicted_iri_str}, error: {e}")
                continue


###testing
print("------------------- Classic Reasoner -------------------")

kb_classic = TripleStore(url="http://localhost:3030/father")

for t in kb_classic.g.get_type_individuals("http://example.com/father#anna"):
    print(t)

print("------------------- instances function -------------------")
for i in kb_classic.g.instances(OWLClass("http://example.com/father#female")):
    print(i)


print("+++++++++++++++++++ Neural Reasoner +++++++++++++++++++")

kb_neural = TripleStore(DemoNeuralReasoner(KGE_path="./Pykeen_QuateEFatherRun"))
for t in kb_neural.g.get_type_individuals("http://example.com/father#anna"):
    print(t)

print("+++++++++++++++++++ instances function +++++++++++++++++++")
for i in kb_neural.g.instances(OWLClass("http://example.com/father#person")):
    print(i)

print("+++++++++++++++++++ data_properties_in_signature function +++++++++++++++++++")
for dp in kb_neural.g.data_properties_in_signature():
    print(dp)
