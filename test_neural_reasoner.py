from ontolearn.triple_store import TripleStore, TripleStoreReasonerOntology
from dicee import KGE
from owlapy.class_expression import OWLClass


class DemoNeuralReasoner:
    model: KGE

    def __init__(self, KGE_path: str):
        self.model = KGE(path=KGE_path)

    def get_type_individuals(self, individual: str, confidence_threshold: float = 0.9):
        predictions = self.model.predict_topk(
            h=[individual],
            r=["http://www.w3.org/1999/02/22-rdf-syntax-ns#type"],
            t=None,
            topk=10,
        )
        for prediction in predictions:
            print(prediction[1])
            yield OWLClass(prediction[0])


print("###########################")

kb_neural = TripleStore(DemoNeuralReasoner(KGE_path="./Pykeen_QuateEFatherRun"))
for t in kb_neural.g.get_type_individuals("http://example.com/father#anna"):
    print(t)
