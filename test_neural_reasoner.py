from ontolearn.triple_store import TripleStore, TripleStoreReasonerOntology


class DemoNeuralReasoner:
    def __init__(self, KGE_path: str):
        pass

    def get_type_individuals(self, individual: str):
        return set()


kb_classic = TripleStore(url="http://localhost:3030/father")


for t in kb_classic.g.get_type_individuals("http://example.com/father#anna"):
    print(t)


print("###########################")

kb_neural = TripleStore(DemoNeuralReasoner())
for t in kb_neural.g.get_type_individuals("http://example.com/father#anna"):
    print(t)
