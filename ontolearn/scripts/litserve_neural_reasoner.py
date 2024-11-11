import argparse
import litserve as ls
from ontolearn.owl_neural_reasoner import TripleStoreNeuralReasoner
from owlapy import dl_to_owl_expression
from owlapy import owl_expression_to_dl

class NeuralReasonerAPI(ls.LitAPI):
	def __init__(self, path_neural_embedding, gamma=0.9):
		super().__init__()
		self.path_neural_embedding = path_neural_embedding
		self.gamma = gamma
	
	def setup(self, device):
		self.neural_owl_reasoner = TripleStoreNeuralReasoner(
			path_neural_embedding=self.path_neural_embedding, gamma=self.gamma
		)
		
	def decode_request(self, request):
		expression = request["expression"]  
		namespace = request["namespace"]    
		return expression, namespace
		
	def predict(self, data):
		expressions, namespace = data
		owl_expression = dl_to_owl_expression(
			namespace=namespace,
			dl_expression=expressions
		)  # Convert DL string to OWLClassExpression
		return set(self.neural_owl_reasoner.individuals(owl_expression))

	def encode_response(self, output):
		return {'retrieval_result': [owl_expression_to_dl(out) for out in output]}




if __name__ == "__main__":
	# Parse command-line arguments
	parser = argparse.ArgumentParser(description="Neural Reasoner API")
	parser.add_argument("--path_kge_model", type=str, default="KGs_Family_father_owl",
						help="Path to the neural embedding file")
	parser.add_argument("--workers", type=int, default=1,
						help="Number of workers per device")
	parser.add_argument("--path_kg", type=str, default="KGs/Family/father.owl",
						help="Path to the ontology file for namespace extraction")
	args = parser.parse_args()

	api = NeuralReasonerAPI(path_neural_embedding=args.path_kge_model)
	server = ls.LitServer(api, accelerator="auto", workers_per_device=args.workers, track_requests=True)
	server.run(port=8000)
