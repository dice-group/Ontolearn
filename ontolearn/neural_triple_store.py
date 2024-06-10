from typing import Generator, Tuple, Union
from dicee.executer import Execute
from dicee.config import Namespace
from dicee.knowledge_graph_embeddings import KGE
from owlapy.model import IRI, OWLNamedIndividual, OWLObjectProperty, OWLClass
import rdflib
from .triple_store import TripleStoreReasonerOntology, TripleStore


class NeuralTripleStore(TripleStore):

    def __init__(self, path: str = None, model: str = "Keci"):
        super().__init__(path=path)
        self.path = path
        self.model = model

        args = Namespace()
        args.model = model
        args.p = 0
        args.q = 1
        args.optim = "adam"
        args.scoring_technique = "AllvsAll"
        args.path_single_kg = path
        args.backend = "rdflib"
        args.num_epochs = 200
        args.batch_size = 1024
        args.lr = 0.1
        args.embedding_dim = 512
        result = Execute(args).start()
        # loading pretrained model
        print("Result:")
        print(result)
        self.g = NeuralTripleStoreReasonerOntology(
            graph=rdflib.Graph().parse(path),
            model=KGE(path=result["path_experiment_folder"]),
        )

    def abox(
        self, individual: OWLNamedIndividual, mode: str = "native", k: int = 10
    ) -> Generator[
        Tuple[
            OWLNamedIndividual,
            Union[IRI, OWLObjectProperty],
            Union[OWLClass, OWLNamedIndividual],
        ],
        None,
        None,
    ]:
        for triple in super().abox(individual=individual, mode=mode):
            yield triple


class NeuralTripleStoreReasonerOntology(TripleStoreReasonerOntology):
    def __init__(self, graph: rdflib.graph.Graph, model: KGE):
        super().__init__(graph=graph)
        self.model = model

    def abox(self, str_iri: str, k: int = 10) -> Generator[
        Tuple[
            OWLNamedIndividual,
            Union[IRI, OWLObjectProperty],
            Union[OWLClass, OWLNamedIndividual],
        ],
        None,
        None,
    ]:
        """
        Get all axioms of a given individual being a subject entity using the trained knowledge graph embeddings model.

        Args:
            str_iri (str): An individual IRI
            k (int): Number of top predictions to return

        Returns:
            Iterable of tuples or owlapy axiom, depending on the configured mode.
        """
        print(
            self.model.find_missing_triples(
                confidence=0.5,
                entities=[str_iri],
            )
        )
        for triple in super().abox(str_iri=str_iri):
            print("Triple:", triple)
            yield triple
