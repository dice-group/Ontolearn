"""
Run Web Application
====================================================================

dicee --path_single_kg KGs/Family/family-benchmark_rich_background.owl --path_to_store_single_run embeddings --backend rdflib --save_embeddings_as_csv --model Keci --num_epoch 10

# Start Webservice
ontolearn-webservice --path_knowledge_base KGs/Family/family-benchmark_rich_background.owl

# Send HTTP Get Request to train DRILL and evaluate it on provided pos and neg
curl -X 'GET' 'http://0.0.0.0:8000/cel'  -H 'accept: application/json' -H 'Content-Type: application/json' -d '{"pos":["http://www.benchmark.org/family#F2F14"], "neg":["http://www.benchmark.org/family#F10F200"], "model":"Drill","path_embeddings":"embeddings/Keci_entity_embeddings.csv"}'

# Send HTTP Get Request to load a pretrained DRILL and evaluate it on provided pos and neg
curl -X 'GET' 'http://0.0.0.0:8000/cel'  -H 'accept: application/json' -H 'Content-Type: application/json' -d '{"pos":["http://www.benchmark.org/family#F2F14"], "neg":["http://www.benchmark.org/family#F10F200"], "model":"Drill","pretrained":"pretrained","path_embeddings":"embeddings/Keci_entity_embeddings.csv"}'


====================================================================
"""
import argparse
from fastapi import FastAPI
import uvicorn
from typing import Dict, Iterable, Union
from owlapy.class_expression import OWLClassExpression
from owlapy.iri import IRI
from owlapy.owl_individual import OWLNamedIndividual
from ..utils.static_funcs import compute_f1_score
from ..knowledge_base import KnowledgeBase
from ..triple_store import TripleStore
from ..learning_problem import PosNegLPStandard
from ..refinement_operators import LengthBasedRefinement
from ..learners import Drill, TDL
from ..metrics import F1
from owlapy.render import DLSyntaxObjectRenderer
from ..utils.static_funcs import save_owl_class_expressions

app = FastAPI()
args = None
# Knowledge Base Loaded once
kb = None


def get_default_arguments():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--path_knowledge_base", type=str, default=None)
    parser.add_argument("--endpoint_triple_store", type=str, default=None)
    return parser.parse_args()


@app.get("/")
async def root():
    global args
    return {"response": "Ontolearn Service is Running"}


def get_drill(data: dict) -> Drill:
    """ Initialize DRILL """
    # (1) Init DRILL.
    global kb
    drill = Drill(knowledge_base=kb,
                  path_embeddings=data.get("path_embeddings", None),
                  quality_func=F1(),
                  iter_bound=data.get("iter_bound", 10),  # total refinement operation applied
                  max_runtime=data.get("max_runtime", 60),  # seconds
                  verbose=1)
    # (2) Either load the weights of DRILL or train it.
    if data.get("pretrained", None):
        drill.load(directory=data["pretrained"])
    else:
        # Train & Save
        drill.train(num_of_target_concepts=data.get("num_of_target_concepts", 1),
                    num_learning_problems=data.get("num_of_training_learning_problems", 1))
        drill.save(directory="pretrained")
    return drill


def get_tdl(data):
    global kb
    return TDL(knowledge_base=kb)


def get_learner(data: dict) -> Union[Drill, TDL]:
    if data["model"] == "Drill":
        return get_drill(data)
    elif data["model"] == "TDL":
        return get_tdl(data)
    else:
        raise NotImplementedError(f"There is no learner {data['model']} available")


@app.get("/cel")
async def cel(data: dict) -> Dict:
    global args
    global kb
    print("Initialized:", kb)
    print(args)
    # (1) Initialize OWL CEL
    owl_learner = get_learner(data)
    # (2) Read Positives and Negatives.
    positives = {OWLNamedIndividual(IRI.create(i)) for i in data['pos']}
    negatives = {OWLNamedIndividual(IRI.create(i)) for i in data['neg']}
    # (5)
    if len(positives) > 0 and len(negatives) > 0:
        dl_render = DLSyntaxObjectRenderer()
        lp = PosNegLPStandard(pos=positives, neg=negatives)
        # Few variable definitions for the sake of the readability.
        learned_owl_expression: OWLClassExpression
        dl_learned_owl_expression: str
        individuals: Iterable[OWLNamedIndividual]
        train_f1: float
        # Learning Process.
        learned_owl_expression = owl_learner.fit(lp).best_hypotheses()
        dl_learned_owl_expression = dl_render.render(learned_owl_expression)
        if data.get("compute_quality", None):
            # Concept Retrieval.
            individuals = kb.individuals(learned_owl_expression)
            train_f1 = compute_f1_score(individuals=frozenset({i for i in individuals}),
                                        pos=lp.pos,
                                        neg=lp.neg)
            save_owl_class_expressions(expressions=learned_owl_expression, path="Predictions")
            return {"Prediction": dl_learned_owl_expression, "F1": train_f1, "saved_prediction": "Predictions.owl"}
        else:
            return {"Prediction": dl_learned_owl_expression}

    else:
        return {"Prediction": "No Learning Problem Given!!!", "F1": 0.0}


def main():
    global args
    global kb
    args = get_default_arguments()
    # (1) Init knowledge base.
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_knowledge_base", type=str, default=None)
    parser.add_argument("--endpoint_triple_store", type=str, default=None)
    if args.path_knowledge_base:
        kb = KnowledgeBase(path=args.path_knowledge_base)
    elif args.endpoint_triple_store:
        kb = TripleStore(url=args.endpoint_triple_store)
    else:
        raise RuntimeError("Either --path_knowledge_base or --endpoint_triplestore must be not None")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == '__main__':
    main()
