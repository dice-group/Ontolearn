"""
Run Web Application
====================================================================

# Learn Embeddings
dicee --path_single_kg KGs/Family/family-benchmark_rich_background.owl --path_to_store_single_run embeddings --backend rdflib --save_embeddings_as_csv --model Keci --num_epoch 10

# Start Webservice
ontolearn --path_knowledge_base KGs/Family/family-benchmark_rich_background.owl

# Send HTTP Requests
curl -X 'GET' 'http://0.0.0.0:8000/cel'  -H 'accept: application/json' -H 'Content-Type: application/json' -d '{"pos":["http://www.benchmark.org/family#F2F14"], "neg":["http://www.benchmark.org/family#F10F200"], "model":"Drill","path_embeddings":"embeddings/Keci_entity_embeddings.csv"}'

curl -X 'GET' 'http://0.0.0.0:8000/cel'  -H 'accept: application/json' -H 'Content-Type: application/json' -d '{"pos":["http://www.benchmark.org/family#F2F14"], "neg":["http://www.benchmark.org/family#F10F200"], "model":"Drill","pretrained":"pretrained","path_embeddings":"embeddings/Keci_entity_embeddings.csv"}'


====================================================================
"""
import json
import argparse
from fastapi import FastAPI
import uvicorn
import logging
import requests

from ..utils.static_funcs import compute_f1_score
from ..knowledge_base import KnowledgeBase
from ..learning_problem import PosNegLPStandard
from ..refinement_operators import LengthBasedRefinement
from ..learners import Drill
from ..metrics import F1
from owlapy.model import OWLNamedIndividual, IRI
from owlapy.render import DLSyntaxObjectRenderer
from ..utils.static_funcs import save_owl_class_expressions

app = FastAPI()
args = None
kb = None


def get_default_arguments():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--path_knowledge_base", type=str, required=True)
    return parser.parse_args()


@app.get("/")
async def root():
    global args
    return {"response": "Ontolearn Service is Running"}


def get_drill(data: dict):
    # (2) Init DRILL.
    global kb
    drill = Drill(knowledge_base=kb,
                  path_embeddings=data.get("path_embeddings", None),
                  refinement_operator=LengthBasedRefinement(knowledge_base=kb),
                  quality_func=F1(),
                  num_of_sequential_actions=data.get("num_of_sequential_actions", 2),
                  iter_bound=data.get("iter_bound", 100),
                  max_runtime=data.get("max_runtime", 3))
    # (3) Load weights or train DRILL.
    if data.get("pretrained", None):
        drill.load(directory=data["pretrained"])
    else:
        # Train & Save
        drill.train(num_of_target_concepts=data.get("num_of_target_concepts", 1),
                    num_learning_problems=data.get("num_of_training_learning_problems", 3))
        drill.save(directory="pretrained")
    return drill


@app.get("/cel")
async def owl_class_expression_learning(data: dict):
    global args
    global kb
    if data["model"] == "Drill":
        owl_learner = get_drill(data)
    else:
        raise NotImplementedError()
    # (4) Read Positives and Negatives.
    positives = {OWLNamedIndividual(IRI.create(i)) for i in data['pos']}
    negatives = {OWLNamedIndividual(IRI.create(i)) for i in data['neg']}

    if len(positives) > 0 and len(negatives) > 0:
        dl_render = DLSyntaxObjectRenderer()
        lp = PosNegLPStandard(pos=positives,
                              neg=negatives)
        prediction = owl_learner.fit(lp).best_hypotheses()
        train_f1 = compute_f1_score(individuals=frozenset({i for i in kb.individuals(prediction)}),
                                    pos=lp.pos,
                                    neg=lp.neg)
        save_owl_class_expressions(expressions=prediction, path="Predictions")
        return {"Prediction": dl_render.render(prediction), "F1": train_f1, "saved_prediction": "Predictions.owl"}
    else:
        return {"Prediction": "No", "F1": 0.0}


def main():
    global args
    args = get_default_arguments()
    global kb
    # (1) Init knowledge base.
    kb = KnowledgeBase(path=args.path_knowledge_base)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == '__main__':
    main()
