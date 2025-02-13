# -----------------------------------------------------------------------------
# MIT License
#
# Copyright (c) 2024 Ontolearn Team
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# -----------------------------------------------------------------------------


import argparse
import glob
from fastapi import FastAPI
import uvicorn
from typing import Dict, Iterable, Union, List
from owlapy.class_expression import OWLClassExpression
from owlapy.iri import IRI
from owlapy.owl_individual import OWLNamedIndividual
from ontolearn.utils import compute_f1_score
from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.triple_store import TripleStore
from ontolearn.learning_problem import PosNegLPStandard
from ontolearn.learners import Drill, TDL
from ontolearn.concept_learner import NCES
from ontolearn.metrics import F1
from ontolearn.verbalizer import LLMVerbalizer
from owlapy import owl_expression_to_dl
import os

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


def get_drill(data: dict):
    """ Initialize DRILL """
    # (1) Init DRILL.
    global kb
    drill = Drill(knowledge_base=kb,
                  path_embeddings=data.get("path_embeddings", None),
                  quality_func=F1(),
                  iter_bound=data.get("iter_bound", 10),  # total refinement operation applied
                  max_runtime=data.get("max_runtime", 60),  # seconds
                  num_episode=data.get("num_episode", 2),  # for the training
                  use_inverse=data.get("use_inverse", True),
                  use_data_properties=data.get("use_data_properties", True),
                  use_card_restrictions=data.get("use_card_restrictions", True),
                  use_nominals=data.get("use_nominals", True),
                  verbose=1)
    # (2) Either load the weights of DRILL or train it.
    if data.get("path_to_pretrained_drill", None) and os.path.isdir(data["path_to_pretrained_drill"]):
        drill.load(directory=data["path_to_pretrained_drill"])
    else:
        # Train & Save
        drill.train(num_of_target_concepts=data.get("num_of_target_concepts", 1),
                    num_learning_problems=data.get("num_of_training_learning_problems", 1))
        drill.save(directory=data.get("path_to_pretrained_drill", None))
    return drill

def get_nces(data: dict) -> NCES:
    """ Load NCES """
    global kb
    global args
    assert args.path_knowledge_base.endswith(".owl"), "NCES supports only a knowledge base file with extension .owl"
    # (1) Init NCES.
    nces = NCES(knowledge_base_path=args.path_knowledge_base,
                    path_of_embeddings=data.get("path_embeddings", None),
                    quality_func=F1(),
                    load_pretrained=False,
                    learner_names=["SetTransformer", "LSTM", "GRU"],
                    num_predictions=64
                   )
    # (2) Either load the weights of NCES or train it.
    if data.get("path_to_pretrained_nces", None) and os.path.isdir(data["path_to_pretrained_nces"]) and glob.glob(data["path_to_pretrained_nces"]+"/*.pt"):
        nces.refresh(data["path_to_pretrained_nces"])
    else:
        nces.train(epochs=data["nces_train_epochs"], batch_size=data["nces_batch_size"], num_lps=data["num_of_training_learning_problems"])
        nces.refresh(nces.trained_models_path)
    return nces


def get_tdl(data) -> TDL:
    global kb
    return TDL(knowledge_base=kb,
               use_inverse=False,
               use_data_properties=False,
               use_nominals=False,
               use_card_restrictions=data.get("use_card_restrictions",False),
               kwargs_classifier=data.get("kwargs_classifier",None),
               verbose=10)


def get_learner(data: dict) -> Union[Drill, TDL, NCES, None]:
    if data["model"] == "Drill":
        return get_drill(data)
    elif data["model"] == "TDL":
        return get_tdl(data)
    elif data["model"] == "NCES":
        return get_nces(data)
    else:
        return None


@app.get("/cel")
async def cel(data: dict) -> Dict:
    global args
    global kb
    print("######### CEL Arguments ###############")
    print(f"Knowledgebase/Triplestore: {kb}\n")
    print(f"Input data: {data}\n")
    print("######### CEL Arguments ###############\n")
    # (1) Initialize OWL CEL and verbalizer
    owl_learner = get_learner(data)
    if owl_learner is None:
        return {"Results": f"There is no learner named as {data['model']}. Available models: Drill, TDL, NCES"}

    # (2) Read Positives and Negatives.
    positives = {OWLNamedIndividual(IRI.create(i)) for i in data['pos']}
    negatives = {OWLNamedIndividual(IRI.create(i)) for i in data['neg']}
    # (5)
    if len(positives) > 0 and len(negatives) > 0:
        # () LP
        lp = PosNegLPStandard(pos=positives, neg=negatives)
        # Few variable definitions for the sake of the readability.
        # ()Learning Process.
        results = []
        learned_owl_expression: OWLClassExpression
        predictions = owl_learner.fit(lp).best_hypotheses(n=data.get("topk", 3))
        if not isinstance(predictions, List):
            predictions = [predictions]
        verbalizer = LLMVerbalizer()
        for ith, learned_owl_expression in enumerate(predictions):
            # () OWL to DL
            dl_learned_owl_expression: str
            dl_learned_owl_expression = owl_expression_to_dl(learned_owl_expression)
            # () Get Individuals
            print(f"Retrieving individuals of {dl_learned_owl_expression}...")
            # TODO:CD: With owlapy:1.3.1, we can move the f1 score computation into triple store.
            # TODO: By this, we do not need to wait for the retrival results to return an answer to the user
            individuals: Iterable[OWLNamedIndividual]
            individuals = kb.individuals(learned_owl_expression)
            # () F1 score training
            train_f1: float
            train_f1 = compute_f1_score(individuals=frozenset({i for i in individuals}),
                                        pos=lp.pos,
                                        neg=lp.neg)
            results.append({"Rank": ith + 1,
                            "Prediction": dl_learned_owl_expression,
                            "Verbalization": verbalizer(dl_learned_owl_expression),
                            "F1": train_f1})

        return {"Results": results}
    else:
        return {"Results": "Error no valid learning problem"}


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
        raise RuntimeError("Either --path_knowledge_base or --endpoint_triplestore must be provided")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == '__main__':
    main()

