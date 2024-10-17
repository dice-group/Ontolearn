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
from ..utils.static_funcs import compute_f1_score
from ..knowledge_base import KnowledgeBase
from ..triple_store import TripleStore
from ..learning_problem import PosNegLPStandard
from ..refinement_operators import LengthBasedRefinement
from ..learners import Drill, TDL
from ..concept_learner import NCES
from ..metrics import F1
from owlapy.render import DLSyntaxObjectRenderer
from ..utils.static_funcs import save_owl_class_expressions
from owlapy import owl_expression_to_dl
import os
from ..verbalizer import LLMVerbalizer
import platform, subprocess

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


def get_embedding_path(ftp_link: str, path_embeddings: str, kb_path_arg: str)->str:
    """
    ftp_link: ftp link to download data
    kb_path_arg:local path of an RDF KG
    """

    def get_kb_name(kb_path_arg):
        if "family" in kb_path_arg:
            return "family"
        elif "carcinogenesis" in kb_path_arg:
            return "carcinogenesis"
        elif "mutagenesis" in kb_path_arg:
            return "mutagenesis"
        elif "nctrer" in kb_path_arg:
            return "nctrer"
        elif "animals" in kb_path_arg:
            return "animals"
        elif "lymphography" in kb_path_arg:
            return "lymphography"
        elif "semantic_bible" in kb_path_arg:
            return "semantic_bible"
        elif "suramin" in kb_path_arg:
            return "suramin"
        elif "vicodi" in kb_path_arg:
            return "vicodi"
        else:
            raise ValueError("Knowledge base name is not recognized")
    kb_name = get_kb_name(kb_path_arg)
    if path_embeddings and os.path.exists(path_embeddings) and path_embeddings.endswith(".csv"):
        return path_embeddings
    elif path_embeddings and os.path.exists(path_embeddings) and not os.path.isfile(path_embeddings) and glob.glob(path_embeddings+"/*.csv") and [f for f in glob.glob(path_embeddings+"/*.csv") if kb_name in f]:
        correct_files = [f for f in glob.glob(path_embeddings+"/*.csv") if kb_name in f]
        return correct_files[0]
    elif not os.path.exists(f"./NCESData/{kb_name}/embeddings/ConEx_entity_embeddings.csv"):
        file_name = ftp_link.split("/")[-1]
        
        if not os.path.exists(os.path.join(os.getcwd(), file_name)):
            subprocess.run(['curl', '-O', ftp_link])

            if platform.system() == "Windows":
                subprocess.run(['tar', '-xf', file_name])
            else:
                subprocess.run(['unzip', file_name])
            os.remove(os.path.join(os.getcwd(), file_name))

        embeddings_path = os.path.join(os.getcwd(), file_name[:-4] + '/')

        embeddings_path += f"{kb_name}/embeddings/ConEx_entity_embeddings.csv"

        return embeddings_path
    else:
        return f"./NCESData/{kb_name}/embeddings/ConEx_entity_embeddings.csv"


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
                  use_inverse=True,
                  use_data_properties=True,
                  use_card_restrictions=True,
                  use_nominals=True,
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

def get_nces(data: dict):
    """ Load NCES """
    global kb
    global args
    assert args.path_knowledge_base.endswith(".owl"), "NCES supports only a knowledge base file with extension .owl"
    # (1) Init NCES.
    nces = NCES(knowledge_base_path=args.path_knowledge_base,
                    path_of_embeddings=get_embedding_path("https://files.dice-research.org/projects/NCES/NCES_Ontolearn_Data/NCESData.zip", data.get("path_embeddings", None), args.path_knowledge_base),
                    quality_func=F1(),
                    load_pretrained=False,
                    learner_names=["SetTransformer", "LSTM", "GRU"],
                    num_predictions=64
                   )
    # (2) Either load the weights of NCES or train it.
    if data.get("path_to_pretrained_nces", None) and os.path.isdir(data["path_to_pretrained_nces"]) and glob.glob("*.pt"):
        nces.refresh(data["path_to_pretrained_nces"])
    else:
        nces.train(epochs=data["nces_train_epochs"], batch_size=data["nces_batch_size"], num_lps=data["num_of_training_learning_problems"])
        nces.refresh(nces.trained_models_path)
    return nces


def get_tdl(data) -> TDL:
    global kb
    return TDL(knowledge_base=kb)


def get_learner(data: dict) -> Union[Drill, TDL, NCES]:
    if data["model"] == "Drill":
        return get_drill(data)
    elif data["model"] == "TDL":
        return get_tdl(data)
    elif data["model"] == "NCES":
        return get_nces(data)
    else:
        raise NotImplementedError(f"There is no learner {data['model']} available")


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
    verbalizer = LLMVerbalizer()
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

        for ith, learned_owl_expression in enumerate(predictions):
            # () OWL to DL
            dl_learned_owl_expression: str
            dl_learned_owl_expression = owl_expression_to_dl(learned_owl_expression)
            # () Get Individuals
            print(f"Retrieving individuals of {dl_learned_owl_expression}...")
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

