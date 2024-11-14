""" StratifiedKFold Cross Validating DL Concept Learning Algorithms

dicee --path_single_kg "KGs/Family/family-benchmark_rich_background.owl" --model Keci --path_to_store_single_run KeciFamilyRun --backend rdflib


python examples/concept_learning_neural_evaluation.py --lps LPs/Family/lps.json --kb KGs/Family/family-benchmark_rich_background.owl --kge KeciFamilyRun --max_runtime 3 --report family.csv


"""

import json
import time
import os
import subprocess
import platform
import pandas as pd
from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.learners import CELOE, OCEL, Drill, TDL
from ontolearn.concept_learner import EvoLearner, NCES, CLIP
from ontolearn.refinement_operators import ExpressRefinement
from ontolearn.learning_problem import PosNegLPStandard
from ontolearn.metrics import F1
from owlapy.owl_individual import OWLNamedIndividual, IRI
import argparse
from sklearn.model_selection import StratifiedKFold
import numpy as np
from ontolearn.utils.static_funcs import compute_f1_score
from ontolearn.triple_store import TripleStore
from ontolearn.owl_neural_reasoner import TripleStoreNeuralReasoner
from owlapy import owl_expression_to_dl

pd.set_option("display.precision", 5)

def dl_concept_learning(args):
    with open(args.lps) as json_file:
        settings = json.load(json_file)


    kb = KnowledgeBase(path=args.kb)
    drill_with_symbolic_retriever = Drill(knowledge_base=kb, path_embeddings=args.path_drill_embeddings,
                                          quality_func=F1(), max_runtime=args.max_runtime,verbose=0)

    neural_kb = TripleStore(reasoner=TripleStoreNeuralReasoner(path_neural_embedding=args.kge))

    drill_with_neural_retriever = Drill(knowledge_base=neural_kb,
                                        path_embeddings=args.path_drill_embeddings,
                                        quality_func=F1(), max_runtime=args.max_runtime, verbose=0)

    # dictionary to store the data
    data = dict()
    if "problems" in settings:
        problems = settings["problems"].items()
        positives_key = "positive_examples"
        negatives_key = "negative_examples"
    else:
        problems = settings.items()
        positives_key = "positive examples"
        negatives_key = "negative examples"

    for str_target_concept, examples in problems:
        print("Target concept: ", str_target_concept)
        p = examples[positives_key]
        n = examples[negatives_key]

        kf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.random_seed)
        X = np.array(p + n)
        y = np.array([1.0 for _ in p] + [0.0 for _ in n])

        for ith, (train_index, test_index) in enumerate(kf.split(X, y)):
            #
            data.setdefault("LP", []).append(str_target_concept)
            data.setdefault("Fold", []).append(ith)
            # () Extract positive and negative examples from train fold
            train_pos = {pos_individual for pos_individual in X[train_index][y[train_index] == 1]}
            train_neg = {neg_individual for neg_individual in X[train_index][y[train_index] == 0]}

            # Sanity checking for individuals used for training.
            assert train_pos.issubset(examples[positives_key])
            assert train_neg.issubset(examples[negatives_key])

            # () Extract positive and negative examples from test fold
            test_pos = {pos_individual for pos_individual in X[test_index][y[test_index] == 1]}
            test_neg = {neg_individual for neg_individual in X[test_index][y[test_index] == 0]}

            # Sanity checking for individuals used for testing.
            assert test_pos.issubset(examples[positives_key])
            assert test_neg.issubset(examples[negatives_key])
            train_lp = PosNegLPStandard(
                pos={OWLNamedIndividual(i) for i in train_pos},
                neg={OWLNamedIndividual(i) for i in train_neg})

            test_lp = PosNegLPStandard(
                pos={OWLNamedIndividual(i) for i in test_pos},
                neg={OWLNamedIndividual(i) for i in test_neg})
            print("DRILL Symbolic starts..", end=" ")
            start_time = time.time()
            # Prediction of DRILL through symbolic retriever.
            pred_symbolic_drill = drill_with_symbolic_retriever.fit(train_lp).best_hypotheses()
            symbolic_rt_drill = time.time() - start_time
            print("DRILL Symbolic ends..", end="\t")
            # Quality of prediction through symbolic retriever on the train split.
            symbolic_train_f1_drill = compute_f1_score(
                individuals=frozenset({i for i in kb.individuals(pred_symbolic_drill)}),
                pos=train_lp.pos,
                neg=train_lp.neg)
            # Quality of prediction through symbolic retriever on the test split.
            symbolic_test_f1_drill = compute_f1_score(
                individuals=frozenset({i for i in kb.individuals(pred_symbolic_drill)}),
                pos=test_lp.pos,
                neg=test_lp.neg)
            print(f"DRILL Symbolic Train Quality: {symbolic_train_f1_drill:.3f}", end="\t")
            print(f"DRILL Symbolic Test Quality: {symbolic_test_f1_drill:.3f}", end="\t")
            print(f"DRILL Symbolic Runtime: {symbolic_rt_drill:.3f}", end="\t")
            print(f"Prediction: {owl_expression_to_dl(pred_symbolic_drill)}")
            data.setdefault("Train-F1-Symbolic-DRILL", []).append(symbolic_train_f1_drill)
            data.setdefault("Test-F1-Symbolic-DRILL", []).append(symbolic_test_f1_drill)
            data.setdefault("RT-Symbolic-DRILL", []).append(symbolic_rt_drill)
            data.setdefault("Prediction-Symbolic-DRILL", []).append(owl_expression_to_dl(pred_symbolic_drill))

            print("DRILL Neural starts..", end=" ")
            start_time = time.time()
            # Prediction of DRILL through symbolic retriever.
            pred_neural_drill = drill_with_neural_retriever.fit(train_lp).best_hypotheses()
            neural_rt_drill = time.time() - start_time
            print("DRILL Neural ends..", end="\t")
            # Quality of prediction through symbolic retriever on the train split.
            neural_train_f1_drill = compute_f1_score(
                individuals=frozenset({i for i in neural_kb.individuals(pred_neural_drill)}),
                pos=train_lp.pos,
                neg=train_lp.neg)
            # Quality of prediction through symbolic retriever on the test split.
            neural_test_f1_drill = compute_f1_score(
                individuals=frozenset({i for i in neural_kb.individuals(pred_neural_drill)}),
                pos=test_lp.pos,
                neg=test_lp.neg)
            print(f"DRILL Neural Train Quality: {neural_train_f1_drill:.3f}", end="\t")
            print(f"DRILL Neural Test Quality: {neural_test_f1_drill:.3f}", end="\t")
            print(f"DRILL Neural Runtime: {neural_rt_drill:.3f}", end="\t")
            print(f"Prediction: {owl_expression_to_dl(pred_neural_drill)}")

            data.setdefault("Train-F1-Neural-DRILL", []).append(neural_train_f1_drill)
            data.setdefault("Test-F1-Neural-DRILL", []).append(neural_test_f1_drill)
            data.setdefault("RT-Neural-DRILL", []).append(neural_rt_drill)
            data.setdefault("Prediction-Neural-DRILL", []).append(owl_expression_to_dl(pred_neural_drill))

    print(data)
    df = pd.DataFrame.from_dict(data)
    df.to_csv(args.report, index=False)
    print(df)
    print(df.select_dtypes(include="number").mean())

    assert df.select_dtypes(include="number").mean()["Train-F1-Symbolic-DRILL"] >= 0.90



if __name__ == "__main__":
    # python examples/concept_learning_neural_evaluation.py --lps LPs/Family/lps.json --kb KGs/Family/family-benchmark_rich_background.owl --kge KeciFamilyRun --max_runtime 3 --report family.csv
    parser = argparse.ArgumentParser(description="OWL Class Expression Learning with Neural Reasoner")
    parser.add_argument("--lps", type=str, default="/home/cdemir/Desktop/Softwares/Ontolearn/LPs/Family/lps.json",
                        help="Path to the learning problems")
    parser.add_argument("--folds", type=int, default=2, help="Number of folds of cross validation.")
    parser.add_argument("--kb", type=str, default="/home/cdemir/Desktop/Softwares/Ontolearn/KGs/Family/family-benchmark_rich_background.owl", help="Knowledge base")
    parser.add_argument("--kge",type=str,default="/home/cdemir/Desktop/Softwares/Ontolearn/KeciFamilyRun",
                        help="Knowledge Graph Embedding Path")

    parser.add_argument("--path_drill_embeddings", type=str, default=None)
    parser.add_argument("--path_of_nces_embeddings", type=str, default=None)
    parser.add_argument("--path_of_clip_embeddings", type=str, default=None)
    parser.add_argument("--report", type=str, default="report.csv")
    parser.add_argument("--max_runtime", type=int, default=1, help="Max runtime")
    parser.add_argument("--random_seed", type=int, default=1)
    dl_concept_learning(parser.parse_args())
