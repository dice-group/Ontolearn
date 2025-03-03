""" StratifiedKFold Cross Validating DL Concept Learning Algorithms
python examples/concept_learning_cv_evaluation.py --lps LPs/Family/lps_difficult.json --kb KGs/Family/family.owl --max_runtime 60 --report family.csv --path_of_nces_embeddings ./NCESData/family/embeddings/DeCaL_entity_embeddings.csv --path_of_nces_trained_models ./NCESData/family/trained_models/ --path_of_nces2_trained_models ./NCES2Data/family/trained_models/ --path_of_roces_trained_models ./ROCESData/family/trained_models/ --path_of_clip_embeddings ./CLIPData/family/embeddings/ConEx_entity_embeddings.csv

python examples/concept_learning_cv_evaluation.py --lps LPs/Carcinogenesis/lps.json --kb KGs/Carcinogenesis/carcinogenesis.owl --max_runtime 60 --report carcinogenesis.csv --path_of_nces_embeddings ./NCESData/carcinogenesis/embeddings/DeCaL_entity_embeddings.csv --path_of_nces_trained_models ./NCESData/carcinogenesis/trained_models/ --path_of_nces2_trained_models ./NCES2Data/carcinogenesis/trained_models/ --path_of_roces_trained_models ./ROCESData/carcinogenesis/trained_models/ --path_of_clip_embeddings ./CLIPData/carcinogenesis/embeddings/ConEx_entity_embeddings.csv

python examples/concept_learning_cv_evaluation.py --lps LPs/Mutagenesis/lps.json --kb KGs/Mutagenesis/mutagenesis.owl --max_runtime 60 --report mutagenesis.csv --path_of_nces_embeddings ./NCESData/mutagenesis/embeddings/DeCaL_entity_embeddings.csv --path_of_nces_trained_models ./NCESData/mutagenesis/trained_models/ --path_of_nces2_trained_models ./NCES2Data/mutagenesis/trained_models/ --path_of_roces_trained_models ./ROCESData/mutagenesis/trained_models/ --path_of_clip_embeddings ./CLIPData/mutagenesis/embeddings/ConEx_entity_embeddings.csv
"""
import json
import time
import os
import pandas as pd
from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.concept_learner import CELOE, EvoLearner, NCES, NCES2, ROCES, CLIP
from ontolearn.refinement_operators import ExpressRefinement, ModifiedCELOERefinement
from ontolearn.learners import Drill, TDL, OCEL
from ontolearn.learning_problem import PosNegLPStandard
from ontolearn.metrics import F1
from owlapy.owl_individual import OWLNamedIndividual, IRI
import argparse
from sklearn.model_selection import StratifiedKFold
import numpy as np

from ontolearn.utils.static_funcs import compute_f1_score

pd.set_option("display.precision", 5)

def dl_concept_learning(args):
    with open(args.lps) as json_file:
        settings = json.load(json_file)
    kb = KnowledgeBase(path=args.kb)

    ocel = OCEL(knowledge_base=kb,
                quality_func=F1(),
                max_runtime=args.max_runtime)

    celoe = CELOE(knowledge_base=kb,
                  quality_func=F1(),
                  max_runtime=args.max_runtime)
    drill = Drill(knowledge_base=kb,
                  path_embeddings=args.path_drill_embeddings,
                  quality_func=F1(),
                  max_runtime=args.max_runtime, verbose=0)
    tdl = TDL(knowledge_base=kb,
              kwargs_classifier={"random_state": 1},
              max_runtime=args.max_runtime,
              verbose=0)
    
    nces = NCES(knowledge_base_path=args.kb,
                quality_func=F1(),
                load_pretrained=True,
                path_of_embeddings=args.path_of_nces_embeddings,
                path_of_trained_models=args.path_of_nces_trained_models,
                learner_names=["LSTM", "GRU", "SetTransformer"],
                num_predictions=200,
                verbose=0)

    nces2 = NCES2(knowledge_base_path=args.kb,
                  quality_func=F1(),
                  load_pretrained=True,
                  path_of_trained_models=args.path_of_nces2_trained_models,
                  num_predictions=200,
                  verbose=0)

    roces = ROCES(knowledge_base_path=args.kb,
                  k=50,
                  quality_func=F1(),
                  load_pretrained=True,
                  path_of_trained_models=args.path_of_roces_trained_models,
                  num_predictions=200,
                  verbose=0)
    
    clip = CLIP(knowledge_base=kb,
                refinement_operator=ModifiedCELOERefinement(kb),
                quality_func=F1(),
                max_num_of_concepts_tested=int(1e9), max_runtime=args.max_runtime,
                path_of_embeddings=args.path_of_clip_embeddings,
                pretrained_predictor_name=["LSTM", "GRU", "SetTransformer"], load_pretrained=True)

    # dictionary to store the data
    data = dict()
    if "problems" in settings:
        problems = settings['problems'].items()
        positives_key = "positive_examples"
        negatives_key = "negative_examples"
    else:
        problems = settings.items()
        positives_key = "positive examples"
        negatives_key = "negative examples"
    for str_target_concept, examples in problems:
        print('Target concept: ', str_target_concept)
        p = examples[positives_key]
        n = examples[negatives_key]

        kf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.random_seed)
        X = np.array(p + n)
        y = np.array([1.0 for _ in p] + [0.0 for _ in n])

        for (ith, (train_index, test_index)) in enumerate(kf.split(X, y)):
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
            train_lp = PosNegLPStandard(pos={OWLNamedIndividual(i) for i in train_pos},
                                        neg={OWLNamedIndividual(i) for i in train_neg})

            test_lp = PosNegLPStandard(pos={OWLNamedIndividual(i) for i in test_pos},
            
                                       neg={OWLNamedIndividual(i) for i in test_neg})
            
            print("OCEL starts..", end="\t")
            start_time = time.time()
            pred_ocel = ocel.fit(train_lp).best_hypotheses()
            rt_ocel = time.time() - start_time
            print("OCEL ends..", end="\t")
            # () Quality on the training data
            train_f1_ocel = compute_f1_score(individuals=frozenset({i for i in kb.individuals(pred_ocel)}),
                                             pos=train_lp.pos,
                                             neg=train_lp.neg)
            # () Quality on test data
            test_f1_ocel = compute_f1_score(individuals=frozenset({i for i in kb.individuals(pred_ocel)}),
                                            pos=test_lp.pos,
                                            neg=test_lp.neg)
            # Reporting
            data.setdefault("Train-F1-OCEL", []).append(train_f1_ocel)
            data.setdefault("Test-F1-OCEL", []).append(test_f1_ocel)
            data.setdefault("RT-OCEL", []).append(rt_ocel)
            print(f"OCEL Train Quality: {train_f1_ocel:.3f}", end="\t")
            print(f"OCEL Test Quality: {test_f1_ocel:.3f}", end="\t")
            print(f"OCEL Runtime: {rt_ocel:.3f}")


            print("CELOE starts..", end="\t")
            start_time = time.time()
            pred_celoe = celoe.fit(train_lp).best_hypotheses()
            rt_celoe = time.time() - start_time
            print("CELOE ends..", end="\t")
            # () Quality on the training data
            train_f1_celoe = compute_f1_score(individuals=frozenset({i for i in kb.individuals(pred_celoe)}),
                                              pos=train_lp.pos,
                                              neg=train_lp.neg)
            # () Quality on test data
            test_f1_celoe = compute_f1_score(individuals=frozenset({i for i in kb.individuals(pred_celoe)}),
                                             pos=test_lp.pos,
                                             neg=test_lp.neg)
            # Reporting
            data.setdefault("Train-F1-CELOE", []).append(train_f1_celoe)
            data.setdefault("Test-F1-CELOE", []).append(test_f1_celoe)
            data.setdefault("RT-CELOE", []).append(rt_celoe)
            print(f"CELOE Train Quality: {train_f1_celoe:.3f}", end="\t")
            print(f"CELOE Test Quality: {test_f1_celoe:.3f}", end="\t")
            print(f"CELOE Runtime: {rt_celoe:.3f}")

            print("Evo starts..", end="\t")
            start_time = time.time()
            # BUG: Evolearner needs to be initalized for each learning problem
            evolearner = EvoLearner(knowledge_base=KnowledgeBase(path=args.kb),
                                    quality_func=F1(),
                                    max_runtime=args.max_runtime)
            pred_evo = evolearner.fit(train_lp).best_hypotheses()
            rt_evo = time.time() - start_time
            print("Evo ends..", end="\t")
            # () Quality on the training data
            train_f1_evo = compute_f1_score(individuals=frozenset({i for i in kb.individuals(pred_evo)}),
                                            pos=train_lp.pos,
                                            neg=train_lp.neg)
            # () Quality on test data
            test_f1_evo = compute_f1_score(individuals=frozenset({i for i in kb.individuals(pred_evo)}),
                                           pos=test_lp.pos,
                                           neg=test_lp.neg)
            # Reporting
            data.setdefault("Train-F1-Evo", []).append(train_f1_evo)
            data.setdefault("Test-F1-Evo", []).append(test_f1_evo)
            data.setdefault("RT-Evo", []).append(rt_evo)
            print(f"Evo Train Quality: {train_f1_evo:.3f}", end="\t")
            print(f"Evo Test Quality: {test_f1_evo:.3f}", end="\t")
            print(f"Evo Runtime: {rt_evo:.3f}")

            print("DRILL starts..", end="\t")
            start_time = time.time()
            pred_drill = drill.fit(train_lp).best_hypotheses()
            rt_drill = time.time() - start_time
            print("DRILL ends..", end="\t")

            # () Quality on the training data
            train_f1_drill = compute_f1_score(individuals=frozenset({i for i in kb.individuals(pred_drill)}),
                                              pos=train_lp.pos,
                                              neg=train_lp.neg)
            # () Quality on test data
            test_f1_drill = compute_f1_score(individuals=frozenset({i for i in kb.individuals(pred_drill)}),
                                             pos=test_lp.pos,
                                             neg=test_lp.neg)
            # Reporting
            data.setdefault("Train-F1-DRILL", []).append(train_f1_drill)
            data.setdefault("Test-F1-DRILL", []).append(test_f1_drill)
            data.setdefault("RT-DRILL", []).append(rt_drill)
            print(f"DRILL Train Quality: {train_f1_drill:.3f}", end="\t")
            print(f"DRILL Test Quality: {test_f1_drill:.3f}", end="\t")
            print(f"DRILL Runtime: {rt_drill:.3f}")


            print("TDL starts..", end="\t")
            start_time = time.time()
            # () Fit model on training dataset
            pred_tdl = tdl.fit(train_lp).best_hypotheses(n=1)
            print("TDL ends..", end="\t")
            rt_tdl = time.time() - start_time

            # () Quality on the training data
            train_f1_tdl = compute_f1_score(individuals=frozenset({i for i in kb.individuals(pred_tdl)}),
                                            pos=train_lp.pos,
                                            neg=train_lp.neg)
            # () Quality on test data
            test_f1_tdl = compute_f1_score(individuals=frozenset({i for i in kb.individuals(pred_tdl)}),
                                           pos=test_lp.pos,
                                           neg=test_lp.neg)

            data.setdefault("Train-F1-TDL", []).append(train_f1_tdl)
            data.setdefault("Test-F1-TDL", []).append(test_f1_tdl)
            data.setdefault("RT-TDL", []).append(rt_tdl)
            print(f"TDL Train Quality: {train_f1_tdl:.3f}", end="\t")
            print(f"TDL Test Quality: {test_f1_tdl:.3f}", end="\t")
            print(f"TDL Runtime: {rt_tdl:.3f}")

            start_time = time.time()
            # () Fit model on training dataset
            pred_nces = nces.fit(train_lp).best_hypotheses(n=1)
            print("NCES ends..", end="\t")
            rt_nces = time.time() - start_time
            
            # () Quality on the training data
            train_f1_nces = compute_f1_score(individuals=frozenset({i for i in kb.individuals(pred_nces)}),
                                             pos=train_lp.pos,
                                             neg=train_lp.neg)
            # () Quality on test data
            test_f1_nces = compute_f1_score(individuals=frozenset({i for i in kb.individuals(pred_nces)}),
                                            pos=test_lp.pos,
                                            neg=test_lp.neg)

            data.setdefault("Train-F1-NCES", []).append(train_f1_nces)
            data.setdefault("Test-F1-NCES", []).append(test_f1_nces)
            data.setdefault("RT-NCES", []).append(rt_nces)
            print(f"NCES Train Quality: {train_f1_nces:.3f}", end="\t")
            print(f"NCES Test Quality: {test_f1_nces:.3f}", end="\t")
            print(f"NCES Runtime: {rt_nces:.3f}")

            start_time = time.time()
            # () Fit model on training dataset
            pred_nces2 = nces2.fit(train_lp).best_hypotheses(n=1)
            print("NCES2 ends..", end="\t")
            rt_nces2 = time.time() - start_time
            
            # () Quality on the training data
            train_f1_nces2 = compute_f1_score(individuals=frozenset({i for i in kb.individuals(pred_nces2)}),
                                             pos=train_lp.pos,
                                             neg=train_lp.neg)
            # () Quality on test data
            test_f1_nces2 = compute_f1_score(individuals=frozenset({i for i in kb.individuals(pred_nces2)}),
                                            pos=test_lp.pos,
                                            neg=test_lp.neg)

            data.setdefault("Train-F1-NCES2", []).append(train_f1_nces2)
            data.setdefault("Test-F1-NCES2", []).append(test_f1_nces2)
            data.setdefault("RT-NCES2", []).append(rt_nces2)
            print(f"NCES2 Train Quality: {train_f1_nces2:.3f}", end="\t")
            print(f"NCES2 Test Quality: {test_f1_nces2:.3f}", end="\t")
            print(f"NCES2 Runtime: {rt_nces2:.3f}")
            ##
            start_time = time.time()
            # () Fit model on training dataset
            pred_roces = roces.fit(train_lp).best_hypotheses(n=1)
            print("ROCES ends..", end="\t")
            rt_roces = time.time() - start_time
            
            # () Quality on the training data
            train_f1_roces = compute_f1_score(individuals=frozenset({i for i in kb.individuals(pred_roces)}),
                                             pos=train_lp.pos,
                                             neg=train_lp.neg)
            # () Quality on test data
            test_f1_roces = compute_f1_score(individuals=frozenset({i for i in kb.individuals(pred_roces)}),
                                            pos=test_lp.pos,
                                            neg=test_lp.neg)

            data.setdefault("Train-F1-ROCES", []).append(train_f1_roces)
            data.setdefault("Test-F1-ROCES", []).append(test_f1_roces)
            data.setdefault("RT-ROCES", []).append(rt_roces)
            print(f"ROCES Train Quality: {train_f1_roces:.3f}", end="\t")
            print(f"ROCES Test Quality: {test_f1_roces:.3f}", end="\t")
            print(f"ROCES Runtime: {rt_roces:.3f}")

            ##
            print("CLIP starts..", end="\t")
            start_time = time.time()
            pred_clip = clip.fit(train_lp).best_hypotheses()
            rt_clip = time.time() - start_time
            print("CLIP ends..", end="\t")
            # () Quality on the training data
            train_f1_clip = compute_f1_score(individuals=frozenset({i for i in kb.individuals(pred_clip)}),
                                              pos=train_lp.pos,
                                              neg=train_lp.neg)
            # () Quality on test data
            test_f1_clip = compute_f1_score(individuals=frozenset({i for i in kb.individuals(pred_clip)}),
                                             pos=test_lp.pos,
                                             neg=test_lp.neg)
            
            data.setdefault("Train-F1-CLIP", []).append(train_f1_clip)
            data.setdefault("Test-F1-CLIP", []).append(test_f1_clip)
            data.setdefault("RT-CLIP", []).append(rt_clip)
            print(f"CLIP Train Quality: {train_f1_clip:.3f}", end="\t")
            print(f"CLIP Test Quality: {test_f1_clip:.3f}", end="\t")
            print(f"CLIP Runtime: {rt_clip:.3f}")

    df = pd.DataFrame.from_dict(data)
    df.to_csv(args.report, index=False)
    print(df)
    print(df.select_dtypes(include="number").mean())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Description Logic Concept Learning')
    parser.add_argument("--max_runtime", type=int, default=10, help="Max runtime")
    parser.add_argument("--lps", type=str, required=True, help="Path to the learning problems")
    parser.add_argument("--folds", type=int, default=10, help="Number of folds of cross validation.")
    parser.add_argument("--kb", type=str, required=True,
                        help="Knowledge base")
    parser.add_argument("--path_drill_embeddings", type=str, default=None)
    parser.add_argument("--path_of_nces_embeddings", type=str, default=None)
    parser.add_argument("--path_of_nces_trained_models", type=str, default=None)
    parser.add_argument("--path_of_nces2_trained_models", type=str, default=None)
    parser.add_argument("--path_of_roces_trained_models", type=str, default=None)
    parser.add_argument("--path_of_clip_embeddings", type=str, default=None)
    parser.add_argument("--report", type=str, default="report.csv")
    parser.add_argument("--random_seed", type=int, default=1)
    dl_concept_learning(parser.parse_args())
