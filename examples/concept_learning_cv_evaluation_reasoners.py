""" StratifiedKFold Cross Validating DL Concept Learning Algorithms
python examples/concept_learning_cv_evaluation.py --lps LPs/Family/lps_difficult.json --kb KGs/Family/family.owl --max_runtime 60 --report family.csv --path_of_nces_embeddings ./NCESData/family/embeddings/DeCaL_entity_embeddings.csv --path_of_nces_trained_models ./NCESData/family/trained_models/ --path_of_nces2_trained_models ./NCES2Data/family/trained_models/ --path_of_roces_trained_models ./ROCESData/family/trained_models/ --path_of_clip_embeddings ./CLIPData/family/embeddings/ConEx_entity_embeddings.csv

python examples/concept_learning_cv_evaluation.py --lps LPs/Carcinogenesis/lps.json --kb KGs/Carcinogenesis/carcinogenesis.owl --max_runtime 60 --report carcinogenesis.csv --path_of_nces_embeddings ./NCESData/carcinogenesis/embeddings/DeCaL_entity_embeddings.csv --path_of_nces_trained_models ./NCESData/carcinogenesis/trained_models/ --path_of_nces2_trained_models ./NCES2Data/carcinogenesis/trained_models/ --path_of_roces_trained_models ./ROCESData/carcinogenesis/trained_models/ --path_of_clip_embeddings ./CLIPData/carcinogenesis/embeddings/ConEx_entity_embeddings.csv

python examples/concept_learning_cv_evaluation.py --lps LPs/Mutagenesis/lps.json --kb KGs/Mutagenesis/mutagenesis.owl --max_runtime 60 --report mutagenesis.csv --path_of_nces_embeddings ./NCESData/mutagenesis/embeddings/DeCaL_entity_embeddings.csv --path_of_nces_trained_models ./NCESData/mutagenesis/trained_models/ --path_of_nces2_trained_models ./NCES2Data/mutagenesis/trained_models/ --path_of_roces_trained_models ./ROCESData/mutagenesis/trained_models/ --path_of_clip_embeddings ./CLIPData/mutagenesis/embeddings/ConEx_entity_embeddings.csv
"""
import json
import time
import os
import pandas as pd
from ontolearn.knowledge_base_ebr import KnowledgeBaseEBR
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
import random
from examples.retrieval_eval_under_incomplete import generate_subgraphs
from ontolearn.knowledge_base import KnowledgeBase


pd.set_option("display.precision", 5)

def dl_concept_learning(args):
    with open(args.lps) as json_file:
        settings = json.load(json_file)

    path = generate_subgraphs(kb_path = args.kb, directory = "inconsistent_semantic_bible", n=1, ratio=0, operation="inconsistent")
    # list(path)[0]
    kb = KnowledgeBaseEBR(path=list(path)[0], which_reasoner=args.reasoner, use_cache=False, path_kge="datasets_semantic_bible_kb_ontology_owl")
    # kb = KnowledgeBase(path=args.kb)


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
    # tdl = TDL(knowledge_base=kb,
    #           kwargs_classifier={"random_state": 1},
    #           max_runtime=args.max_runtime,
    #           verbose=0)
    
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
        positives_key = "positive_examples"
        negatives_key = "negative_examples"
    
    random.seed(0)
    selected_problems = random.sample(problems, 10)
   
    for str_target_concept, examples in selected_problems:
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
            evolearner = EvoLearner(knowledge_base=KnowledgeBaseEBR(path=args.kb, which_reasoner="Pellet"),
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

            #
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
    parser.add_argument("--max_runtime", type=int, default=30, help="Max runtime")
    parser.add_argument("--lps", type=str, default="LPs/Family/lps.json", help="Path to the learning problems")  # LPs/Family/lps.json
    parser.add_argument("--folds", type=int, default=2, help="Number of folds of cross validation.")
    parser.add_argument("--kb", type=str, default="datasets/semantic_bible/kb/ontology.owl",
                        help="Knowledge base")
    parser.add_argument("--path_drill_embeddings", type=str, default=None)
    parser.add_argument("--path_of_nces_embeddings", type=str, default=None)
    parser.add_argument("--path_of_nces_trained_models", type=str, default=None)
    parser.add_argument("--path_of_nces2_trained_models", type=str, default=None)
    parser.add_argument("--path_of_roces_trained_models", type=str, default=None)
    parser.add_argument("--path_of_clip_embeddings", type=str, default=None)
    parser.add_argument("--report", type=str, default="report.csv")
    parser.add_argument("--random_seed", type=int, default=1)
    parser.add_argument("--reasoner", type=str, default="Pellet", choices=["EBR", "Pellet", "HermiT", "JFact", "Openllet", "abstract_reasoner"])

    dl_concept_learning(parser.parse_args())