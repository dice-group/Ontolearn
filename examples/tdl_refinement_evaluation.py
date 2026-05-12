""" StratifiedKFold Cross Validating DL Concept Learning Algorithms
python examples/concept_learning_cv_evaluation.py --lps LPs/Family/lps_difficult.json --kb KGs/Family/family.owl --max_runtime 60 --report family.csv --path_of_nces_embeddings ./NCESData/family/embeddings/DeCaL_entity_embeddings.csv --path_of_nces_trained_models ./NCESData/family/trained_models/ --path_of_nces2_trained_models ./NCES2Data/family/trained_models/ --path_of_roces_trained_models ./ROCESData/family/trained_models/ --path_of_clip_embeddings ./CLIPData/family/embeddings/ConEx_entity_embeddings.csv

python examples/concept_learning_cv_evaluation.py --lps LPs/Carcinogenesis/lps.json --kb KGs/Carcinogenesis/carcinogenesis.owl --max_runtime 60 --report carcinogenesis.csv --path_of_nces_embeddings ./NCESData/carcinogenesis/embeddings/DeCaL_entity_embeddings.csv --path_of_nces_trained_models ./NCESData/carcinogenesis/trained_models/ --path_of_nces2_trained_models ./NCES2Data/carcinogenesis/trained_models/ --path_of_roces_trained_models ./ROCESData/carcinogenesis/trained_models/ --path_of_clip_embeddings ./CLIPData/carcinogenesis/embeddings/ConEx_entity_embeddings.csv

python examples/concept_learning_cv_evaluation.py --lps LPs/Mutagenesis/lps.json --kb KGs/Mutagenesis/mutagenesis.owl --max_runtime 60 --report mutagenesis.csv --path_of_nces_embeddings ./NCESData/mutagenesis/embeddings/DeCaL_entity_embeddings.csv --path_of_nces_trained_models ./NCESData/mutagenesis/trained_models/ --path_of_nces2_trained_models ./NCES2Data/mutagenesis/trained_models/ --path_of_roces_trained_models ./ROCESData/mutagenesis/trained_models/ --path_of_clip_embeddings ./CLIPData/mutagenesis/embeddings/ConEx_entity_embeddings.csv
"""
from ast import List
import json
import time
import os
from typing import Union
import pandas as pd
from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.learners import TDL_refinement
from ontolearn.learning_problem import PosNegLPStandard
from ontolearn.metrics import F1
from owlapy.owl_individual import OWLNamedIndividual, IRI
import argparse
from sklearn.model_selection import StratifiedKFold
import numpy as np
from ontolearn.utils.static_funcs import compute_f1_score
import matplotlib.pyplot as plt
pd.set_option("display.precision", 5)

def plot_importance_evolution(initial_dict, refined_dicts, top_k=10,
                              train_f1_scores=None, test_f1_scores=None,
                              save_path=None):
    def to_str(obj):
        return str(obj).strip()

    initial_set = {to_str(k) for k in initial_dict.keys()}
    feature_birthdays = {feat: 0 for feat in initial_set}

    for i, d in enumerate(refined_dicts):
        for feat_obj in d.keys():
            feat_str = to_str(feat_obj)
            if feat_str not in feature_birthdays:
                feature_birthdays[feat_str] = i + 1

    features_to_plot_set = set()
    top_initial = sorted(initial_dict.items(), key=lambda x: x[1], reverse=True)
    features_to_plot_set.update(to_str(f[0]) for f in top_initial[:int(top_k)])
    for d in refined_dicts:
        top_iter = sorted(d.items(), key=lambda x: x[1], reverse=True)
        features_to_plot_set.update(to_str(f[0]) for f in top_iter[:int(top_k)])

    last_lookup = {to_str(k): v for k, v in refined_dicts[-1].items()}
    features_to_plot = sorted(features_to_plot_set, key=lambda f: last_lookup.get(f, 0), reverse=True)

    labels = ["Initial"] + ["Base DP"] + [f"Iteration {i+1}" for i in range(len(refined_dicts) - 1)]

    has_f1 = train_f1_scores is not None and test_f1_scores is not None
    n_feats = len(features_to_plot)

    if has_f1:
        fig, (ax_imp, ax_f1) = plt.subplots(
            2, 1, figsize=(13, max(10, n_feats * 0.5) + 3),
            gridspec_kw={"height_ratios": [2, 1]}
        )
    else:
        fig, ax_imp = plt.subplots(figsize=(13, max(8, n_feats * 0.5)))

    cmap = plt.cm.get_cmap('tab20' if n_feats <= 20 else 'turbo', n_feats)
    feature_colors = {feat: cmap(i) for i, feat in enumerate(features_to_plot)}

    # --- 4. FEATURE IMPORTANCE SUBPLOT ---
    for feat in features_to_plot:
        birth_idx = feature_birthdays.get(feat, 0)
        v = []
        color = feature_colors[feat]
        is_new = birth_idx > 0

        if birth_idx == 0:
            val = next((_v for _k, _v in initial_dict.items() if to_str(_k) == feat), 0)
            v.append(val)
        else:
            v.append(None)

        for i, d in enumerate(refined_dicts):
            current_step = i + 1
            if current_step < birth_idx:
                v.append(None)
            else:
                step_val = next((_v for _k, _v in d.items() if to_str(_k) == feat), 0)
                v.append(step_val)

        ax_imp.plot(labels, v, marker='o', color=color,
                    alpha=0.9 if is_new else 0.6,
                    linewidth=3 if is_new else 1.5,
                    linestyle='--' if is_new else '-',
                    zorder=5 if is_new else 2)

        if v[-1] is not None:
            ax_imp.text(len(labels) - 1, v[-1], f"  {feat}",
                        va='center', fontsize=8, color=color,
                        fontweight='bold' if is_new else 'normal')

    ax_imp.set_title(f"Feature Importance Evolution (Top {top_k} per iteration)")
    ax_imp.set_ylabel("Global SHAP Importance")
    ax_imp.grid(axis='y', linestyle='--', alpha=0.3)

    if has_f1:
        # concept_per_iteration has one entry per label — must match
        x = range(len(train_f1_scores))
        iter_labels = (["Initial", "Base DP"] + [f"Iteration {i+1}" for i in range(len(train_f1_scores) - 2)])[:len(train_f1_scores)]
        ax_f1.plot(x, train_f1_scores, marker='o', color='#2ecc71', linewidth=2, label='Train F1')
        ax_f1.plot(x, test_f1_scores, marker='s', color='#e67e22', linewidth=2, linestyle='--', label='Test F1')

        for i, (tr, te) in enumerate(zip(train_f1_scores, test_f1_scores)):
            ax_f1.annotate(f"{tr:.2f}", (i, tr), textcoords="offset points",
                           xytext=(0, 6), ha='center', fontsize=8, color='#2ecc71')
            ax_f1.annotate(f"{te:.2f}", (i, te), textcoords="offset points",
                           xytext=(0, -12), ha='center', fontsize=8, color='#e67e22')

        ax_f1.set_xticks(list(x))
        ax_f1.set_xticklabels(iter_labels, rotation=15, ha='right')
        ax_f1.set_ylabel("F1 Score")
        ax_f1.set_title("F1 Score per Concept Iteration")
        ax_f1.legend()
        ax_f1.set_ylim(0, 1.05)
        ax_f1.grid(axis='y', linestyle='--', alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
        plt.close()
    else:
        plt.show()

def parse_boolean_arg(arg_value: Union[str, bool]) -> bool:
    """
    Convert a string or boolean input into a proper boolean value.

    Args:
        arg_value (Union[str, bool]): The input value.
            Acceptable string values (case-insensitive) for True: "yes", "true", "1"
            Acceptable string values (case-insensitive) for False: "no", "false", "0"

    Returns:
        bool: The parsed boolean value.

    Raises:
        ValueError: If the input cannot be interpreted as a boolean.
    """
    if isinstance(arg_value, bool):
        return arg_value
    if isinstance(arg_value, str):
        lowered = arg_value.lower()
        if lowered in ('yes', 'true', '1'):
            return True
        elif lowered in ('no', 'false', '0'):
            return False
    raise ValueError('Boolean value expected (true/false).')

def dl_concept_learning(args):
    args.kb = os.path.abspath(args.kb)

    with open(args.lps) as json_file:
        settings = json.load(json_file)
    kb = KnowledgeBase(path=args.kb)
                
    if not args.learner_types or 'tdl_refinement' in args.learner_types:
        tdl_refinement = TDL_refinement(knowledge_base=kb,
                            kwargs_classifier={"random_state": 1},
                            max_runtime=args.max_runtime,
                            verbose=0)    

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

            if not args.learner_types or 'tdl_refinement' in args.learner_types:
                print("TDL Refinement starts..", end="\t")
                start_time = time.time()
                # () Fit model on training dataset
                tdl_refinement_object = tdl_refinement.fit(train_lp)
                tdl_concept_per_iteration = tdl_refinement_object.concept_per_iteration
                tdl_concept_per_iterion_scores_train:List = list()
                tdl_concept_per_iterion_scores_test:List = list()
                print("TDL Refinement ends..", end="\t")
                rt_tdl_refinement = time.time() - start_time

                
                for concept in tdl_concept_per_iteration:
                # () Quality on the training data
                    tdl_concept_per_iterion_scores_train.append(compute_f1_score(individuals=frozenset({i for i in kb.individuals(concept)}),
                                                pos=train_lp.pos,
                                                neg=train_lp.neg))
                # () Quality on test data
                    tdl_concept_per_iterion_scores_test.append(compute_f1_score(individuals=frozenset({i for i in kb.individuals(concept)}),
                                            pos=test_lp.pos,
                                            neg=test_lp.neg))
                train_scores = "\n".join([str(x) for x in tdl_concept_per_iterion_scores_train])
                test_scores  = "\n".join([str(x) for x in tdl_concept_per_iterion_scores_test])
                print(f"TDL Refinement Runtime: {rt_tdl_refinement:.3f}")
                if tdl_refinement_object.initial_importance_dict and tdl_refinement_object.top_feature_dicts:
                    save_path = os.path.join(f"mutagenesis_fold{ith}.png")
                    plot_importance_evolution(
                    initial_dict=tdl_refinement_object.initial_importance_dict,
                    refined_dicts=tdl_refinement_object.top_feature_dicts,
                    top_k=10,
                    train_f1_scores=tdl_concept_per_iterion_scores_train,
                    test_f1_scores=tdl_concept_per_iterion_scores_test,
                    save_path=save_path
                    )

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
    parser.add_argument("--learner_types", type=str, nargs='*', default=None, 
                        choices=["tdl_refinement"],
                        help="List of available concept learning models")
    parser.add_argument("--path_of_clip_embeddings", type=str, default=None)
    parser.add_argument("--report", type=str, default="report.csv")
    parser.add_argument("--random_seed", type=int, default=1)

    # valid neural concept guarantee
    parser.add_argument("--enforce_validity", type=parse_boolean_arg, default=None,
                    help="Use true/false to enable enforcement. If passed without value, defaults to True.")
    dl_concept_learning(parser.parse_args())
