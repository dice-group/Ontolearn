"""
StratifiedKFold Cross Validating DL Concept Learning Algorithms
Usage
python examples/concept_learning_evaluation.py
                    --lps LPs/Family/lps.json
                    --kb KGs/Family/family.owl
                    --max_runtime 30
                    --report family.csv

"""
import json
import time
import pandas as pd
from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.concept_learner import CELOE, OCEL, EvoLearner
from ontolearn.learners import Drill, TDL
from ontolearn.learning_problem import PosNegLPStandard
from ontolearn.metrics import F1
from owlapy.model import OWLNamedIndividual, IRI
import argparse
from rdflib import Graph
from sklearn.model_selection import StratifiedKFold
import numpy as np

pd.set_option("display.precision", 5)

# @TODO This should be standalone function that can be imported from ontolearn/static_funcs.py
def compute_f1_score(individuals, pos, neg):
    tp = len(pos.intersection(individuals))
    # tn = len(neg.difference(individuals))

    fp = len(neg.intersection(individuals))
    fn = len(pos.difference(individuals))

    try:
        recall = tp / (tp + fn)
    except ZeroDivisionError:
        return 0

    try:
        precision = tp / (tp + fp)
    except ZeroDivisionError:
        return 0

    if precision == 0 or recall == 0:
        return 0

    f_1 = 2 * ((precision * recall) / (precision + recall))
    return f_1


def dl_concept_learning(args):
    with open(args.lps) as json_file:
        settings = json.load(json_file)

    kb = KnowledgeBase(path=args.kb)
    ocel = OCEL(knowledge_base=KnowledgeBase(path=args.kb), quality_func=F1(),
                max_runtime=args.max_runtime)
    celoe = CELOE(knowledge_base=KnowledgeBase(path=args.kb), quality_func=F1(),
                  max_runtime=args.max_runtime)
    drill = Drill(knowledge_base=KnowledgeBase(path=args.kb), path_pretrained_kge=args.path_pretrained_kge,
                  quality_func=F1(), max_runtime=args.max_runtime)
    tdl = TDL(knowledge_base=KnowledgeBase(path=args.kb),
              dataframe_triples=pd.DataFrame(
                  data=sorted([(str(s), str(p), str(o)) for s, p, o in Graph().parse(args.kb)], key=lambda x: len(x)),
                  columns=['subject', 'relation', 'object'], dtype=str),
              kwargs_classifier={"random_state": 0},
              # default params for Decision Tree classifier
              # criterion='gini', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1,
              # min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None,
              # min_impurity_decrease=0.0, class_weight=None, ccp_alpha=0.0
              grid_search_over={'criterion': ["entropy", "gini", "log_loss"],
                                "splitter": ["random", "best"],
                                "max_features": [None, "sqrt", "log2"],
                                "min_samples_leaf": [1, 2, 3, 4, 5, 10],
                                "max_depth": [1, 2, 3, 4, 5, 10, None]},
              max_runtime=args.max_runtime)

    # dictionary to store the data
    data = dict()
    for str_target_concept, examples in settings['problems'].items():
        print('Target concept: ', str_target_concept)
        p = examples['positive_examples']
        n = examples['negative_examples']

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
            assert train_pos.issubset(examples['positive_examples'])
            assert train_neg.issubset(examples['negative_examples'])

            # () Extract positive and negative examples from test fold
            test_pos = {pos_individual for pos_individual in X[test_index][y[test_index] == 1]}
            test_neg = {neg_individual for neg_individual in X[test_index][y[test_index] == 0]}

            # Sanity checking for individuals used for testing.
            assert test_pos.issubset(examples['positive_examples'])
            assert test_neg.issubset(examples['negative_examples'])
            train_lp = PosNegLPStandard(pos=set(map(OWLNamedIndividual, map(IRI.create, train_pos))),
                                        neg=set(map(OWLNamedIndividual, map(IRI.create, train_neg))))

            test_lp = PosNegLPStandard(pos=set(map(OWLNamedIndividual, map(IRI.create, test_pos))),
                                       neg=set(map(OWLNamedIndividual, map(IRI.create, test_neg))))
            print("OCEL starts..", end="\t")
            start_time = time.time()
            pred_ocel = ocel.fit(train_lp).best_hypotheses(n=1)
            rt_ocel = time.time() - start_time
            print("OCEL ends..", end="\t")
            # () Quality on the training data
            train_f1_ocel = compute_f1_score(individuals={i for i in kb.individuals(pred_ocel.concept)},
                                             pos=train_lp.pos,
                                             neg=train_lp.neg)
            # () Quality on test data
            test_f1_ocel = compute_f1_score(individuals={i for i in kb.individuals(pred_ocel.concept)},
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
            pred_celoe = celoe.fit(train_lp).best_hypotheses(n=1)
            rt_celoe = time.time() - start_time
            print("CELOE ends..", end="\t")
            # () Quality on the training data
            train_f1_celoe = compute_f1_score(individuals={i for i in kb.individuals(pred_celoe.concept)},
                                              pos=train_lp.pos,
                                              neg=train_lp.neg)
            # () Quality on test data
            test_f1_celoe = compute_f1_score(individuals={i for i in kb.individuals(pred_celoe.concept)},
                                             pos=test_lp.pos,
                                             neg=test_lp.neg)
            # Reporting
            data.setdefault("Train-F1-CELOE", []).append(train_f1_celoe)
            data.setdefault("Test-F1-CELOE", []).append(test_f1_celoe)
            data.setdefault("RT-CELOE", []).append(rt_ocel)
            print(f"CELOE Train Quality: {train_f1_celoe:.3f}", end="\t")
            print(f"CELOE Test Quality: {test_f1_celoe:.3f}", end="\t")
            print(f"CELOE Runtime: {rt_celoe:.3f}")

            print("Evo starts..", end="\t")
            start_time = time.time()
            # BUG: Evolearner needs to be intialized for each learning problem
            evolearner = EvoLearner(knowledge_base=KnowledgeBase(path=args.kb), quality_func=F1(),
                                    max_runtime=args.max_runtime,
                                    use_data_properties=False,
                                    use_inverse=False, use_card_restrictions=False)
            pred_evo = evolearner.fit(train_lp).best_hypotheses(n=1)
            rt_evo = time.time() - start_time
            print("Evo ends..", end="\t")

            # () Quality on the training data
            train_f1_evo = compute_f1_score(individuals={i for i in kb.individuals(pred_evo.concept)},
                                            pos=train_lp.pos,
                                            neg=train_lp.neg)
            # () Quality on test data
            test_f1_evo = compute_f1_score(individuals={i for i in kb.individuals(pred_evo.concept)},
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
            pred_drill = drill.fit(train_lp).best_hypotheses(n=1)
            rt_drill = time.time() - start_time
            print("DRILL ends..", end="\t")

            # () Quality on the training data
            train_f1_drill = compute_f1_score(individuals={i for i in kb.individuals(pred_drill.concept)},
                                              pos=train_lp.pos,
                                              neg=train_lp.neg)
            # () Quality on test data
            test_f1_drill = compute_f1_score(individuals={i for i in kb.individuals(pred_drill.concept)},
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
            # () Fit model training dataset
            pred_tdl = tdl.fit(train_lp).best_hypotheses(n=1)
            print("TDL ends..", end="\t")
            rt_tdl = time.time() - start_time

            # () Quality on the training data
            train_f1_tdl = compute_f1_score(individuals={i for i in kb.individuals(pred_tdl)},
                                            pos=train_lp.pos,
                                            neg=train_lp.neg)
            # () Quality on test data
            test_f1_tdl = compute_f1_score(individuals={i for i in kb.individuals(pred_tdl)},
                                           pos=test_lp.pos,
                                           neg=test_lp.neg)

            data.setdefault("Train-F1-TDL", []).append(train_f1_tdl)
            data.setdefault("Test-F1-TDL", []).append(test_f1_tdl)
            data.setdefault("RT-TDL", []).append(rt_tdl)
            print(f"TDL Train Quality: {train_f1_tdl:.3f}", end="\t")
            print(f"TDL Test Quality: {test_f1_tdl:.3f}", end="\t")
            print(f"TDL Runtime: {rt_tdl:.3f}")

    df = pd.DataFrame.from_dict(data)
    df.to_csv(args.report, index=False)
    print(df)
    print(df.select_dtypes(include="number").mean())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Description Logic Concept Learning')
    parser.add_argument("--max_runtime", type=int, default=10, help="Max runtime")
    parser.add_argument("--lps", type=str, required=True, help="Path fto the learning problems")
    parser.add_argument("--folds", type=int, default=10, help="Number of folds of cross validation.")
    parser.add_argument("--kb", type=str, required=True,
                        help="Knowledge base")
    parser.add_argument("--path_pretrained_kge", type=str, default=None)
    parser.add_argument("--report", type=str, default="report.csv")
    parser.add_argument("--random_seed", type=int, default=1)
    dl_concept_learning(parser.parse_args())