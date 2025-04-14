"""
Fitting OWL Class Expression Learners:

Given positive examples (E^+)  and negative examples (E^-),
Evaluate the performances of OWL Class Expression Learners  w.r.t. the quality of learned/found OWL Class Expression

Example to run the script
python examples/concept_learning_evaluation.py --lps LPs/Family/lps.json --kb KGs/Family/family-benchmark_rich_background.owl --max_runtime 3 --report family.csv

"""
import json
import time
import pandas as pd
from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.knowledge_base_ebr import KnowledgeBaseEBR
from ontolearn.learners import CELOE, OCEL
from ontolearn.concept_learner import EvoLearner
from ontolearn.learners import Drill, TDL
from ontolearn.concept_learner import NCES, NCES2, ROCES, CLIP
from ontolearn.learning_problem import PosNegLPStandard
from ontolearn.metrics import F1
from owlapy.owl_individual import OWLNamedIndividual, IRI
import argparse
from ontolearn.utils.static_funcs import compute_f1_score
import random
from examples.retrieval_eval_under_incomplete import generate_subgraphs
import numpy as np
import os
from ontolearn.refinement_operators import ExpressRefinement, ModifiedCELOERefinement

pd.set_option("display.precision", 5)

def dl_concept_learning(args):
    random.seed(8)
    with open(args.lps) as json_file:
        settings = json.load(json_file)

    kb_origin = KnowledgeBase(path=args.kb)

    data = dict()
    # Randomly select 10 learning problems# Select learning problems
    if "problems" in settings:
        problems = settings['problems']
    else:
        problems = settings

    if args.lps_difficult:
        with open(args.lps_difficult) as json_file:
            settings_difficult = json.load(json_file)

        if "problems" in settings_difficult:
            problems_difficult = settings_difficult['problems']
        else:
            problems_difficult = settings_difficult

        problems = dict(random.sample(problems.items(), 5))
        problems_difficult = dict(random.sample(problems_difficult.items(), 5))
    else:
        problems_difficult = {}

    selected_problems = {**problems, **problems_difficult}

    # Prepare paths if needed
    if args.operation in ["incomplete", "inconsistent"]:
        paths = generate_subgraphs(kb_path=args.kb, directory=f"{args.operation}_{args.data_name}", n=3, ratio=args.ratio, operation=args.operation)
    else:
        paths = [args.kb]

    data = {"LP": []}

    # Initialize learners (except EvoLearner) once per path
    learners_per_algo = dict()
    for algo_name, learner_cls in {
        "OCEL": OCEL,
        "CELOE": CELOE,
        "Evo": EvoLearner,
        "nces2": NCES2,
        "clip": CLIP,
        "roces":ROCES
    }.items():
        learners_per_algo[algo_name] = dict()
        for path in paths:
            kb_local = KnowledgeBaseEBR(path=path, which_reasoner=args.reasoner, use_cache=args.use_cache, path_kge=None)

            if algo_name == "Evo":
                continue  

            if algo_name == "clip":
                learner = learner_cls(
                    knowledge_base=kb_local,
                    refinement_operator=ModifiedCELOERefinement(kb_local),
                    quality_func=F1(),
                    max_num_of_concepts_tested=int(1e9),
                    max_runtime=args.max_runtime,
                    path_of_embeddings=None,
                    pretrained_predictor_name=["LSTM", "GRU", "SetTransformer"],
                    load_pretrained=True
                )
            elif algo_name in ["nces2", "roces"]:
                learner = learner_cls(
                    kb=kb_local,
                    knowledge_base_path=path,
                    quality_func=F1(),
                    m=[128],
                    auto_train=True,
                    load_pretrained=True
                )
            else:
                learner = learner_cls(
                    knowledge_base=kb_local,
                    quality_func=F1(),
                    max_runtime=args.max_runtime
                )
            learners_per_algo[algo_name][path] = learner

    # Run each LP
    for str_target_concept, examples in selected_problems.items():
        print('\n\nTarget concept:', str_target_concept)
        p = set(examples['positive_examples'])
        n = set(examples['negative_examples'])
        typed_pos = set(map(OWLNamedIndividual, map(IRI.create, p)))
        typed_neg = set(map(OWLNamedIndividual, map(IRI.create, n)))
        lp = PosNegLPStandard(pos=typed_pos, neg=typed_neg)
        data["LP"].append(str_target_concept)

        for algo_name, learner_cls in {
            "OCEL": OCEL,
            "CELOE": CELOE,
            "Evo": EvoLearner,
            "nces2": NCES2,
            "clip": CLIP,
            "roces": ROCES
        }.items():
            f1s, runtimes = [], []

            for path in paths:
                try:
                    if algo_name == "Evo":
                        kb_local = KnowledgeBaseEBR(path=path, which_reasoner=args.reasoner, use_cache=args.use_cache, path_kge=None)
                        learner = learner_cls(
                            knowledge_base=kb_local,
                            quality_func=F1(),
                            max_runtime=args.max_runtime
                        )
                    else:
                        learner = learners_per_algo[algo_name][path]

                    start_time = time.time()
                    pred = learner.fit(lp).best_hypotheses(n=1)
                    runtime = time.time() - start_time

                    f1 = compute_f1_score(
                        individuals=frozenset({i for i in kb_origin.individuals(pred)}),
                        pos=lp.pos, neg=lp.neg
                    )
                    f1s.append(f1)
                    runtimes.append(runtime)

                except AssertionError as e:
                    print(f"⚠️ Skipping learning problem due to invalid pos/neg examples: {e}")
                except Exception as e:
                    print(f"❌ Unexpected error during learner run: {e}")

            if f1s:
                data.setdefault(f"F1-{algo_name}-mean", []).append(np.mean(f1s))
                data.setdefault(f"F1-{algo_name}-std", []).append(np.std(f1s))
                data.setdefault(f"RT-{algo_name}-mean", []).append(np.mean(runtimes))
                data.setdefault(f"RT-{algo_name}-std", []).append(np.std(runtimes))
                print(f"{algo_name}: F1={np.mean(f1s):.3f} ± {np.std(f1s):.3f}, RT={np.mean(runtimes):.3f} ± {np.std(runtimes):.3f}")
            else:
                data.setdefault(f"F1-{algo_name}-mean", []).append(None)
                data.setdefault(f"F1-{algo_name}-std", []).append(None)
                data.setdefault(f"RT-{algo_name}-mean", []).append(None)
                data.setdefault(f"RT-{algo_name}-std", []).append(None)
                print(f"{algo_name}: Skipped all paths due to invalid LP.")

    df = pd.DataFrame.from_dict(data)
    output_dir = f"Experiments_{args.operation}"
    os.makedirs(output_dir, exist_ok=True)

    if args.operation == "normal":
        df.to_csv(f"{output_dir}/{args.data_name}_{args.reasoner}.csv", index=False)
    else:
        ratio_str = str(args.ratio).replace(".", "_")
        df.to_csv(f"{output_dir}/{args.data_name}_{args.reasoner}_{ratio_str}.csv", index=False)

    print(df)
    print(df.select_dtypes(include="number").mean())



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Description Logic Concept Learning')
    parser.add_argument("--max_runtime", type=int, default=60)
    parser.add_argument("--lps", type=str, default="LPs/Family/lps.json")#, required=True)
    parser.add_argument("--lps_difficult", type=str, default="datasets/family/training_data/training_data_prep.json")#, required=True)
    parser.add_argument("--kb", type=str, default="KGs/Family/family-benchmark_rich_background.owl")#,required=True)
    parser.add_argument("--path_pretrained_kge", type=str, default=None)
    parser.add_argument("--data_name", type=str, default="family")
    parser.add_argument("--reasoner", type=str, default="EBR", choices=["EBR", "Pellet", "HermiT", "JFact", "Openllet", "Structural", "abstract_reasoner"])
    parser.add_argument("--operation", type=str, default="normal", choices=["incomplete", "inconsistent", "normal"])
    parser.add_argument("--use_cache", type=bool, default=False, help="Use the semantic cache for the reasoners")
    parser.add_argument("--ratio", type=float, default=0.1, help="level of incompleteness, inconsistencies")
    dl_concept_learning(parser.parse_args())
