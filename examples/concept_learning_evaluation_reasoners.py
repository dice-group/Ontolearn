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
from ontolearn.learning_problem import PosNegLPStandard
from ontolearn.metrics import F1
from owlapy.owl_individual import OWLNamedIndividual, IRI
import argparse
from ontolearn.utils.static_funcs import compute_f1_score
import random
from examples.retrieval_eval_under_incomplete import generate_subgraphs
import numpy as np

pd.set_option("display.precision", 5)

def dl_concept_learning(args):
    with open(args.lps) as json_file:
        settings = json.load(json_file)

    kb_origin = KnowledgeBase(path=args.kb)

    if args.operation in ["incomplete", "inconsistent"]:
        paths = generate_subgraphs(kb_path = args.kb, directory = f"{args.operation}_{args.data_name}", n=3, ratio=args.ratio, operation=args.operation)
        kb = KnowledgeBaseEBR(path=list(paths)[0], which_reasoner=args.reasoner, use_cache=args.use_cache, path_kge=None)
    else:
        kb = KnowledgeBaseEBR(path=args.kb, which_reasoner=args.reasoner, use_cache=args.use_cache, path_kge=None)

    ocel = OCEL(knowledge_base=kb, quality_func=F1(), max_runtime=args.max_runtime)
    celoe = CELOE(knowledge_base=kb, quality_func=F1(), max_runtime=args.max_runtime)
    tdl = TDL(knowledge_base=KnowledgeBase(path=args.kb),
              kwargs_classifier={"random_state": 0},
              max_runtime=args.max_runtime)
    
    # dictionary to store the data

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

    def run_on_path(kb_path, learner_cls, lp):
        try:
            kb_local = KnowledgeBaseEBR(path=kb_path, which_reasoner=args.reasoner, use_cache=args.use_cache, path_kge=None)
            learner = learner_cls(knowledge_base=kb_local, quality_func=F1(), max_runtime=args.max_runtime)
            start_time = time.time()
            pred = learner.fit(lp).best_hypotheses(n=1)
            runtime = time.time() - start_time
            f1 = compute_f1_score(
                individuals=frozenset({i for i in kb_origin.individuals(pred)}),
                pos=lp.pos, neg=lp.neg
            )
            return f1, runtime
        except AssertionError as e:
            print(f"⚠️ Skipping learning problem due to invalid pos/neg examples: {e}")
            return None, None
        except Exception as e:
            print(f"❌ Unexpected error during learner run: {e}")
            return None, None


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
            "Evo": lambda **kw: EvoLearner(knowledge_base=KnowledgeBase(path=args.kb), quality_func=F1(), max_runtime=args.max_runtime),
            "TDL": lambda **kw: TDL(knowledge_base=KnowledgeBase(path=args.kb), kwargs_classifier={"random_state": 0}, max_runtime=args.max_runtime),
        }.items():
            f1s, runtimes = [], []
            for path in paths:
                f1, rt = run_on_path(path, learner_cls, lp)
                if f1 is not None:
                    f1s.append(f1)
                    runtimes.append(rt)

            if f1s:  # Only compute if at least one successful run
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

            print(f"{algo_name}: F1={np.mean(f1s):.3f} ± {np.std(f1s):.3f}, RT={np.mean(runtimes):.3f} ± {np.std(runtimes):.3f}")

    df = pd.DataFrame.from_dict(data)
    df.to_csv(f"{args.data_name}_{args.reasoner}.csv", index=False)
    print(df)
    print(df.select_dtypes(include="number").mean())



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Description Logic Concept Learning')
    parser.add_argument("--max_runtime", type=int, default=10)
    parser.add_argument("--lps", type=str, default="LPs/Family/lps.json")#, required=True)
    parser.add_argument("--lps_difficult", type=str, default="datasets/family/training_data/training_data_prep.json")#, required=True)
    parser.add_argument("--kb", type=str, default="KGs/Family/family-benchmark_rich_background.owl")#,required=True)
    parser.add_argument("--path_pretrained_kge", type=str, default=None)
    parser.add_argument("--data_name", type=str, default=None, required=True)
    parser.add_argument("--reasoner", type=str, default="EBR", choices=["EBR", "Pellet", "HermiT", "Jfact", "Openllet"])
    parser.add_argument("--operation", type=str, default="normal", choices=["incomplete", "inconsistent", "normal"])
    parser.add_argument("--use_cache", type=bool, default=False)
    parser.add_argument("--ratio", type=float, default=0.1)
    dl_concept_learning(parser.parse_args())
