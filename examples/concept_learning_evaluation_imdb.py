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
from ontolearn.learners import CELOE, OCEL, CELOE_PREF, OCEL_PREF, CLIP_PREF
from ontolearn.concept_learner import CLIP, EvoLearner_pref
from ontolearn.concept_learner import EvoLearner
from ontolearn.learners import Drill, TDL, DRILL_PREF
from ontolearn.learning_problem import PosNegLPStandard
from ontolearn.metrics import F1
from owlapy.owl_individual import OWLNamedIndividual, IRI
from owlapy import owl_expression_to_dl
from ontolearn.refinement_operators import ExpressRefinement, ModifiedCELOERefinement, LengthBasedRefinement
from ontolearn.fitness_functions import PreferenceBasedFitness
from ontolearn.preference_functions import preference_score_utility_based

import argparse
import requests

from ontolearn.utils.static_funcs import compute_f1_score

pd.set_option("display.precision", 5)

def run_algorithm(name, learner, lp, kb, data, url=None, with_pref=False):
    """Run a learner safely with timing, error handling, and logging."""
    try:
        print(f"{name} starts..", end="\t")
        start_time = time.time()
        pred = learner.fit(lp).best_hypotheses(n=1)
        runtime = time.time() - start_time
        print(f"{name} ends..", end="\t")

        f1 = compute_f1_score(
            individuals=frozenset(kb.individuals(pred)),
            pos=lp.pos,
            neg=lp.neg
        )

        pref = None
        if with_pref and url is not None:
            pref = preference_score_utility_based(pred, url)

        # Save results
        data.setdefault(f"F1-{name}", []).append(f1)
        data.setdefault(f"RT-{name}", []).append(runtime)
        data.setdefault(f"Solution-{name}", []).append(owl_expression_to_dl(pred))
        if pref is not None:
            data.setdefault(f"Pref-{name}", []).append(pref)

        # Logs
        print(f"{name} Quality: {f1:.3f}", end="\t")
        print(f"{name} Runtime: {runtime:.3f}")
        if pref is not None:
            print(f"{name} Preference: {pref:.3f}")

    except Exception as e:
        print(f"{name} failed: {e}")
        data.setdefault(f"F1-{name}", []).append(None)
        data.setdefault(f"RT-{name}", []).append(None)
        data.setdefault(f"Solution-{name}", []).append(None)
        if with_pref:
            data.setdefault(f"Pref-{name}", []).append(None)


def dl_concept_learning(args):
    kb = KnowledgeBase(path=args.kb)
    op = ModifiedCELOERefinement(knowledge_base=kb, use_inverse=True,
                           use_numeric_datatypes=True)

    # Instantiate learners
    celoe_pref  = CELOE_PREF(kb, quality_func=F1(), max_runtime=args.max_runtime, refinement_operator=op, url=args.url)
    celoe       = CELOE(kb, quality_func=F1(), max_runtime=args.max_runtime, refinement_operator=op)
    ocel_pref   = OCEL_PREF(kb, quality_func=F1(), max_runtime=args.max_runtime, url=args.url)
    ocel        = OCEL(kb, quality_func=F1(), max_runtime=args.max_runtime)
    clip_pref   = CLIP_PREF(kb, path_of_embeddings=None, refinement_operator=op, load_pretrained=False,
                            max_runtime=args.max_runtime, url=args.url)
    clip        = CLIP(kb, path_of_embeddings=None, refinement_operator=op, load_pretrained=False,
                       max_runtime=args.max_runtime)


    data = {}

    # Load learning problems
    with open(args.lps, 'r') as f:
        problems_data = json.load(f)

    for idx, problem in enumerate(problems_data.get("problems", [])) if not isinstance(problems_data, list) else enumerate(problems_data):
        try:
            # Handle different JSON formats
            if isinstance(problem, dict) and "positives" in problem:
                positives_uris = problem["positives"]
                negatives_uris = problem["negatives"]
                lp_name = f"LP-{idx+1}-Persona"
                print(f"\n solving learning problem {lp_name}")
            elif isinstance(problem, list) and len(problem) >= 2:
                positives_uris = problem[1].get("positive examples", [])
                negatives_uris = problem[1].get("negative examples", [])
                lp_name = f"{problem[0]}"
                print(f"\n solving learning problem {lp_name}")

            else:
                print(f"Skipping malformed LP at index {idx}")
                continue
        except Exception as e:
            print(f"Error parsing problem {idx}: {e}")
            continue

        data.setdefault("LPs", []).append(lp_name)

        positives = {OWLNamedIndividual(IRI.create(uri)) for uri in positives_uris}
        negatives = {OWLNamedIndividual(IRI.create(uri)) for uri in negatives_uris}
        lp = PosNegLPStandard(pos=positives, neg=negatives, all_instances=None)

        # Run all algorithms with safety
        # run_algorithm("OCEL", ocel, lp, kb, data, args.url, with_pref=True)
        # run_algorithm("OCEL_Pref", ocel_pref, lp, kb, data, args.url, with_pref=True)
        # run_algorithm("CELOE", celoe, lp, kb, data, args.url, with_pref=True)
        run_algorithm("CELOE_Pref", celoe_pref, lp, kb, data, args.url, with_pref=True)
        # run_algorithm("CLIP", clip, lp, kb, data, args.url, with_pref=True)
        # run_algorithm("CLIP_Pref", clip_pref, lp, kb, data, args.url, with_pref=True)

    # Save results
    df = pd.DataFrame.from_dict(data)
    df.to_csv(args.report, index=False)
    print(df)
    print(df.select_dtypes(include="number").mean())
    print(df.select_dtypes(include="number").mean().values.tolist())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Description Logic Concept Learning')
    parser.add_argument("--max_runtime", type=int, default=30)
    parser.add_argument("--lps", type=str, default="LPs/IMDB/lps_personas.json")
    parser.add_argument("--kb", type=str, default="/home/dice/Downloads/IMDB/imdb_10000.owl")
    parser.add_argument("--path_pretrained_kge", type=str, default=None)
    parser.add_argument("--report", type=str, default="report.csv")
    parser.add_argument('--url', default="http://localhost:3030/imdb_10000/sparql",
                        type=str, help='The triplestore endpoint.')
    dl_concept_learning(parser.parse_args())



#   tdl = TDL(knowledge_base=KnowledgeBase(path=args.kb),
#               kwargs_classifier={"random_state": 0},
#               max_runtime=args.max_runtime)


 # print("Evo starts..", end="\t")
        # start_time = time.time()
        # # Evolearner has a bug and KB needs to be reloaded
        # evo = EvoLearner_pref(knowledge_base=KnowledgeBase(path=args.kb), quality_func=F1(), max_runtime=args.max_runtime,
        #                  use_data_properties=True, use_card_restrictions=True, population_size=50, num_generations=25, preference_func=preference_score_utility_based)
        # # evo = EvoLearner(knowledge_base=KnowledgeBase(path=args.kb), quality_func=F1(),
        # #                       max_runtime=args.max_runtime,
        # #                       use_data_properties=True, use_card_restrictions=True, population_size=50,
        # #                       num_generations=25)
        # pred_evo = evo.fit(lp).best_hypotheses(n=1)
        # print("Evo ends..", end="\t")
        #
        # rt_evo = time.time() - start_time
        # f1_evo = compute_f1_score(individuals=frozenset({i for i in kb.individuals(pred_evo)}), pos=lp.pos, neg=lp.neg)
        # data.setdefault("F1-Evo", []).append(f1_evo)
        # data.setdefault("RT-Evo", []).append(rt_evo)
        # data.setdefault("Solution-Evo", []).append(owl_expression_to_dl(pred_evo))
        # print(f"Evo Quality: {f1_evo:.3f}", end="\t")
        # print(f"Evo Runtime: {rt_evo:.3f}")
        # print("Best solution found", owl_expression_to_dl(pred_evo))
        # exit(0)
        #
        # print("DRILL starts..", end="\t")
        # start_time = time.time()
        # # try:
        # pred_drill = drill.fit(lp).best_hypotheses(n=1)
        # print("DRILL ends..", end="\t")
        # rt_drill = time.time() - start_time
        # f1_drill = compute_f1_score(
        #     individuals=frozenset({i for i in kb.individuals(pred_drill)}),
        #     pos=lp.pos,
        #     neg=lp.neg
        # )
        # print(f"DRILL Quality: {f1_drill:.3f}", end="\t")
        # print(f"DRILL Runtime: {rt_drill:.3f}")
        # print(owl_expression_to_dl(pred_drill))
        # solution_drill = owl_expression_to_dl(pred_drill)
        # # except Exception as e:
        # #     print(f"DRILL failed: {str(e)}", end="\t")
        # #     pred_drill = None
        # #     f1_drill = 0.0
        # #     rt_drill = -1
        # #     solution_drill = "FAIL"
        # #
        # # Make sure you always append to all keys
        # data.setdefault("F1-DRILL", []).append(f1_drill)
        # data.setdefault("RT-DRILL", []).append(rt_drill)
        # data.setdefault("Solution-DRILL", []).append(solution_drill)
        #



