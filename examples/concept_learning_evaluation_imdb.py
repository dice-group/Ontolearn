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
from ontolearn.learners import CELOE, OCEL
from ontolearn.concept_learner import EvoLearner
from ontolearn.learners import Drill, TDL
from ontolearn.learning_problem import PosNegLPStandard
from ontolearn.metrics import F1
from owlapy.owl_individual import OWLNamedIndividual, IRI
from owlapy import owl_expression_to_dl

import argparse
import requests

from ontolearn.utils.static_funcs import compute_f1_score

pd.set_option("display.precision", 5)


def dl_concept_learning(args):
    kb = KnowledgeBase(path=args.kb)

    drill = Drill(knowledge_base=KnowledgeBase(path=args.kb),
                  quality_func=F1(),
                  max_runtime=args.max_runtime)
    celoe = CELOE(knowledge_base=kb, quality_func=F1(), max_runtime=args.max_runtime)
    ocel = OCEL(knowledge_base=kb, quality_func=F1(), max_runtime=args.max_runtime)
    tdl = TDL(knowledge_base=KnowledgeBase(path=args.kb),
              kwargs_classifier={"random_state": 0},
              max_runtime=args.max_runtime)
    data = dict()

    def run_query(sparql_query: str):
        return requests.Session().post(args.url, data={"query": sparql_query})

    def individuals_imdb(min_rate: float, max_rate: float):
        imdb_prefix = "PREFIX imdb: <http://example.org/imdb/>"  # <https://www.imdb.com/>" (this for the 26GB dataset)
        xsd_prefix = "PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>"

        query = f"""
            {imdb_prefix}
            {xsd_prefix}
            SELECT DISTINCT ?x WHERE {{
                ?x imdb:hasRatingValue ?rating .
                FILTER(?rating >= {min_rate} && ?rating <= {max_rate})
            }} LIMIT 250
            """

        for binding in run_query(query).json()["results"]["bindings"]:
            yield OWLNamedIndividual(IRI.create(binding["x"]["value"]))

    with open('LPs/IMDB/learning_problems.json', 'r') as f:
        problems_data = json.load(f)

    # Extract and iterate over the learning problems
    for idx, problem in enumerate(problems_data["problems"]):
        print(f"\nSolving LP {idx + 1} with threshold {problem['threshold']}")

        data.setdefault("LPs", []).append(f"LP-{idx + 1}_thr-{problem['threshold']}")

        positives_uris = problem["positives"]
        negatives_uris = problem["negatives"]

        positives = {OWLNamedIndividual(IRI.create(uri)) for uri in positives_uris}
        negatives = {OWLNamedIndividual(IRI.create(uri)) for uri in negatives_uris}
        # all_instances=individuals_imdb(1.1, 2.0)

        # Create learning problem
        lp = PosNegLPStandard(pos=positives, neg=negatives, all_instances=individuals_imdb(1.1, 2.0))

        # () Fit the learning problem to the model

        print("OCEL starts..", end="\t")
        start_time = time.time()
        pred_ocel = ocel.fit(lp).best_hypotheses(n=1)
        print("OCEL ends..", end="\t")
        rt_ocel = time.time() - start_time
        f1_ocel = compute_f1_score(individuals=frozenset({i for i in kb.individuals(pred_ocel)}), pos=lp.pos,
                                   neg=lp.neg)
        data.setdefault("F1-OCEL", []).append(f1_ocel)
        data.setdefault("RT-OCEL", []).append(rt_ocel)
        data.setdefault("Solution-OCEL", []).append(owl_expression_to_dl(pred_ocel))

        print(f"OCEL Quality: {f1_ocel:.3f}", end="\t")
        print(f"OCEL Runtime: {rt_ocel:.3f}")
        print("CELOE starts..", end="\t")
        start_time = time.time()
        pred_celoe = celoe.fit(lp).best_hypotheses(n=1)
        print("CELOE Ends..", end="\t")
        rt_celoe = time.time() - start_time
        f1_celoe = compute_f1_score(individuals=frozenset({i for i in kb.individuals(pred_celoe)}), pos=lp.pos,
                                    neg=lp.neg)
        data.setdefault("F1-CELOE", []).append(f1_celoe)
        data.setdefault("RT-CELOE", []).append(rt_celoe)
        data.setdefault("Solution-CELOE", []).append(owl_expression_to_dl(pred_celoe))

        print(f"CELOE Quality: {f1_celoe:.3f}", end="\t")
        print(f"CELOE Runtime: {rt_celoe:.3f}")

        print("Evo starts..", end="\t")
        start_time = time.time()
        # Evolearner has a bug and KB needs to be reloaded
        evo = EvoLearner(knowledge_base=KnowledgeBase(path=args.kb), quality_func=F1(), max_runtime=args.max_runtime,
                         use_data_properties=False, use_card_restrictions=False, population_size=10, num_generations=5)
        pred_evo = evo.fit(lp).best_hypotheses(n=1)
        print("Evo ends..", end="\t")
        rt_evo = time.time() - start_time
        f1_evo = compute_f1_score(individuals=frozenset({i for i in kb.individuals(pred_evo)}), pos=lp.pos, neg=lp.neg)
        data.setdefault("F1-Evo", []).append(f1_evo)
        data.setdefault("RT-Evo", []).append(rt_evo)
        data.setdefault("Solution-Evo", []).append(owl_expression_to_dl(pred_evo))
        print(f"Evo Quality: {f1_evo:.3f}", end="\t")
        print(f"Evo Runtime: {rt_evo:.3f}")

        print("DRILL starts..", end="\t")
        start_time = time.time()
        try:
            pred_drill = drill.fit(lp).best_hypotheses(n=1)
            print("DRILL ends..", end="\t")
            rt_drill = time.time() - start_time
            f1_drill = compute_f1_score(
                individuals=frozenset({i for i in kb.individuals(pred_drill)}),
                pos=lp.pos,
                neg=lp.neg
            )
            print(f"DRILL Quality: {f1_drill:.3f}", end="\t")
            print(f"DRILL Runtime: {rt_drill:.3f}")
            print(owl_expression_to_dl(pred_drill))
            solution_drill = owl_expression_to_dl(pred_drill)
        except Exception as e:
            print(f"DRILL failed: {str(e)}", end="\t")
            pred_drill = None
            f1_drill = 0.0
            rt_drill = -1
            solution_drill = "FAIL"

        # Make sure you always append to all keys
        data.setdefault("F1-DRILL", []).append(f1_drill)
        data.setdefault("RT-DRILL", []).append(rt_drill)
        data.setdefault("Solution-DRILL", []).append(solution_drill)

        print("TDL starts..", end="\t")
        start_time = time.time()
        # () Fit model training dataset
        pred_tdl = tdl.fit(lp).best_hypotheses(n=1)
        print("TDL ends..", end="\t")
        rt_tdl = time.time() - start_time
        print(owl_expression_to_dl(pred_tdl))

        # () Quality on the training data
        f1_tdl = compute_f1_score(individuals=frozenset({i for i in kb.individuals(pred_tdl)}),
                                  pos=lp.pos,
                                  neg=lp.neg)

        data.setdefault("F1-TDL", []).append(f1_tdl)
        data.setdefault("RT-TDL", []).append(rt_tdl)
        data.setdefault("Solution-TDL", []).append(owl_expression_to_dl(pred_tdl))
        print(f"TDL Quality: {f1_tdl:.3f}", end="\t")
        print(f"TDL Runtime: {rt_tdl:.3f}")
    df = pd.DataFrame.from_dict(data)
    df.to_csv(args.report, index=False)
    print(df)
    print(df.select_dtypes(include="number").mean())

    print(df.select_dtypes(include="number").mean().values.tolist())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Description Logic Concept Learning')
    parser.add_argument("--max_runtime", type=int, default=3)
    parser.add_argument("--lps", type=str, default="LPs/IMDB/learning_problems.json")  # , required=True)
    parser.add_argument("--kb", type=str, default="/home/dice/Downloads/IMDB/imdb_small.owl")  # ,required=True)
    parser.add_argument("--path_pretrained_kge", type=str, default=None)
    parser.add_argument("--report", type=str, default="report.csv")
    parser.add_argument('--url', default="http://localhost:3030/imdb_small/sparql",
                        type=str, help='The triplestore endpoint.')  # http://localhost:3030/imdb/sparql
    dl_concept_learning(parser.parse_args())
