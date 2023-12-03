import json
import os
import time
import pandas as pd
from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.concept_learner import CELOE, OCEL, EvoLearner, Drill
from ontolearn.learning_problem import PosNegLPStandard
from ontolearn.metrics import Accuracy, F1
from owlapy.model import OWLClass, OWLNamedIndividual, IRI
import argparse

pd.set_option("display.precision", 5)


def dl_concept_learning(args):
    try:
        os.chdir("examples")
    except FileNotFoundError:
        pass

    with open(args.lps) as json_file:
        settings = json.load(json_file)

    kb = KnowledgeBase(path=args.kb)

    ocel=OCEL(knowledge_base=kb, quality_func=F1(), max_runtime=args.max_runtime)
    celoe=CELOE(knowledge_base=kb, quality_func=F1(), max_runtime=args.max_runtime)
    evo=EvoLearner(knowledge_base=kb, quality_func=F1(), max_runtime=args.max_runtime)
    drill=Drill(knowledge_base=kb, quality_func=F1(), max_runtime=args.max_runtime)
    columns = ["LP",
               "OCEL", "F1-OCEL", "RT-OCEL",
               "CELOE", "F1-CELOE", "RT-CELOE",
               "EvoLearner", "F1-EvoLearner", "RT-EvoLearner",
               "DRILL", "F1-DRILL", "RT-DRILL"]
    values = []
    for str_target_concept, examples in settings['problems'].items():
        p = set(examples['positive_examples'])
        n = set(examples['negative_examples'])

        print('Target concept: ', str_target_concept)

        typed_pos = set(map(OWLNamedIndividual, map(IRI.create, p)))
        typed_neg = set(map(OWLNamedIndividual, map(IRI.create, n)))
        lp = PosNegLPStandard(pos=typed_pos, neg=typed_neg)

        start_time = time.time()
        # Untrained & max runtime is not fully integrated.
        pred_drill = drill.fit(lp).best_hypotheses(n=1)
        rt_drill = time.time() - start_time


        start_time = time.time()
        pred_ocel = ocel.fit(lp).best_hypotheses(n=1)
        rt_ocel = time.time() - start_time

        start_time = time.time()
        pred_celoe = celoe.fit(lp).best_hypotheses(n=1)
        rt_celoe = time.time() - start_time

        start_time = time.time()
        pred_evo = evo.fit(lp).best_hypotheses(n=1)
        rt_evo = time.time() - start_time

        values.append(
            [str_target_concept,
             pred_ocel.str, pred_ocel.quality, rt_ocel,
             pred_celoe.str, pred_celoe.quality, rt_celoe,
             pred_evo.str, pred_evo.quality, rt_evo,
             pred_drill.str, pred_drill.quality, rt_drill])

    df = pd.DataFrame(values, columns=columns)
    print(df)
    print(df.select_dtypes(include="number").mean())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Description Logic Concept Learning')

    parser.add_argument("--max_runtime", type=int, default=10)
    parser.add_argument("--lps", type=str, default="synthetic_problems.json")
    parser.add_argument("--kb", type=str, default="../KGs/Family/family-benchmark_rich_background.owl")

    dl_concept_learning(parser.parse_args())
