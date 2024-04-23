"""
Fitting DL Concept Learning Algorithms:

Given E^+  and E^-, a learner finds a concept H and F1 score is computed w.r.t. E^+, E^-, and R(H) retrieval of H.

python examples/concept_learning_evaluation.py --lps LPs/Family/lps.json --kb KGs/Family/family.owl --max_runtime 30 --report family.csv

"""

import json
import os
import time
import pandas as pd
from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.concept_learner import CELOE, OCEL, EvoLearner
from ontolearn.learners import Drill, TDL
from ontolearn.learning_problem import PosNegLPStandard
from ontolearn.metrics import F1
from owlapy.owl_individual import OWLNamedIndividual, IRI
import argparse
from rdflib import Graph

from ontolearn.utils.static_funcs import compute_f1_score

pd.set_option("display.precision", 5)


def dl_concept_learning(args):
    with open(args.lps) as json_file:
        settings = json.load(json_file)

    kb = KnowledgeBase(path=args.kb)

    ocel = OCEL(knowledge_base=kb, quality_func=F1(), max_runtime=args.max_runtime)
    celoe = CELOE(knowledge_base=kb, quality_func=F1(), max_runtime=args.max_runtime)
    drill = Drill(knowledge_base=KnowledgeBase(path=args.kb),
                  path_pretrained_kge=args.path_pretrained_kge,
                  quality_func=F1(),
                  max_runtime=args.max_runtime)
    tdl = TDL(knowledge_base=KnowledgeBase(path=args.kb),
              dataframe_triples=pd.DataFrame(
                  data=sorted([(str(s), str(p), str(o)) for s, p, o in Graph().parse(args.kb)], key=lambda x: len(x)),
                  columns=['subject', 'relation', 'object'], dtype=str),
              kwargs_classifier={"random_state": 0},
              max_runtime=args.max_runtime)
    # dictionary to store the data
    data = dict()
    for str_target_concept, examples in settings['problems'].items():
        p = set(examples['positive_examples'])
        n = set(examples['negative_examples'])
        print('\n\n')

        print('Target concept: ', str_target_concept)
        data.setdefault("LP", []).append(str_target_concept)

        typed_pos = set(map(OWLNamedIndividual, map(IRI.create, p)))
        typed_neg = set(map(OWLNamedIndividual, map(IRI.create, n)))
        lp = PosNegLPStandard(pos=typed_pos, neg=typed_neg)

        print("OCEL starts..", end="\t")
        start_time = time.time()
        pred_ocel = ocel.fit(lp).best_hypotheses(n=1)
        print("OCEL ends..", end="\t")
        rt_ocel = time.time() - start_time
        f1_ocel = compute_f1_score(individuals={i for i in kb.individuals(pred_ocel.concept)}, pos=lp.pos, neg=lp.neg)
        data.setdefault("F1-OCEL", []).append(f1_ocel)
        data.setdefault("RT-OCEL", []).append(rt_ocel)
        print(f"OCEL Quality: {f1_ocel:.3f}", end="\t")
        print(f"OCEL Runtime: {rt_ocel:.3f}")

        print("CELOE starts..", end="\t")
        start_time = time.time()
        pred_celoe = celoe.fit(lp).best_hypotheses(n=1)
        print("CELOE Ends..", end="\t")
        rt_celoe = time.time() - start_time
        f1_celoe = compute_f1_score(individuals={i for i in kb.individuals(pred_celoe.concept)}, pos=lp.pos, neg=lp.neg)
        data.setdefault("F1-CELOE", []).append(f1_celoe)
        data.setdefault("RT-CELOE", []).append(rt_celoe)
        print(f"CELOE Quality: {f1_celoe:.3f}", end="\t")
        print(f"CELOE Runtime: {rt_celoe:.3f}")

        print("Evo starts..", end="\t")
        start_time = time.time()
        # Evolearner has a bug and KB needs to be reloaded
        evo = EvoLearner(knowledge_base=KnowledgeBase(path=args.kb), quality_func=F1(), max_runtime=args.max_runtime)
        pred_evo = evo.fit(lp).best_hypotheses(n=1)
        print("Evo ends..", end="\t")
        rt_evo = time.time() - start_time
        f1_evo = compute_f1_score(individuals={i for i in kb.individuals(pred_evo.concept)}, pos=lp.pos, neg=lp.neg)
        data.setdefault("F1-Evo", []).append(f1_evo)
        data.setdefault("RT-Evo", []).append(rt_evo)
        print(f"Evo Quality: {f1_evo:.3f}", end="\t")
        print(f"Evo Runtime: {rt_evo:.3f}")

        print("DRILL starts..", end="\t")
        start_time = time.time()
        pred_drill = drill.fit(lp).best_hypotheses(n=1)
        print("DRILL ends..", end="\t")
        rt_drill = time.time() - start_time
        f1_drill = compute_f1_score(individuals=set(kb.individuals(pred_drill.concept)), pos=lp.pos, neg=lp.neg)
        data.setdefault("F1-DRILL", []).append(f1_drill)
        data.setdefault("RT-DRILL", []).append(rt_drill)
        print(f"DRILL Quality: {f1_drill:.3f}", end="\t")
        print(f"DRILL Runtime: {rt_drill:.3f}")

        print("TDL starts..", end="\t")
        start_time = time.time()
        # () Fit model training dataset
        pred_tdl = tdl.fit(lp).best_hypotheses(n=1)
        print("TDL ends..", end="\t")
        rt_tdl = time.time() - start_time

        # () Quality on the training data
        f1_tdl = compute_f1_score(individuals={i for i in kb.individuals(pred_tdl)},
                                        pos=lp.pos,
                                        neg=lp.neg)

        data.setdefault("F1-TDL", []).append(f1_tdl)
        data.setdefault("RT-TDL", []).append(rt_tdl)
        print(f"TDL Quality: {f1_tdl:.3f}", end="\t")
        print(f"TDL Runtime: {rt_tdl:.3f}")
    df = pd.DataFrame.from_dict(data)
    df.to_csv(args.report, index=False)
    print(df)
    print(df.select_dtypes(include="number").mean())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Description Logic Concept Learning')
    parser.add_argument("--max_runtime", type=int, default=1)
    parser.add_argument("--lps", type=str, required=True)
    parser.add_argument("--kb", type=str, required=True)
    parser.add_argument("--path_pretrained_kge", type=str, default=None)
    parser.add_argument("--report", type=str, default="report.csv")
    dl_concept_learning(parser.parse_args())
