# examples/faulty_concept_learning_evaluation.py
import json
import os
import time
import pandas as pd
from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.concept_learner import CELOE, OCEL, EvoLearner
from ontolearn.learners import Drill, TDL
from ontolearn.learning_problem import PosNegLPStandard
from ontolearn.metrics import Accuracy, F1
from owlapy.model import OWLClass, OWLNamedIndividual, IRI
import argparse
from rdflib import Graph

from ontolearn.utils.static_funcs import compute_f1_score

pd.set_option("display.precision", 5)


def dl_concept_learning(args):
    with open(args.lps) as json_file:
        settings = json.load(json_file)

    kb = KnowledgeBase(path=args.kb)
    # Our ongoing work
    # kwargs_classifier is for sklearn.tree.DecisionTreeClassifier.html#sklearn-tree-decisiontreeclassifier
    tdl = TDL(knowledge_base=kb,
              # From rdflib into dataframe sorted by subject
              dataframe_triples=pd.DataFrame(
                  data=[(str(s), str(p), str(o)) for s, p, o in Graph().parse(args.kb)],
                  columns=['subject', 'relation', 'object'], dtype=str).sort_values('subject'),
              kwargs_classifier={"criterion": "gini", "random_state": 0},
              max_runtime=args.max_runtime)

    drill = Drill(knowledge_base=kb,
                  path_pretrained_kge=args.path_pretrained_kge,
                  quality_func=F1(),
                  max_runtime=args.max_runtime)
    ocel = OCEL(knowledge_base=kb, quality_func=F1(), max_runtime=args.max_runtime)
    celoe = CELOE(knowledge_base=kb, quality_func=F1(), max_runtime=args.max_runtime)
    evo = EvoLearner(knowledge_base=kb, quality_func=F1(), max_runtime=args.max_runtime)

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

        start_time = time.time()
        print("OCEL starts..", end="\t")
        pred_ocel = ocel.fit(lp).best_hypotheses(n=1)
        print("OCEL ends..", end="\t")
        rt_ocel = time.time() - start_time
        f1_ocel = compute_f1_score(individuals={i for i in kb.individuals(pred_ocel.concept)}, pos=lp.pos, neg=lp.neg)
        print(f"OCEL Quality: {f1_ocel:.3f}")
        data.setdefault("F1-OCEL", []).append(f1_ocel)
        data.setdefault("RT-OCEL", []).append(rt_ocel)
        print(f"OCEL Runtime: {rt_ocel:.3f}")

        start_time = time.time()
        print("CELOE starts..", end="\t")
        pred_celoe = celoe.fit(lp).best_hypotheses(n=1)
        print("CELOE Ends..", end="\t")
        rt_celoe = time.time() - start_time
        f1_celoe = compute_f1_score(individuals={i for i in kb.individuals(pred_celoe.concept)}, pos=lp.pos, neg=lp.neg)
        print(f"CELOE Quality: {f1_celoe:.3f}")
        data.setdefault("F1-CELOE", []).append(f1_celoe)
        data.setdefault("RT-CELOE", []).append(rt_celoe)
        print(f"CELOE Runtime: {rt_celoe:.3f}", end="\t")

        start_time = time.time()
        print("Evo starts..", end="\t")
        pred_evo = evo.fit(lp).best_hypotheses(n=1)
        print("Evo ends..", end="\t")
        rt_evo = time.time() - start_time
        f1_evo = compute_f1_score(individuals={i for i in kb.individuals(pred_evo.concept)}, pos=lp.pos, neg=lp.neg)
        print(f"Evo Quality: {f1_evo:.3f}")
        data.setdefault("F1-Evo", []).append(f1_evo)
        data.setdefault("RT-Evo", []).append(rt_evo)
        print(f"Evo Runtime: {rt_evo:.3f}", end="\t")

        start_time = time.time()
        print("DRILL starts..", end="\t")
        pred_drill = drill.fit(lp).best_hypotheses(n=1)
        print("DRILL ends..", end="\t")
        rt_drill = time.time() - start_time
        f1_drill = compute_f1_score(individuals=set(kb.individuals(pred_drill.concept)), pos=lp.pos, neg=lp.neg)
        print(f"DRILL Quality: {f1_drill:.3f}")
        data.setdefault("F1-DRILL", []).append(f1_drill)
        data.setdefault("RT-DRILL", []).append(rt_drill)
        print(f"DRILL Runtime: {rt_drill:.3f}", end="\t")

        start_time = time.time()
        # Get best prediction
        print("TDL starts..", end="\t")
        pred_tdl = tdl.fit(lp).best_hypotheses(n=1)
        print("TDL ends..", end="\t")
        rt_tdl = time.time() - start_time
        # Compute quality of best prediction
        f1_tdl = compute_f1_score(individuals={i for i in kb.individuals(pred_tdl)}, pos=lp.pos, neg=lp.neg)
        print(f"TDL Quality: {f1_tdl:.3f}", end="\t")
        print(f"TDL Runtime: {rt_tdl:.3f}")

        data.setdefault("F1-TDL", []).append(f1_tdl)
        data.setdefault("RT-TDL", []).append(rt_tdl)



    df = pd.DataFrame.from_dict(data)
    df.to_csv(args.report, index=False)
    print(df)
    print(df.select_dtypes(include="number").mean())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Description Logic Concept Learning')

    parser.add_argument("--max_runtime", type=int, default=60)
    parser.add_argument("--lps", type=str, required=True)
    parser.add_argument("--kb", type=str, required=True)
    parser.add_argument("--path_pretrained_kge", type=str, default=None)
    parser.add_argument("--report", type=str, default="report.csv")
    dl_concept_learning(parser.parse_args())