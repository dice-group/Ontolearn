import json
import os
import time
import pandas as pd
from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.concept_learner import CELOE, OCEL, EvoLearner
from ontolearn.learners import Drill, TreeLearner
from ontolearn.learning_problem import PosNegLPStandard
from ontolearn.metrics import Accuracy, F1
from owlapy.model import OWLClass, OWLNamedIndividual, IRI
import argparse
from rdflib import Graph

pd.set_option("display.precision", 5)



def compute_f1_score(individuals, pos, neg):
    tp = len(pos.intersection(individuals))
    tn = len(neg.difference(individuals))

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
    # Our ongoing work
    dtl = TreeLearner(knowledge_base=kb,
                      dataframe_triples=pd.DataFrame(
                          data=[(str(s), str(p), str(o)) for s, p, o in Graph().parse(args.kb)],
                          columns=['subject', 'relation', 'object'], dtype=str), quality_func=F1(),
                      max_runtime=args.max_runtime)
    drill = Drill(knowledge_base=kb, path_pretrained_kge=args.path_pretrained_kge, quality_func=F1(),
                  max_runtime=args.max_runtime).train(num_of_target_concepts=2, num_learning_problems=2)

    ocel = OCEL(knowledge_base=kb, quality_func=F1(), max_runtime=args.max_runtime)
    celoe = CELOE(knowledge_base=kb, quality_func=F1(), max_runtime=args.max_runtime)
    evo = EvoLearner(knowledge_base=kb, quality_func=F1(), max_runtime=args.max_runtime)
    columns = ["LP",
               "F1-OCEL", "RT-OCEL",
               "F1-CELOE", "RT-CELOE",
               "F1-EvoLearner", "RT-EvoLearner",
               "F1-DRILL", "RT-DRILL",
               "F1-DTL", "RT-DTL"]
    values = []
    for str_target_concept, examples in settings['problems'].items():
        p = set(examples['positive_examples'])
        n = set(examples['negative_examples'])

        print('Target concept: ', str_target_concept)

        typed_pos = set(map(OWLNamedIndividual, map(IRI.create, p)))
        typed_neg = set(map(OWLNamedIndividual, map(IRI.create, n)))
        lp = PosNegLPStandard(pos=typed_pos, neg=typed_neg)


        start_time = time.time()
        # Get best prediction
        pred_dtl = dtl.fit(lp).best_hypotheses(n=1)
        rt_dtl = time.time() - start_time
        # Compute quality of best prediction
        f1_dtl = compute_f1_score(individuals={i for i in kb.individuals(pred_dtl)}, pos=lp.pos, neg=lp.neg)

        start_time = time.time()
        pred_drill = drill.fit(lp).best_hypotheses(n=1)
        rt_drill = time.time() - start_time
        f1_drill = compute_f1_score(individuals={i for i in kb.individuals(pred_drill.concept)}, pos=lp.pos, neg=lp.neg)

        start_time = time.time()
        pred_ocel = ocel.fit(lp).best_hypotheses(n=1)
        rt_ocel = time.time() - start_time
        f1_ocel = compute_f1_score(individuals={i for i in kb.individuals(pred_ocel.concept)}, pos=lp.pos, neg=lp.neg)

        start_time = time.time()
        pred_celoe = celoe.fit(lp).best_hypotheses(n=1)
        rt_celoe = time.time() - start_time
        f1_celoe = compute_f1_score(individuals={i for i in kb.individuals(pred_celoe.concept)}, pos=lp.pos, neg=lp.neg)

        start_time = time.time()
        pred_evo = evo.fit(lp).best_hypotheses(n=1)
        rt_evo = time.time() - start_time
        f1_evo = compute_f1_score(individuals={i for i in kb.individuals(pred_evo.concept)}, pos=lp.pos, neg=lp.neg)

        values.append(
            [str_target_concept,
             f1_ocel, rt_ocel,
             f1_celoe, rt_celoe,
             f1_drill, rt_drill,
             f1_evo, rt_evo,
             f1_dtl, rt_dtl])

    df = pd.DataFrame(values, columns=columns)
    print(df)
    print(df.select_dtypes(include="number").mean())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Description Logic Concept Learning')

    parser.add_argument("--max_runtime", type=int, default=3)
    parser.add_argument("--lps", type=str, default="synthetic_problems.json")
    parser.add_argument("--kb", type=str, default="../KGs/Family/family-benchmark_rich_background.owl")
    parser.add_argument("--path_pretrained_kge", type=str, default=None)
    dl_concept_learning(parser.parse_args())
