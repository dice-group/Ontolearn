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
    df.to_csv(args.report)
    print(df)
    print(df.select_dtypes(include="number").mean())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Description Logic Concept Learning')

    parser.add_argument("--max_runtime", type=int, default=3)
    parser.add_argument("--lps", type=str, default="lp_dl_learner_family.json")
    parser.add_argument("--kb", type=str, default="../KGs/Family/family-benchmark_rich_background.owl")
    parser.add_argument("--path_pretrained_kge", type=str, default=None)
    parser.add_argument("--report", type=str, default="report.csv")
    dl_concept_learning(parser.parse_args())

"""
# python concept_learning_evaluation.py --lps lp_dl_learner_family.json --report report1.csv --max_runtime 10
# python concept_learning_evaluation.py --lps lp_dl_learner_family.json --report report2.csv --max_runtime 10
# python concept_learning_evaluation.py --lps lp_dl_learner_family.json --report report3.csv --max_runtime 10
>>> import pandas as pd
>>> df1=pd.read_csv("report1.csv",index_col=0)
>>> df2=pd.read_csv("report2.csv",index_col=0)
>>> df3=pd.read_csv("report3.csv",index_col=0)
>>> dfs = pd.concat([df1, df2, df3]).groupby("LP", as_index=False)
>>> dfs.mean()
                    LP   F1-OCEL    RT-OCEL  F1-CELOE  RT-CELOE  F1-EvoLearner  RT-EvoLearner  F1-DRILL  RT-DRILL  F1-DTL    RT-DTL
0                 Aunt  0.837209  10.004693  0.911111  7.244170       0.863158      16.489554  1.000000  2.832960     1.0  0.609264
1              Brother  1.000000   0.024886  1.000000  0.007008       1.000000       0.059726  1.000000  0.392340     1.0  0.526719
2               Cousin  0.720812  10.006845  0.793296  8.462469       0.825581      17.335273  0.347826  0.381539     1.0  0.923723
3             Daughter  1.000000   0.022262  1.000000  0.007457       1.000000       0.074893  1.000000  0.391499     1.0  0.693691
4               Father  1.000000   0.003639  1.000000  0.001616       1.000000       0.008700  1.000000  0.325800     1.0  0.786219
5        Granddaughter  1.000000   0.002821  1.000000  0.001356       1.000000       0.006261  1.000000  0.287831     1.0  0.517601
6          Grandfather  1.000000   0.002848  1.000000  0.001521       1.000000       0.005708  0.939394  0.260955     1.0  0.498533
7   Grandgranddaughter  1.000000   0.002641  1.000000  0.001254       1.000000       0.003309  0.929825  0.263724     1.0  0.285099
8     Grandgrandfather  1.000000   0.750386  1.000000  0.151460       1.000000       0.631793  0.829268  0.308655     1.0  0.284541
9     Grandgrandmother  1.000000   1.116429  1.000000  0.196696       1.000000       0.636285  1.000000  0.258587     1.0  0.333231
10       Grandgrandson  1.000000   0.526538  1.000000  0.161984       1.000000       0.400462  0.324324  0.323039     1.0  0.367552
11         Grandmother  1.000000   0.004236  1.000000  0.001526       1.000000       0.006882  0.636388  0.273613     1.0  0.498075
12            Grandson  1.000000   0.003351  1.000000  0.001551       1.000000       0.006219  0.816901  0.309253     1.0  0.586232
13              Mother  1.000000   0.003790  1.000000  0.001600       1.000000       0.008417  0.650588  0.278624     1.0  0.826277
14  PersonWithASibling  1.000000   0.003490  1.000000  0.001403       0.736842      11.989331  0.479876  0.298704     1.0  1.075167
15              Sister  1.000000   0.002792  1.000000  0.001332       1.000000       0.037401  0.800000  0.268525     1.0  0.577626
16                 Son  1.000000   0.003975  1.000000  0.001746       1.000000       0.007381  0.614273  0.263386     1.0  0.689355
17               Uncle  0.904762  10.007136  0.904762  6.322552       0.926829      16.993706  0.598730  0.235830     1.0  0.529034
>>> dfs.std()
                    LP  F1-OCEL   RT-OCEL  F1-CELOE  RT-CELOE  F1-EvoLearner  RT-EvoLearner  F1-DRILL  RT-DRILL  F1-DTL    RT-DTL
0                 Aunt      0.0  0.002746       0.0  0.070133            0.0       2.080847  0.000000  1.137039     0.0  0.021724
1              Brother      0.0  0.000522       0.0  0.000120            0.0       0.006391  0.000000  0.072232     0.0  0.168754
2               Cousin      0.0  0.004921       0.0  0.105553            0.0       3.010723  0.000000  0.158457     0.0  0.033289
3             Daughter      0.0  0.000755       0.0  0.000092            0.0       0.010524  0.000000  0.084080     0.0  0.016884
4               Father      0.0  0.000093       0.0  0.000042            0.0       0.000905  0.000000  0.007273     0.0  0.022652
5        Granddaughter      0.0  0.000039       0.0  0.000003            0.0       0.000708  0.000000  0.008462     0.0  0.014275
6          Grandfather      0.0  0.000029       0.0  0.000186            0.0       0.000730  0.052486  0.008328     0.0  0.010287
7   Grandgranddaughter      0.0  0.000069       0.0  0.000031            0.0       0.000143  0.060774  0.009935     0.0  0.005352
8     Grandgrandfather      0.0  0.003388       0.0  0.000991            0.0       0.391167  0.000000  0.088471     0.0  0.011191
9     Grandgrandmother      0.0  0.992414       0.0  0.004056            0.0       0.392481  0.000000  0.010672     0.0  0.076575
10       Grandgrandson      0.0  0.017976       0.0  0.004164            0.0       0.366923  0.280873  0.074623     0.0  0.007897
11         Grandmother      0.0  0.000052       0.0  0.000057            0.0       0.000884  0.372647  0.010497     0.0  0.020158
12            Grandson      0.0  0.000040       0.0  0.000025            0.0       0.000136  0.317136  0.076084     0.0  0.023504
13              Mother      0.0  0.000106       0.0  0.000002            0.0       0.000663  0.304540  0.010068     0.0  0.096360
14  PersonWithASibling      0.0  0.000092       0.0  0.000079            0.0       0.706314  0.187684  0.009211     0.0  0.237812
15              Sister      0.0  0.000013       0.0  0.000040            0.0       0.000607  0.000000  0.010687     0.0  0.019867
16                 Son      0.0  0.000071       0.0  0.000030            0.0       0.000142  0.101701  0.002897     0.0  0.026600
17               Uncle      0.0  0.003704       0.0  0.231309            0.0       1.258143  0.100435  0.014167     0.0  0.018062
 
"""