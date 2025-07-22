"""
Regression Test for the example.
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
import argparse
from ontolearn.utils.static_funcs import compute_f1_score
pd.set_option("display.precision", 5)

class TestConceptLearning:
    def test_learning(self):

        with open('LPs/Family/lps.json') as json_file:
            settings = json.load(json_file)

        path_kb="KGs/Family/family-benchmark_rich_background.owl"
        max_runtime=1
        kb = KnowledgeBase(path=path_kb)
        ocel = OCEL(knowledge_base=kb, quality_func=F1(), max_runtime=max_runtime)
        celoe = CELOE(knowledge_base=kb, quality_func=F1(), max_runtime=max_runtime)
        drill = Drill(knowledge_base=KnowledgeBase(path=path_kb),
                      quality_func=F1(),
                      max_runtime=max_runtime)
        tdl = TDL(knowledge_base=KnowledgeBase(path=path_kb),
                  kwargs_classifier={"random_state": 0},
                  max_runtime=max_runtime)
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
            f1_ocel = compute_f1_score(individuals=frozenset({i for i in kb.individuals(pred_ocel)}), pos=lp.pos, neg=lp.neg)
            data.setdefault("F1-OCEL", []).append(f1_ocel)
            data.setdefault("RT-OCEL", []).append(rt_ocel)
            print(f"OCEL Quality: {f1_ocel:.3f}", end="\t")
            print(f"OCEL Runtime: {rt_ocel:.3f}")
            print("CELOE starts..", end="\t")
            start_time = time.time()
            pred_celoe = celoe.fit(lp).best_hypotheses(n=1)
            print("CELOE Ends..", end="\t")
            rt_celoe = time.time() - start_time
            f1_celoe = compute_f1_score(individuals=frozenset({i for i in kb.individuals(pred_celoe)}), pos=lp.pos, neg=lp.neg)
            data.setdefault("F1-CELOE", []).append(f1_celoe)
            data.setdefault("RT-CELOE", []).append(rt_celoe)
            print(f"CELOE Quality: {f1_celoe:.3f}", end="\t")
            print(f"CELOE Runtime: {rt_celoe:.3f}")

            print("Evo starts..", end="\t")
            start_time = time.time()
            # Evolearner has a bug and KB needs to be reloaded
            evo = EvoLearner(knowledge_base=KnowledgeBase(path=path_kb), quality_func=F1(), max_runtime=max_runtime)
            pred_evo = evo.fit(lp).best_hypotheses(n=1)
            print("Evo ends..", end="\t")
            rt_evo = time.time() - start_time
            f1_evo = compute_f1_score(individuals=frozenset({i for i in kb.individuals(pred_evo)}), pos=lp.pos, neg=lp.neg)
            data.setdefault("F1-Evo", []).append(f1_evo)
            data.setdefault("RT-Evo", []).append(rt_evo)
            print(f"Evo Quality: {f1_evo:.3f}", end="\t")
            print(f"Evo Runtime: {rt_evo:.3f}")

            print("DRILL starts..", end="\t")
            start_time = time.time()
            pred_drill = drill.fit(lp).best_hypotheses(n=1)
            print("DRILL ends..", end="\t")
            rt_drill = time.time() - start_time
            f1_drill = compute_f1_score(individuals=frozenset({i for i in kb.individuals(pred_drill)}), pos=lp.pos, neg=lp.neg)
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
            f1_tdl = compute_f1_score(individuals=frozenset({i for i in kb.individuals(pred_tdl)}),
                                            pos=lp.pos,
                                            neg=lp.neg)

            data.setdefault("F1-TDL", []).append(f1_tdl)
            data.setdefault("RT-TDL", []).append(rt_tdl)
            print(f"TDL Quality: {f1_tdl:.3f}", end="\t")
            print(f"TDL Runtime: {rt_tdl:.3f}")
        df = pd.DataFrame.from_dict(data)
        """
        F1-OCEL     0.96130
        RT-OCEL     0.32531
        F1-CELOE    0.96744
        RT-CELOE    0.19228
        F1-Evo      0.98937
        RT-Evo      0.55212
        F1-DRILL    0.94741
        RT-DRILL    0.52200
        F1-TDL      1.00000
        RT-TDL      0.35019
        """
        for i, (x,y) in enumerate(zip(df.select_dtypes(include="number").mean().values.tolist(), [0.94,
                                                                                                  0.3,
                                                                                                  0.95,
                                                                                                  0.2,
                                                                                                  0.97,
                                                                                                  0.1,
                                                                                                  0.90,
                                                                                                  0.4,
                                                                                                  0.95,
                                                                                                  0.3])):
            if i==0 or i%2==0:
                assert x>y