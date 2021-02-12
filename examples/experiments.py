from ontolearn import KnowledgeBase, LearningProblemGenerator, DrillSample, DrillAverage
from ontolearn import CELOE, OCEL, DLFOILHeuristic, CustomConceptLearner
from ontolearn import Experiments
from ontolearn.binders import DLLearnerBinder
import pandas as pd
from argparse import ArgumentParser
import os


def sanity_checking_args(args):
    assert os.path.isfile(args.path_knowledge_base)
    assert os.path.isfile(args.path_knowledge_base_embeddings)
    assert args.min_num_concepts > 0
    assert args.min_length_of_concepts > 0
    assert os.path.isfile(args.path_knowledge_base)


def start(args):
    sanity_checking_args(args)
    kb = KnowledgeBase(args.path_knowledge_base)
    emb = pd.read_csv(args.path_knowledge_base_embeddings, index_col=0)

    lp = LearningProblemGenerator(knowledge_base=kb)
    balanced_examples = lp.get_balanced_examples(min_num_problems=args.min_num_concepts,
                                                 num_diff_runs=1,  # This must be optimized
                                                 min_length=args.min_length_of_concepts,
                                                 min_num_instances=args.min_num_instances_per_concept)

    # Initialize models
    celoe = DLLearnerBinder(binary_path=args.path_dl_learner, kb_path=args.path_knowledge_base, model='celoe')
    ocel = DLLearnerBinder(binary_path=args.path_dl_learner, kb_path=args.path_knowledge_base, model='ocel')
    eltl = DLLearnerBinder(binary_path=args.path_dl_learner, kb_path=args.path_knowledge_base, model='eltl')

    celoe_python = CELOE(knowledge_base=kb, verbose=0)
    ocel_python = OCEL(knowledge_base=kb, verbose=0)
    dl_foil = CustomConceptLearner(knowledge_base=kb, heuristic_func=DLFOILHeuristic(), verbose=0)

    drill_average = DrillAverage(knowledge_base=kb, instance_embeddings=emb, num_episode=1, verbose=0)
    drill_sample = DrillSample(knowledge_base=kb, instance_embeddings=emb, num_episode=1, verbose=0)

    exp = Experiments()
    k_fold_cross_validation = exp.start_KFold(k=args.num_fold_for_k_fold_cv, dataset=balanced_examples,
                                              models=[drill_average, drill_sample, celoe_python, ocel_python, dl_foil,
                                                      celoe, ocel, eltl],
                                              max_runtime_per_problem=args.max_num_seconds_for_search_per_concept)
    print('\n##### K-FOLD CROSS VALUATION RESULTS #####')
    for k, v in k_fold_cross_validation.items():
        f1 = v['F-measure']
        acc = v['Accuracy']
        runtime = v['Runtime']
        fol_result = '{}\t F-measure:(avg.{:.2f} | std.{:.2f})\tAccuracy:(avg.{:.2f} | std.{:.2f})\t' \
                     'Runtime:(avg.{:.2f} | std.{:.2f})'.format(k,
                                                                f1.mean(), f1.std(),
                                                                acc.mean(),
                                                                acc.std(),
                                                                runtime.mean(), runtime.std())
        print(fol_result)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--path_knowledge_base", type=str,
                        default='/home/demir/Desktop/Onto-learn_dev/data/family-benchmark_rich_background.owl')
    parser.add_argument("--path_knowledge_base_embeddings", type=str,
                        default='../embeddings/dismult_family_benchmark/instance_emb.csv')
    parser.add_argument("--path_dl_learner", type=str, default='/home/demir/Desktop/DL/dllearner-1.4.0/')

    parser.add_argument("--min_num_concepts", type=int, default=10)
    parser.add_argument("--min_length_of_concepts", type=int, default=2)
    parser.add_argument("--min_num_instances_per_concept", type=int, default=5)
    parser.add_argument("--max_num_seconds_for_search_per_concept", type=int, default=5)
    parser.add_argument("--num_fold_for_k_fold_cv", type=int, default=2)

    start(parser.parse_args())
