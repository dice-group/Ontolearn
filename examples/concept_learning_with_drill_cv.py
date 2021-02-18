from ontolearn import KnowledgeBase, LearningProblemGenerator, DrillSample, DrillAverage
from ontolearn import Experiments
from ontolearn.util import sanity_checking_args
from ontolearn.binders import DLLearnerBinder
import pandas as pd
from argparse import ArgumentParser
import os


def start(args):
    sanity_checking_args(args)
    kb = KnowledgeBase(args.path_knowledge_base)
    lp = LearningProblemGenerator(knowledge_base=kb, min_length=args.min_length, max_length=args.max_length)
    balanced_examples = lp.get_balanced_n_samples_per_examples(n=args.num_of_randomly_created_problems_per_concept,
                                                               min_num_problems=args.min_num_concepts,
                                                               num_diff_runs=1,  # This must be optimized
                                                               min_num_instances=args.min_num_instances_per_concept)

    drill_average = DrillAverage(pretrained_model_path=args.pretrained_drill_avg_path,
                                 knowledge_base=kb, path_of_embeddings=args.path_knowledge_base_embeddings,
                                 num_episode=args.num_episode, verbose=args.verbose,
                                 num_workers=args.num_workers)

    drill_sample = DrillSample(pretrained_model_path=args.pretrained_drill_sample_path,
                               knowledge_base=kb,
                               path_of_embeddings=args.path_knowledge_base_embeddings,
                               num_episode=args.num_episode, verbose=args.verbose,
                               num_workers=args.num_workers)
    exp = Experiments()
    k_fold_cross_validation = exp.start_KFold(k=args.num_fold_for_k_fold_cv, dataset=balanced_examples,
                                              models=[drill_average, drill_sample],
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
                        default='/home/demir/Desktop/Onto-learn_dev/KGs/Family/family-benchmark_rich_background.owl')
    parser.add_argument("--path_knowledge_base_embeddings", type=str,
                        default='../embeddings/Shallom_Family/Shallom_entity_embeddings.csv')
    parser.add_argument("--min_num_concepts", type=int, default=2)
    parser.add_argument("--min_length", type=int, default=3, help='Min length of concepts to be used')
    parser.add_argument("--max_length", type=int, default=5, help='Max length of concepts to be used')
    parser.add_argument("--min_num_instances_per_concept", type=int, default=1)
    parser.add_argument("--num_of_randomly_created_problems_per_concept", type=int, default=2)
    parser.add_argument("--num_episode", type=int, default=2)
    parser.add_argument("--verbose", type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=32, help='Number of cpus used during batching')
    parser.add_argument('--pretrained_drill_sample_path', type=str, default='', help='Provide a path of .pth file')
    parser.add_argument('--pretrained_drill_avg_path', type=str, default='', help='Provide a path of .pth file')
    parser.add_argument('--num_fold_for_k_fold_cv', type=int, default=5, help='K for the k-fold cross validation')
    start(parser.parse_args())
