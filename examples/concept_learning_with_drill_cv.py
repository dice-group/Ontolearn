"""
====================================================================
Drill -- Deep Reinforcement Learning for Refinement Operators in ALC
====================================================================
Drill with k-fold cross validation.
Author: Caglar Demir
"""
from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.experiments import Experiments
from ontolearn.learning_problem_generator import LearningProblemGenerator
from ontolearn.rl import DrillAverage, DrillSample
from ontolearn.utils import sanity_checking_args
from argparse import ArgumentParser


def start(args):
    sanity_checking_args(args)
    kb = KnowledgeBase(args.path_knowledge_base)
    lp = LearningProblemGenerator(knowledge_base=kb, min_length=args.min_length, max_length=args.max_length)
    balanced_examples = lp.get_balanced_n_samples_per_examples(n=args.num_of_randomly_created_problems_per_concept,
                                                               min_num_problems=args.min_num_concepts,
                                                               num_diff_runs=1,  # This must be optimized
                                                               min_num_instances=args.min_num_instances_per_concept)

    drill_average = DrillAverage(pretrained_model_path=args.pretrained_drill_avg_path,
                                 num_of_sequential_actions=args.num_of_sequential_actions,
                                 knowledge_base=kb, path_of_embeddings=args.path_knowledge_base_embeddings,
                                 num_episode=args.num_episode, verbose=args.verbose,
                                 num_workers=args.num_workers)

    drill_sample = DrillSample(pretrained_model_path=args.pretrained_drill_sample_path,
                               num_of_sequential_actions=args.num_of_sequential_actions,
                               knowledge_base=kb,
                               path_of_embeddings=args.path_knowledge_base_embeddings,
                               num_episode=args.num_episode, verbose=args.verbose,
                               num_workers=args.num_workers)

    Experiments(max_test_time_per_concept=args.max_test_time_per_concept).start_KFold(k=args.num_fold_for_k_fold_cv,
                                                                                      dataset=balanced_examples,
                                                                                      models=[drill_average,
                                                                                              drill_sample])


if __name__ == '__main__':
    parser = ArgumentParser()
    # General
    parser.add_argument("--path_knowledge_base", type=str,
                        default='/home/demir/Desktop/Onto-learn_dev/KGs/Family/family-benchmark_rich_background.owl')
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=32, help='Number of cpus used during batching')

    # Concept Generation Related
    parser.add_argument("--min_num_concepts", type=int, default=2)
    parser.add_argument("--min_length", type=int, default=3, help='Min length of concepts to be used')
    parser.add_argument("--max_length", type=int, default=6, help='Max length of concepts to be used')
    parser.add_argument("--min_num_instances_per_concept", type=int, default=1)
    parser.add_argument("--num_of_randomly_created_problems_per_concept", type=int, default=2)

    # Evaluation related
    parser.add_argument('--num_fold_for_k_fold_cv', type=int, default=3, help='Number of cpus used during batching')
    parser.add_argument('--max_test_time_per_concept', type=int, default=3,
                        help='Maximum allowed runtime during testing')
    # DQL related
    parser.add_argument("--path_knowledge_base_embeddings", type=str,
                        default='../embeddings/Shallom_Family/Shallom_entity_embeddings.csv')
    parser.add_argument("--num_episode", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument('--num_of_sequential_actions', type=int, default=2)
    parser.add_argument('--pretrained_drill_sample_path', type=str, default='', help='Provide a path of .pth file')
    parser.add_argument('--pretrained_drill_avg_path', type=str, default='', help='Provide a path of .pth file')
    start(parser.parse_args())
