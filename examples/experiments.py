from ontolearn import KnowledgeBase
from ontolearn.binders import DLLearnerBinder
import pandas as pd
from argparse import ArgumentParser
import os

from ontolearn.concept_learner import CELOE, OCEL, CustomConceptLearner
from ontolearn.experiments import Experiments
from ontolearn.heuristics import DLFOILHeuristic
from ontolearn.learning_problem_generator import LearningProblemGenerator
from ontolearn.rl import DrillAverage, DrillSample


def sanity_checking_args(args):
    try:
        assert os.path.isfile(args.path_knowledge_base)
    except AssertionError as e:
        print(f'--path_knowledge_base ***{args.path_knowledge_base}*** does not lead to a file.')
        raise

    assert os.path.isfile(args.path_knowledge_base_embeddings)
    assert args.min_num_concepts > 0
    assert args.min_length > 0
    assert os.path.isfile(args.path_knowledge_base)


def start(args):
    sanity_checking_args(args)
    kb = KnowledgeBase(args.path_knowledge_base)

    lp = LearningProblemGenerator(knowledge_base=kb)
    balanced_examples = lp.get_balanced_n_samples_per_examples(n=args.num_of_randomly_created_problems_per_concept,
                                                               min_num_problems=args.min_num_concepts,
                                                               num_diff_runs=1,  # This must be optimized
                                                               min_length=args.min_length,
                                                               min_num_instances=args.min_num_instances_per_concept)

    # Initialize models
    celoe = DLLearnerBinder(binary_path=args.path_dl_learner, kb_path=args.path_knowledge_base, model='celoe')
    ocel = DLLearnerBinder(binary_path=args.path_dl_learner, kb_path=args.path_knowledge_base, model='ocel')
    eltl = DLLearnerBinder(binary_path=args.path_dl_learner, kb_path=args.path_knowledge_base, model='eltl')

    celoe_python = CELOE(knowledge_base=kb, verbose=0)
    ocel_python = OCEL(knowledge_base=kb, verbose=0)
    dl_foil = CustomConceptLearner(knowledge_base=kb, heuristic_func=DLFOILHeuristic(), verbose=0)

    drill_average = DrillAverage(knowledge_base=kb,
                                 path_of_embeddings=args.path_knowledge_base_embeddings,
                                 num_episode=1, verbose=0,
                                 num_workers=args.num_workers)
    drill_sample = DrillSample(knowledge_base=kb,
                               path_of_embeddings=args.path_knowledge_base_embeddings,
                               num_episode=1, verbose=0,
                               num_workers=args.num_workers)

    Experiments(max_test_time_per_concept=args.max_test_time_per_concept).start_KFold(k=args.num_fold_for_k_fold_cv,
                                                                                      dataset=balanced_examples,
                                                                                      models=[
                                                                                          # drill_average,drill_sample,
                                                                                          # celoe_python, ocel_python,
                                                                                          # dl_foil,
                                                                                          celoe, ocel, eltl])


if __name__ == '__main__':
    parser = ArgumentParser()
    # General
    parser.add_argument("--path_knowledge_base", type=str,
                        default='/home/demir/Desktop/Onto-learn_dev/KGs/Family/family-benchmark_rich_background.owl')
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=32, help='Number of cpus used during batching')
    parser.add_argument("--path_dl_learner", type=str, default='/home/demir/Desktop/DL/dllearner-1.4.0/')

    # Concept Generation Related
    parser.add_argument("--min_num_concepts", type=int, default=10)
    parser.add_argument("--min_length", type=int, default=5, help='Min length of concepts to be used')
    parser.add_argument("--max_length", type=int, default=6, help='Max length of concepts to be used')
    parser.add_argument("--min_num_instances_per_concept", type=int, default=10)
    parser.add_argument("--num_of_randomly_created_problems_per_concept", type=int, default=5)

    # Evaluation related
    parser.add_argument('--num_fold_for_k_fold_cv', type=int, default=10, help='Number of cpus used during batching')
    parser.add_argument('--max_test_time_per_concept', type=int, default=5,
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
