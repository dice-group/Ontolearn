"""
====================================================================
Drill -- Deep Reinforcement Learning for Refinement Operators in ALC
====================================================================
Drill with training.
Author: Caglar Demir
"""
from argparse import ArgumentParser

from ontolearn import KnowledgeBase
from ontolearn.refinement_operators import LengthBasedRefinement
from ontolearn.learning_problem_generator import LearningProblemGenerator
from ontolearn.concept_learner import Drill
from ontolearn.metrics import F1



def start(args):
    kb = KnowledgeBase(path=args.path_knowledge_base)

    lp = LearningProblemGenerator(knowledge_base=kb, min_length=args.min_length, max_length=args.max_length)

    balanced_examples = lp.get_balanced_n_samples_per_examples(n=args.num_of_randomly_created_problems_per_concept,
                                                               min_num_problems=args.min_num_concepts,
                                                               num_diff_runs=1,  # This must be optimized
                                                               min_num_instances=args.min_num_instances_per_concept)

    drill = Drill(knowledge_base=kb,
                  path_of_embeddings=args.path_knowledge_base_embeddings,
                  refinement_operator=LengthBasedRefinement(knowledge_base=kb),
                  quality_func=F1()
                  )
    drill.train(balanced_examples)
    exit(1)
    # Vanilla testing
    for result in drill_average.fit_from_iterable(balanced_examples, max_runtime=args.max_test_time_per_concept):
        print(result)


if __name__ == '__main__':
    parser = ArgumentParser()
    # General
    parser.add_argument("--path_knowledge_base", type=str,
                        default='../KGs/Family/family-benchmark_rich_background.owl')
    parser.add_argument("--verbose", type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=32, help='Number of cpus used during batching')

    # Concept Generation Related
    parser.add_argument("--min_num_concepts", type=int, default=3)
    parser.add_argument("--min_length", type=int, default=3, help='Min length of concepts to be used')
    parser.add_argument("--max_length", type=int, default=6, help='Max length of concepts to be used')
    parser.add_argument("--min_num_instances_per_concept", type=int, default=5)
    parser.add_argument("--num_of_randomly_created_problems_per_concept", type=int, default=1)
    parser.add_argument('--max_test_time_per_concept', type=int, default=5,
                        help='Maximum allowed runtime during testing')

    # DQL related
    parser.add_argument("--num_episode", type=int, default=50, help='Number of trajectories created for a given lp.')
    parser.add_argument('--num_of_sequential_actions', type=int, default=1, help='Length of the trajectory.')
    parser.add_argument('--relearn_ratio', type=int, default=1,
                        help='Number of times the set of learning problems are reused during training.')

    parser.add_argument("--path_knowledge_base_embeddings", type=str,
                        default='../embeddings/Shallom_Family/Shallom_entity_embeddings.csv')
    # The next two params shows the flexibility of our framework as agents can be continuously trained
    parser.add_argument('--pretrained_drill_avg_path', type=str,
                        default='../pre_trained_agents/DrillHeuristic_averaging/DrillHeuristic_averaging.pth',
                        help='Provide a path of .pth file')

    # NN related
    parser.add_argument("--batch_size", type=int, default=64)

    start(parser.parse_args())
