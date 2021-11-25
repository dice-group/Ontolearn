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
from ontolearn.heuristics import Reward
from owlapy.model import OWLOntology, OWLReasoner
from ontolearn.utils import setup_logging

setup_logging()


def ClosedWorld_ReasonerFactory(onto: OWLOntology) -> OWLReasoner:
    from owlapy.owlready2 import OWLOntology_Owlready2
    from owlapy.owlready2.temp_classes import OWLReasoner_Owlready2_TempClasses
    from owlapy.fast_instance_checker import OWLReasoner_FastInstanceChecker
    assert isinstance(onto, OWLOntology_Owlready2)
    base_reasoner = OWLReasoner_Owlready2_TempClasses(ontology=onto)
    reasoner = OWLReasoner_FastInstanceChecker(ontology=onto,
                                               base_reasoner=base_reasoner,
                                               negation_default=True)
    return reasoner


def start(args):
    kb = KnowledgeBase(path=args.path_knowledge_base, reasoner_factory=ClosedWorld_ReasonerFactory)

    min_num_instances = args.min_num_instances_ratio_per_concept * kb.individuals_count()
    max_num_instances = args.max_num_instances_ratio_per_concept * kb.individuals_count()

    # 2. Generate Learning Problems.
    lp = LearningProblemGenerator(knowledge_base=kb,
                                  min_length=args.min_length,
                                  max_length=args.max_length,
                                  min_num_instances=min_num_instances,
                                  max_num_instances=max_num_instances)

    balanced_examples = lp.get_balanced_n_samples_per_examples(
        n=args.num_of_randomly_created_problems_per_concept,
        min_length=args.min_length,
        max_length=args.max_length,
        min_num_problems=args.min_num_concepts,
        num_diff_runs=args.min_num_concepts // 2)
    drill = Drill(knowledge_base=kb, path_of_embeddings=args.path_knowledge_base_embeddings,
                  refinement_operator=LengthBasedRefinement(knowledge_base=kb), quality_func=F1(), reward_func=Reward(),
                  batch_size=args.batch_size, num_workers=args.num_workers,
                  pretrained_model_path=args.pretrained_drill_avg_path, verbose=args.verbose,
                  max_len_replay_memory=args.max_len_replay_memory, epsilon_decay=args.epsilon_decay,
                  num_epochs_per_replay=args.num_epochs_per_replay,
                  num_episodes_per_replay=args.num_episodes_per_replay, learning_rate=args.learning_rate,
                  num_of_sequential_actions=args.num_of_sequential_actions, num_episode=args.num_episode)
    drill.train(balanced_examples)
    # Vanilla testing
    for result_dict, learning_problem in zip(
            drill.fit_from_iterable(balanced_examples, max_runtime=args.max_test_time_per_concept),
            balanced_examples):
        target_class_expression, sampled_positive_examples, sampled_negative_examples = learning_problem
        print(f'\nTarget Class Expression:{target_class_expression}')
        print(f'| sampled E^+|:{len(sampled_positive_examples)}\t| sampled E^-|:{len(sampled_negative_examples)}')
        for k, v in result_dict.items():
            print(f'{k}:{v}')


if __name__ == '__main__':
    parser = ArgumentParser()
    # General
    parser.add_argument("--path_knowledge_base", type=str,
                        default='../KGs/Family/family-benchmark_rich_background.owl')
    parser.add_argument("--path_knowledge_base_embeddings", type=str,
                        default='../embeddings/ConEx_Family/ConEx_entity_embeddings.csv')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of cpus used during batching')
    parser.add_argument("--verbose", type=int, default=0, help='Higher integer reflects more info during computation')

    # Concept Generation Related
    parser.add_argument("--min_num_concepts", type=int, default=1)
    parser.add_argument("--min_length", type=int, default=3, help='Min length of concepts to be used')
    parser.add_argument("--max_length", type=int, default=5, help='Max length of concepts to be used')
    parser.add_argument("--min_num_instances_ratio_per_concept", type=float, default=.01)  # %1
    parser.add_argument("--max_num_instances_ratio_per_concept", type=float, default=.90)  # %30
    parser.add_argument("--num_of_randomly_created_problems_per_concept", type=int, default=1)
    # DQL related
    parser.add_argument("--num_episode", type=int, default=1, help='Number of trajectories created for a given lp.')
    parser.add_argument('--relearn_ratio', type=int, default=1,
                        help='Number of times the set of learning problems are reused during training.')
    parser.add_argument("--gamma", type=float, default=.99, help='The discounting rate')
    parser.add_argument("--epsilon_decay", type=float, default=.01, help='Epsilon greedy trade off per epoch')
    parser.add_argument("--max_len_replay_memory", type=int, default=1024,
                        help='Maximum size of the experience replay')
    parser.add_argument("--num_epochs_per_replay", type=int, default=2,
                        help='Number of epochs on experience replay memory')
    parser.add_argument("--num_episodes_per_replay", type=int, default=10, help='Number of episodes per repay')
    parser.add_argument('--num_of_sequential_actions', type=int, default=3, help='Length of the trajectory.')

    # The next two params shows the flexibility of our framework as agents can be continuously trained
    parser.add_argument('--pretrained_drill_avg_path', type=str,
                        default='', help='Provide a path of .pth file')
    # NN related
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--learning_rate", type=int, default=.01)
    parser.add_argument("--drill_first_out_channels", type=int, default=32)

    # Concept Learning Testing
    parser.add_argument("--iter_bound", type=int, default=10_000, help='iter_bound during testing.')
    parser.add_argument('--max_test_time_per_concept', type=int, default=3, help='Max. runtime during testing')

    start(parser.parse_args())
