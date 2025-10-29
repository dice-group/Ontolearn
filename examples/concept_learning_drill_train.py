"""
====================================================================
Drill -- Neuro-Symbolic Class Expression Learning

# Learn Embeddings
dicee --path_single_kg KGs/Family/family-benchmark_rich_background.owl --path_to_store_single_run embeddings --backend rdflib --save_embeddings_as_csv --model Keci --num_epoch 10


====================================================================
"""
import json
from argparse import ArgumentParser
import time
import numpy as np
from sklearn.model_selection import StratifiedKFold
from ontolearn.utils.static_funcs import compute_f1_score
from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.learning_problem import PosNegLPStandard
from ontolearn.refinement_operators import LengthBasedRefinement
from ontolearn.learners import Drill, DrillV
from ontolearn.metrics import F1
from ontolearn.heuristics import CeloeBasedReward
from owlapy.owl_individual import OWLNamedIndividual, IRI
from owlapy.render import DLSyntaxObjectRenderer


def run_and_time_training(learner, train_args, directory):
    start_time = time.time()
    learner.train(**train_args)
    end_time = time.time()
    learner.save(directory=directory)
    return end_time - start_time

def run_and_time_prediction(learner, train_lp, test_lp, kb):
    dl_render = DLSyntaxObjectRenderer()
    start_time = time.time()
    pred = learner.fit(train_lp).best_hypotheses()
    pred_time = time.time() - start_time
    train_f1 = compute_f1_score(individuals=frozenset({i for i in kb.individuals(pred)}),
                                pos=train_lp.pos, neg=train_lp.neg)
    test_f1 = compute_f1_score(individuals=frozenset({i for i in kb.individuals(pred)}),
                               pos=test_lp.pos, neg=test_lp.neg)
    return {
        "prediction": dl_render.render(pred),
        "train_f1": train_f1,
        "test_f1": test_f1,
        "prediction_time": pred_time
    }

def start(args):
    kb = KnowledgeBase(path=args.path_knowledge_base)
    train_args = dict(
        num_of_target_concepts=args.num_of_target_concepts,
        num_learning_problems=args.num_of_training_learning_problems
    )

    # Initialize both learners
    drill = Drill(
        knowledge_base=kb,
        path_embeddings=args.path_embeddings,
        refinement_operator=LengthBasedRefinement(knowledge_base=kb),
        quality_func=F1(),
        reward_func=CeloeBasedReward(),
        epsilon_decay=args.epsilon_decay,
        learning_rate=args.learning_rate,
        verbose=0,
        num_of_sequential_actions=args.num_of_sequential_actions,
        num_episode=args.num_episode,
        iter_bound=args.iter_bound,
        max_runtime=args.max_runtime
    )

    drillv = DrillV(
        knowledge_base=kb,
        path_embeddings=args.path_embeddings,
        refinement_operator=LengthBasedRefinement(knowledge_base=kb),
        quality_func=F1(),
        reward_func=CeloeBasedReward(),
        epsilon_decay=args.epsilon_decay,
        learning_rate=args.learning_rate,
        verbose=0,
        num_of_sequential_actions=args.num_of_sequential_actions,
        num_episode=args.num_episode,
        iter_bound=args.iter_bound,
        max_runtime=args.max_runtime
    )
    
    # Apply lean performance optimizations to DrillV
    if args.enable_drillv_optimizations:
        print("Applying lean performance optimizations to DrillV...")
        drillv.optimize_for_performance()
    else:
        print("DrillV optimizations disabled by user.")

    # Initialize DrillV with random V-values for comparison (if requested)
    drillv_random = None
    if args.compare_random_v:
        drillv_random = DrillV(
            knowledge_base=kb,
            path_embeddings=args.path_embeddings,
            refinement_operator=LengthBasedRefinement(knowledge_base=kb),
            quality_func=F1(),
            reward_func=CeloeBasedReward(),
            epsilon_decay=args.epsilon_decay,
            learning_rate=args.learning_rate,
            verbose=0,
            num_of_sequential_actions=args.num_of_sequential_actions,
            num_episode=args.num_episode,
            iter_bound=args.iter_bound,
            max_runtime=args.max_runtime,
            use_random_v_values=True  # Enable random V-values
        )
        # Apply lean optimizations to random V-value DrillV as well
        print("Applying lean performance optimizations to DrillV (random V-values)...")
        drillv_random.optimize_for_performance()

    # # Train and time both models
    # print("Training Drill (DQN)...")
    # drill_train_time = run_and_time_training(drill, train_args, directory="pretrained_drill")
    # print(f"Drill training time: {drill_train_time:.2f} seconds\n")

    # print("Training DrillV (V-learning)...")
    # drillv_train_time = run_and_time_training(drillv, train_args)
    # print(f"DrillV training time: {drillv_train_time:.2f} seconds\n")


     # Train and time both models
    # if args.path_pretrained_dir:
    print("Loading pretrained Drill agent...")
    drill.load(directory="pretrained_drill")
    print("Loading pretrained DrillV agent...")
    drillv.load(directory="pretrained_drillv")
    # else:
    #     print("Training Drill agent...")
    #     drill_train_time = run_and_time_training(drill, train_args, directory="pretrained_drill")
    #     print("Training DrillV agent...")
    #     drillv_train_time = run_and_time_training(drillv, train_args, directory="pretrained_drillv")
    #     print(f"DrillV training time: {drillv_train_time:.2f} seconds\n")
    #     print(f"Drill training time: {drill_train_time:.2f} seconds\n")
    #     # Note: drillv_random doesn't need training as it uses random V-values
    #     if args.compare_random_v:
    #         print("DrillV with random V-values doesn't require training.\n")

    time.sleep(10)  # Just to have a small break between training and testing
    # Load learning problems
    with open(args.path_learning_problem, "r", encoding="utf-8") as json_file:
        data = json.load(json_file)

    problems = data.get("problems", {})


    # Collect prediction times
    drill_times = []
    drillv_times = []
    drillv_random_times = []
    print("\nComparing models on each class expression:\n")
    for str_target_concept, examples in problems.items():
        p = examples['positive_examples']
        n = examples['negative_examples']
        print(f"\nTarget concept: {str_target_concept}")

        kf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.random_seed)
        X = np.array(p + n)
        Y = np.array([1.0 for _ in p] + [0.0 for _ in n])

        for (ith, (train_index, test_index)) in enumerate(kf.split(X, Y)):
            train_pos = {pos_individual for pos_individual in X[train_index][Y[train_index] == 1]}
            train_neg = {neg_individual for neg_individual in X[train_index][Y[train_index] == 0]}
            test_pos = {pos_individual for pos_individual in X[test_index][Y[test_index] == 1]}
            test_neg = {neg_individual for neg_individual in X[test_index][Y[test_index] == 0]}
            train_lp = PosNegLPStandard(pos=set(map(OWLNamedIndividual, map(IRI.create, train_pos))),
                                        neg=set(map(OWLNamedIndividual, map(IRI.create, train_neg))))
            test_lp = PosNegLPStandard(pos=set(map(OWLNamedIndividual, map(IRI.create, test_pos))),
                                       neg=set(map(OWLNamedIndividual, map(IRI.create, test_neg))))

            # # Drill
            drill_result = run_and_time_prediction(drill, train_lp, test_lp, kb)
            drill_times.append(drill_result['prediction_time'])

            # DrillV with trained V-values
            drillv_result = run_and_time_prediction(drillv, train_lp, test_lp, kb)
            drillv_times.append(drillv_result['prediction_time'])
            
            # Get performance statistics after this prediction
            perf_stats = drillv.get_performance_stats()

            # DrillV with random V-values (if comparison is enabled)
            drillv_random_result = None
            perf_stats_random = None
            if args.compare_random_v:
                drillv_random_result = run_and_time_prediction(drillv_random, train_lp, test_lp, kb)
                drillv_random_times.append(drillv_random_result['prediction_time'])
                perf_stats_random = drillv_random.get_performance_stats()

            # Print results
            print(f"Fold {ith + 1}:")
            print(f"  Drill (DQN):")
            print(f"    Prediction: {drill_result['prediction']}")
            print(f"    Train F1: {drill_result['train_f1']:.3f} | Test F1: {drill_result['test_f1']:.3f}")
            print(f"    Prediction time: {drill_result['prediction_time']:.2f} seconds")
            print(f"  DrillV (V-learning with trained values):")
            print(f"    Prediction: {drillv_result['prediction']}")
            print(f"    Train F1: {drillv_result['train_f1']:.3f} | Test F1: {drillv_result['test_f1']:.3f}")
            print(f"    Prediction time: {drillv_result['prediction_time']:.2f} seconds")
            print(f"    Optimization: {perf_stats['optimization_type']} (memory efficient: {perf_stats['memory_efficient']})")
            if args.compare_random_v and drillv_random_result:
                print(f"  DrillV (V-learning with random values):")
                print(f"    Prediction: {drillv_random_result['prediction']}")
                print(f"    Train F1: {drillv_random_result['train_f1']:.3f} | Test F1: {drillv_random_result['test_f1']:.3f}")
                print(f"    Prediction time: {drillv_random_result['prediction_time']:.2f} seconds")
                if perf_stats_random:
                    print(f"    Optimization: {perf_stats_random['optimization_type']} (memory efficient: {perf_stats_random['memory_efficient']})")


    # Print average prediction times
    avg_drill_time = np.mean(drill_times) if drill_times else 0
    avg_drillv_time = np.mean(drillv_times) if drillv_times else 0
    avg_drillv_random_time = np.mean(drillv_random_times) if drillv_random_times else 0
    
    print("\n" + "="*60)
    print("SUMMARY:")
    print("="*60)
    # print(f"Drill training time: {drill_train_time:.2f} seconds")
    # print(f"DrillV training time: {drillv_train_time:.2f} seconds")
    print(f"Average Drill (DQN) prediction time: {avg_drill_time:.2f} seconds")
    print(f"Average DrillV (trained V-values) prediction time: {avg_drillv_time:.2f} seconds")
    
    # Show DrillV performance optimizations statistics
    print("\nDrillV Lean Performance Optimizations:")
    drillv.print_performance_summary()
    
    if args.compare_random_v:
        print(f"Average DrillV (random V-values) prediction time: {avg_drillv_random_time:.2f} seconds")
        print("\nDrillV (Random V-values) Lean Performance Optimizations:")
        drillv_random.print_performance_summary()
        
        print("\nV-Values Importance Analysis:")
        if avg_drillv_time > 0 and avg_drillv_random_time > 0:
            time_diff_percent = ((avg_drillv_random_time - avg_drillv_time) / avg_drillv_time) * 100
            print(f"Random V-values vs Trained V-values time difference: {time_diff_percent:+.1f}%")
            if abs(time_diff_percent) < 5:
                print("→ V-values have minimal impact on prediction time")
            elif time_diff_percent > 0:
                print("→ Random V-values are slower (less efficient search)")
            else:
                print("→ Random V-values are faster (but potentially less accurate)")
                
    # Performance comparison
    if avg_drill_time > 0 and avg_drillv_time > 0:
        speedup = avg_drill_time / avg_drillv_time
        print(f"\nPerformance Comparison:")
        print(f"DrillV vs Drill speedup: {speedup:.2f}x")
        if speedup > 1.0:
            print("→ DrillV is faster than Drill!")
        elif speedup < 1.0:
            print("→ Drill is faster than DrillV")
        else:
            print("→ Similar performance")
    print("="*60)

if __name__ == '__main__':
    parser = ArgumentParser()
    # General
    parser.add_argument("--path_knowledge_base", type=str,
                        default='KGs/Family/family-benchmark_rich_background.owl')
    parser.add_argument("--path_embeddings", type=str,
                        default='Experiments/embeddings/Keci_entity_embeddings.csv')
    parser.add_argument("--num_of_target_concepts",
                        type=int,
                        default=1)
    parser.add_argument("--num_of_training_learning_problems",
                        type=int,
                        default=1)
    parser.add_argument("--path_pretrained_dir", type=str, default=None)

    parser.add_argument("--path_learning_problem", type=str, default='LPs/Family/lps.json',
                        help="Path to a .json file that contains 2 properties 'positive_examples' and "
                             "'negative_examples'. Each of this properties should contain the IRIs of the respective"
                             "instances. e.g. 'some/path/lp.json'")
    parser.add_argument("--max_runtime", type=int, default=10, help="Max runtime")
    parser.add_argument("--folds", type=int, default=2, help="Number of folds of cross validation.")
    parser.add_argument("--random_seed", type=int, default=1)
    parser.add_argument("--iter_bound", type=int, default=10_000, help='iter_bound during testing.')
    parser.add_argument("--compare_random_v", action='store_true', 
                        help='Include comparison with random V-values to study V-values importance')
    parser.add_argument("--enable_drillv_optimizations", action='store_true', default=True,
                        help='Enable DrillV performance optimizations (caching, batch processing)')
    # DQL related
    parser.add_argument("--num_episode", type=int, default=1, help='Number of trajectories created for a given lp.')

    parser.add_argument("--epsilon_decay", type=float, default=.01, help='Epsilon greedy trade off per epoch')
    parser.add_argument("--max_len_replay_memory", type=int, default=1024,
                        help='Maximum size of the experience replay')
    parser.add_argument("--num_epochs_per_replay", type=int, default=1,
                        help='Number of epochs on experience replay memory')
    parser.add_argument('--num_of_sequential_actions', type=int, default=1, help='Length of the trajectory.')

    # NN related
    parser.add_argument("--learning_rate", type=int, default=.01)

    start(parser.parse_args())
