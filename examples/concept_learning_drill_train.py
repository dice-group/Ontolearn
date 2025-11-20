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

# Import DrillV variants
import sys
sys.path.insert(0, '.')
from drillv_variants import DrillV_Minimal, DrillV_Standard, DrillV_Enhanced, DrillV_Complex


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
    
    # Get concepts tested if available
    concepts_tested = getattr(learner, '_number_of_tested_concepts', 0)
    
    return {
        "prediction": dl_render.render(pred),
        "train_f1": train_f1,
        "test_f1": test_f1,
        "prediction_time": pred_time,
        "concepts_tested": concepts_tested
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

    # Select DrillV variant based on user choice
    variant_map = {
        'default': DrillV,
        'minimal': DrillV_Minimal,
        'standard': DrillV_Standard,
        'enhanced': DrillV_Enhanced,
        'complex': DrillV_Complex
    }
    
    DrillVClass = variant_map.get(args.drill_variant, DrillV)
    variant_name = args.drill_variant if args.drill_variant in variant_map else 'default'
    
    print(f"\nUsing DrillV variant: {variant_name.upper()}")
    if variant_name == 'minimal':
        print("  → Simplest NN (2 layers, high LR, 1 epoch)")
    elif variant_name == 'standard':
        print("  → Balanced approach (3 layers, LayerNorm, dropout, multi-epoch)")
    elif variant_name == 'enhanced':
        print("  → Standard + curriculum learning + curiosity bonus")
    elif variant_name == 'complex':
        print("  → 4 layers with residuals, target network, LR scheduling")
    elif variant_name == 'default':
        print("  → Original DrillV with all advanced RL features")
    print()

    drillv = DrillVClass(
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
    
    # Apply lean performance optimizations to DrillV (only if method exists)
    if args.enable_drillv_optimizations and hasattr(drillv, 'optimize_for_performance'):
        print("Applying lean performance optimizations to DrillV...")
        drillv.optimize_for_performance()
    elif args.enable_drillv_optimizations:
        print("Note: optimize_for_performance not available for this variant.")
    else:
        print("DrillV optimizations disabled by user.")

    # Initialize DrillV with random V-values for comparison (if requested)
    drillv_random = None
    if args.compare_random_v:
        # Use the same variant for fair comparison, but with random V-values
        drillv_random = DrillVClass(
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
            use_random_v_values=True  # Enable random V-values (only works for default DrillV)
        )
        # Apply lean optimizations to random V-value DrillV as well (if method exists)
        if hasattr(drillv_random, 'optimize_for_performance'):
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
    if args.path_pretrained_dir:
        print("Loading pretrained Drill agent...")
        drill.load(directory="pretrained_drill")
        print("Loading pretrained DrillV agent...")
        drillv.load(directory="pretrained_drillv")
    else:
        print("Training Drill agent...")
        drill_train_time = run_and_time_training(drill, train_args, directory="pretrained_drill")
        print("Training DrillV agent...")
        drillv_train_time = run_and_time_training(drillv, train_args, directory="pretrained_drillv")
        print(f"DrillV training time: {drillv_train_time:.2f} seconds\n")
        print(f"Drill training time: {drill_train_time:.2f} seconds\n")
        # Note: drillv_random doesn't need training as it uses random V-values
        if args.compare_random_v:
            print("DrillV with random V-values doesn't require training.\n")

    # time.sleep(10)  # Just to have a small break between training and testing
    # exit(0)
    # Load learning problems
    with open(args.path_learning_problem, "r", encoding="utf-8") as json_file:
        data = json.load(json_file)

    problems = data.get("problems", {})


    # Collect prediction times
    drill_times = []
    drillv_times = []
    drillv_random_times = []
    # Collect concepts tested counts
    drill_concepts = []
    drillv_concepts = []
    drillv_random_concepts = []
    # Collect F1 scores
    drill_train_f1s = []
    drill_test_f1s = []
    drillv_train_f1s = []
    drillv_test_f1s = []
    drillv_random_train_f1s = []
    drillv_random_test_f1s = []
    print("\nComparing models on each class expression:\n")
    
    # Limit the number of problems if specified
    problem_items = list(problems.items())
    if args.num_problems > 0:
        problem_items = problem_items[:args.num_problems]
        print(f"Evaluating on {len(problem_items)} problems (limited by --num_problems)\n")
    else:
        print(f"Evaluating on all {len(problem_items)} problems\n")
    
    for str_target_concept, examples in problem_items:
        p = examples['positive_examples']
        n = examples['negative_examples']
        print(f"\n{'='*80}")
        print(f"Target concept: {str_target_concept}")
        print(f"{'='*80}")
        
        # Reset V-learning agent for new LP (deletes old memory file)
        if hasattr(drillv, 'reset_for_new_lp'):
            drillv.reset_for_new_lp()

        kf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.random_seed)
        X = np.array(p + n)
        Y = np.array([1.0 for _ in p] + [0.0 for _ in n])
        
        # Track concepts per fold for analysis
        fold_drillv_concepts = []

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
            drill_concepts.append(drill_result.get('concepts_tested', 0))
            drill_train_f1s.append(drill_result['train_f1'])
            drill_test_f1s.append(drill_result['test_f1'])

            # DrillV with trained V-values
            drillv_result = run_and_time_prediction(drillv, train_lp, test_lp, kb)
            drillv_times.append(drillv_result['prediction_time'])
            drillv_concepts.append(drillv_result.get('concepts_tested', 0))
            drillv_train_f1s.append(drillv_result['train_f1'])
            drillv_test_f1s.append(drillv_result['test_f1'])
            fold_drillv_concepts.append(drillv_result.get('concepts_tested', 0))
            
            # Get performance statistics after this prediction (if method exists)
            perf_stats = drillv.get_performance_stats() if hasattr(drillv, 'get_performance_stats') else None
            
            # Get V-learning stats if using DrillV_Complex with termination agent
            v_learning_stats = None
            if hasattr(drillv, 'termination_agent') and drillv.termination_agent:
                v_learning_stats = drillv.termination_agent.get_statistics()

            # DrillV with random V-values (if comparison is enabled)
            drillv_random_result = None
            perf_stats_random = None
            if args.compare_random_v:
                drillv_random_result = run_and_time_prediction(drillv_random, train_lp, test_lp, kb)
                drillv_random_times.append(drillv_random_result['prediction_time'])
                drillv_random_concepts.append(drillv_random_result.get('concepts_tested', 0))
                drillv_random_train_f1s.append(drillv_random_result['train_f1'])
                drillv_random_test_f1s.append(drillv_random_result['test_f1'])
                perf_stats_random = drillv_random.get_performance_stats() if hasattr(drillv_random, 'get_performance_stats') else None

            # Print results
            print(f"Fold {ith + 1}:")
            print(f"  Drill (DQN):")
            print(f"    Prediction: {drill_result['prediction']}")
            print(f"    Train F1: {drill_result['train_f1']:.3f} | Test F1: {drill_result['test_f1']:.3f}")
            print(f"    Concepts tested: {drill_result['concepts_tested']}")
            print(f"    Prediction time: {drill_result['prediction_time']:.2f} seconds")
            print(f"  DrillV ({variant_name}):")
            print(f"    Prediction: {drillv_result['prediction']}")
            print(f"    Train F1: {drillv_result['train_f1']:.3f} | Test F1: {drillv_result['test_f1']:.3f}")
            print(f"    Concepts tested: {drillv_result['concepts_tested']}")
            print(f"    Prediction time: {drillv_result['prediction_time']:.2f} seconds")
            if perf_stats:
                print(f"    Optimization: {perf_stats['optimization_type']} (memory efficient: {perf_stats['memory_efficient']})")
            if v_learning_stats:
                print(f"    🤖 V-Learning: Total runs={v_learning_stats['total_runs']}, "
                      f"Best ever={v_learning_stats['best_ever_quality']:.3f}, "
                      f"Termination={v_learning_stats['termination_reason']}")
            if args.compare_random_v and drillv_random_result:
                print(f"  DrillV ({variant_name} with random values):")
                print(f"    Prediction: {drillv_random_result['prediction']}")
                print(f"    Train F1: {drillv_random_result['train_f1']:.3f} | Test F1: {drillv_random_result['test_f1']:.3f}")
                print(f"    Concepts tested: {drillv_random_result['concepts_tested']}")
                print(f"    Prediction time: {drillv_random_result['prediction_time']:.2f} seconds")
                if perf_stats_random:
                    print(f"    Optimization: {perf_stats_random['optimization_type']} (memory efficient: {perf_stats_random['memory_efficient']})")
        
        # Print V-learning trend for this problem (if agent is learning)
        if len(fold_drillv_concepts) > 1 and hasattr(drillv, 'termination_agent') and drillv.termination_agent:
            print(f"\nV-Learning Analysis for '{str_target_concept}':")
            print(f"   Concepts per fold: {fold_drillv_concepts}")
            first_fold = fold_drillv_concepts[0]
            last_fold = fold_drillv_concepts[-1]
            if first_fold > 0:
                improvement = ((first_fold - last_fold) / first_fold) * 100
                print(f"   Learning trend: {first_fold} → {last_fold} ({improvement:+.1f}% efficiency)")
                if improvement > 5:
                    print(f"   Agent is learning! Concepts decreased across folds")
                elif improvement < -5:
                    print(f"   Concepts increased (agent exploring more)")
                else:
                    print(f"   Stable performance across folds")


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
    
    # Show DrillV performance optimizations statistics (if method exists)
    if hasattr(drillv, 'print_performance_summary'):
        print("\nDrillV Lean Performance Optimizations:")
        drillv.print_performance_summary()
    
    if args.compare_random_v:
        print(f"Average DrillV (random V-values) prediction time: {avg_drillv_random_time:.2f} seconds")
        if hasattr(drillv_random, 'print_performance_summary'):
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

    # === Concepts tested comparison ===
    def safe_mean(lst):
        return float(np.mean(lst)) if lst else 0.0

    avg_drill_concepts = safe_mean(drill_concepts)
    avg_drillv_concepts = safe_mean(drillv_concepts)
    avg_drillv_random_concepts = safe_mean(drillv_random_concepts)

    print("\nConcepts tested (averages):")
    print(f"  Drill (DQN):           {avg_drill_concepts:.1f}")
    print(f"  DrillV (trained):      {avg_drillv_concepts:.1f}")
    if args.compare_random_v:
        print(f"  DrillV (random):       {avg_drillv_random_concepts:.1f}")

    # Relative improvement
    if avg_drillv_concepts > 0:
        reduction = (avg_drill_concepts - avg_drillv_concepts) / avg_drill_concepts * 100 if avg_drill_concepts > 0 else 0.0
        print(f"\nDrillV vs Drill: average concepts tested reduced by: {reduction:+.1f}%")
    if args.compare_random_v and avg_drillv_random_concepts > 0:
        rel = (avg_drillv_random_concepts - avg_drillv_concepts) / avg_drillv_random_concepts * 100 if avg_drillv_random_concepts > 0 else 0.0
        print(f"DrillV (trained) vs DrillV (random): trained reduces concepts by: {rel:+.1f}%")

    # === F1 Score comparison ===
    avg_drill_train_f1 = safe_mean(drill_train_f1s)
    avg_drill_test_f1 = safe_mean(drill_test_f1s)
    avg_drillv_train_f1 = safe_mean(drillv_train_f1s)
    avg_drillv_test_f1 = safe_mean(drillv_test_f1s)
    avg_drillv_random_train_f1 = safe_mean(drillv_random_train_f1s)
    avg_drillv_random_test_f1 = safe_mean(drillv_random_test_f1s)

    print("\nF1 Scores (averages):")
    print(f"  Drill (DQN):           Train F1={avg_drill_train_f1:.3f}, Test F1={avg_drill_test_f1:.3f}")
    print(f"  DrillV (trained):      Train F1={avg_drillv_train_f1:.3f}, Test F1={avg_drillv_test_f1:.3f}")
    if args.compare_random_v:
        print(f"  DrillV (random):       Train F1={avg_drillv_random_train_f1:.3f}, Test F1={avg_drillv_random_test_f1:.3f}")

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
                        default=5)
    parser.add_argument("--path_pretrained_dir", type=str, default=None)

    parser.add_argument("--path_learning_problem", type=str, default='LPs/Family/lps.json',
                        help="Path to a .json file that contains 2 properties 'positive_examples' and "
                             "'negative_examples'. Each of this properties should contain the IRIs of the respective"
                             "instances. e.g. 'some/path/lp.json'")
    parser.add_argument("--num_problems", type=int, default=1,
                        help="Number of problems to evaluate from the learning problem file. 0 means all problems.")
    parser.add_argument("--max_runtime", type=int, default=30, help="Max runtime")
    parser.add_argument("--folds", type=int, default=5, help="Number of folds of cross validation.")
    parser.add_argument("--random_seed", type=int, default=1)
    parser.add_argument("--iter_bound", type=int, default=10_000, help='iter_bound during testing.')
    parser.add_argument("--compare_random_v", action='store_true', 
                        help='Include comparison with random V-values to study V-values importance')
    parser.add_argument("--enable_drillv_optimizations", action='store_true', default=False,
                        help='Enable DrillV performance optimizations (caching, batch processing)')
    parser.add_argument("--drill_variant", type=str, default='default',
                        choices=['default', 'minimal', 'standard', 'enhanced', 'complex'],
                        help='DrillV variant to use: default (original with all RL features), '
                             'minimal (simplest 2-layer NN), standard (balanced 3-layer), '
                             'enhanced (standard + curriculum + curiosity), '
                             'complex (4-layer residual with target network)')
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
