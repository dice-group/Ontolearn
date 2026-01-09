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
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from ontolearn.utils.static_funcs import compute_f1_score
from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.learning_problem import PosNegLPStandard
from ontolearn.refinement_operators import LengthBasedRefinement
from ontolearn.learners import Drill, DrillV, OCEL, CELOE, TDL
from ontolearn.learners import EvoLearner, ALCSAT, SPELL, NERO
from ontolearn.metrics import F1
from ontolearn.heuristics import CeloeBasedReward
from owlapy.owl_individual import OWLNamedIndividual, IRI
from owlapy.render import DLSyntaxObjectRenderer

# Import DrillV variants
import sys
sys.path.insert(0, '.')
from drillv_variants import DrillV_Minimal, DrillV_Standard, DrillV_Enhanced, DrillV_Complex


class ExperimentTracker:
    """Track and store experiment results for CSV export."""
    
    def __init__(self):
        self.results = []
        
    def add_result(self, method, problem, fold, train_time, inference_time, 
                   train_f1, test_f1, concepts_tested, prediction, concept_limit=None):
        """Add a single experiment result."""
        self.results.append({
            'method': method,
            'problem': problem,
            'fold': fold,
            'concept_limit': concept_limit,
            'train_time': train_time,
            'inference_time': inference_time,
            'total_time': train_time + inference_time,
            'train_f1': train_f1,
            'test_f1': test_f1,
            'concepts_tested': concepts_tested,
            'prediction': prediction
        })
    
    def to_dataframe(self):
        """Convert results to pandas DataFrame."""
        return pd.DataFrame(self.results)
    
    def save_to_csv(self, filename):
        """Save results to CSV file."""
        df = self.to_dataframe()
        output_path = Path(filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"\n{'='*80}")
        print(f"✓ Results saved to: {output_path}")
        print(f"{'='*80}")
        return df


def run_and_time_training(learner, train_args, directory):
    start_time = time.time()
    learner.train(**train_args)
    end_time = time.time()
    learner.save(directory=directory)
    return end_time - start_time

def run_and_time_prediction(learner, train_lp, test_lp, kb, force_iter_bound=None):
    """
    Run learner and time the prediction, optionally forcing iter_bound for Drill/DrillV.
    
    Args:
        learner: The learner to run
        train_lp: Training learning problem (for backward compatibility, same as test_lp now)
        test_lp: Test learning problem (same as train_lp in simplified version)
        kb: Knowledge base
        force_iter_bound: If provided, force learner.iter_bound AND max_num_of_concepts_tested to this value
    """
    # Force iter_bound and max_num_of_concepts_tested if requested (for Drill/DrillV)
    if force_iter_bound is not None:
        if hasattr(learner, 'iter_bound'):
            learner.iter_bound = force_iter_bound
        if hasattr(learner, 'max_num_of_concepts_tested'):
            learner.max_num_of_concepts_tested = force_iter_bound
        
    dl_render = DLSyntaxObjectRenderer()
    start_time = time.time()
    
    # Different learners use different methods to get best hypothesis
    # ALCSAT, SPELL, NERO use best_hypothesis() (singular)
    # Drill, DrillV, OCEL, CELOE, TDL use best_hypotheses() (plural)
    if hasattr(learner, 'best_hypothesis'):
        pred = learner.fit(train_lp).best_hypothesis()
    else:
        pred = learner.fit(train_lp).best_hypotheses()
    
    pred_time = time.time() - start_time
    
    # Since train_lp == test_lp now, both F1 scores will be the same
    f1 = compute_f1_score(individuals=frozenset({i for i in kb.individuals(pred)}),
                          pos=train_lp.pos, neg=train_lp.neg)
    
    # Get concepts tested if available
    concepts_tested = getattr(learner, '_number_of_tested_concepts', 0)
    
    return {
        "prediction": dl_render.render(pred),
        "train_f1": f1,  # Kept for backward compatibility
        "test_f1": f1,   # Same as train_f1 now
        "prediction_time": pred_time,
        "concepts_tested": concepts_tested
    }

def start(args):
    kb = KnowledgeBase(path=args.path_knowledge_base)
    train_args = dict(
        num_of_target_concepts=args.num_of_target_concepts,
        num_learning_problems=args.num_of_training_learning_problems
    )
    
    # Initialize experiment tracker
    tracker = ExperimentTracker() if args.save_results else None

    # Determine which learners to include based on --learner_mode
    include_non_search = args.learner_mode == 'all'

    # Initialize search-based learners (always included in 'search' and 'all' modes)
    print("\nInitializing learners...")
    print(f"Mode: {args.learner_mode} ({'search-based only' if args.learner_mode == 'search' else 'all learners'})")
    
    ocel = OCEL(knowledge_base=kb, quality_func=F1(), max_runtime=args.max_runtime, max_num_of_concepts_tested=args.max_num_of_concepts_tested)
    celoe = CELOE(knowledge_base=kb, quality_func=F1(), max_runtime=args.max_runtime, max_num_of_concepts_tested=args.max_num_of_concepts_tested)
    print("  ✓ OCEL initialized (search-based)")
    print("  ✓ CELOE initialized (search-based)")
    
    # Initialize non-search-based learners only if 'all' mode
    if include_non_search:
        tdl = TDL(knowledge_base=kb, 
                  use_nominals=False,
                  kwargs_classifier={"random_state": args.random_seed},
                  max_runtime=args.max_runtime)
        
        alcsat = ALCSAT(knowledge_base=kb,
                        max_runtime=args.max_runtime,
                        max_concept_size=30)
        
        spell = SPELL(knowledge_base=kb,
                      max_runtime=args.max_runtime,
                      max_query_size=10,
                      search_mode="full_approx")
        
        nero = NERO(knowledge_base=kb,
                    num_embedding_dim=128,
                    neural_architecture='DeepSet',
                    learning_rate=0.001,
                    num_epochs=50,
                    batch_size=32)
        
        print("  ✓ TDL initialized (tree-based)")
        print("  ✓ ALCSAT initialized (SAT-based)")
        print("  ✓ SPELL initialized (SAT-based)")
        print("  ✓ NERO initialized (neural)")
    
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
        max_num_of_concepts_tested=args.max_num_of_concepts_tested,
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

    drillv = DrillVClass(
        knowledge_base=kb,
        path_embeddings=args.path_embeddings,
        refinement_operator=LengthBasedRefinement(knowledge_base=kb),
        quality_func=F1(),
        reward_func=CeloeBasedReward(),
        # epsilon_decay=args.epsilon_decay,
        termination_epsilon=args.epsilon_decay, 
        learning_rate=args.learning_rate,
        verbose=0,
        num_of_sequential_actions=args.num_of_sequential_actions,
        num_episode=args.num_episode,
        max_num_of_concepts_tested=args.max_num_of_concepts_tested,
        max_runtime=args.max_runtime
    )

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

    # time.sleep(10)  # Just to have a small break between training and testing
    # exit(0)
    # Load learning problems
    with open(args.path_learning_problem, "r", encoding="utf-8") as json_file:
        data = json.load(json_file)

    problems = data.get("problems", {})

    # Define concept limits to test
    concept_limits = [0, 50, 100, 150, 250, 350, 500, 700, 1000]
    print(f"\nTesting with concept limits: {concept_limits}")

    # Collect prediction times
    drill_times = []
    drillv_times = []
    drillv_random_times = []
    ocel_times = []
    celoe_times = []
    tdl_times = []
    alcsat_times = []
    spell_times = []
    nero_times = []
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
    ocel_train_f1s = []
    ocel_test_f1s = []
    celoe_train_f1s = []
    celoe_test_f1s = []
    tdl_train_f1s = []
    tdl_test_f1s = []
    alcsat_train_f1s = []
    alcsat_test_f1s = []
    spell_train_f1s = []
    spell_test_f1s = []
    nero_train_f1s = []
    nero_test_f1s = []
    print("\nComparing models on each class expression:\n")
    
    # Limit the number of problems if specified
    problem_items = list(problems.items())
    if args.num_problems > 0:
        problem_items = problem_items[:args.num_problems]
        print(f"Evaluating on {len(problem_items)} problems (limited by --num_problems)\n")
    else:
        print(f"Evaluating on all {len(problem_items)} problems\n")
    
    # Loop over different concept limits
    for concept_limit in concept_limits:
        print(f"\n{'#'*80}")
        print(f"# TESTING WITH CONCEPT LIMIT: {concept_limit}")
        print(f"{'#'*80}\n")
        
        # Special case: concept_limit 0 means F1 = 0 for all learners
        if concept_limit == 0:
            for str_target_concept, examples in problem_items:
                # Add zero F1 results for all learners
                if tracker:
                    for method in ['Drill', f'DrillV_{variant_name}', 'OCEL', 'CELOE']:
                        tracker.add_result(
                            method=method,
                            problem=str_target_concept,
                            fold=1,
                            concept_limit=0,
                            train_time=0.0,
                            inference_time=0.0,
                            train_f1=0.0,
                            test_f1=0.0,
                            concepts_tested=0,
                            prediction="None"
                        )
                    if include_non_search:
                        for method in ['TDL', 'ALCSAT', 'SPELL', 'NERO']:
                            tracker.add_result(
                                method=method,
                                problem=str_target_concept,
                                fold=1,
                                concept_limit=0,
                                train_time=0.0,
                                inference_time=0.0,
                                train_f1=0.0,
                                test_f1=0.0,
                                concepts_tested=0,
                                prediction="None"
                            )
            continue  # Skip to next concept limit
        
        for str_target_concept, examples in problem_items:
            p = examples['positive_examples']
            n = examples['negative_examples']
            print(f"\n{'='*80}")
            print(f"Target concept: {str_target_concept} [Limit: {concept_limit} concepts]")
            print(f"{'='*80}")
            
            # Reset V-learning agent for new LP (deletes old memory file)
            if hasattr(drillv, 'reset_for_new_lp'):
                drillv.reset_for_new_lp()

            # Use all examples (no train/test split, no folds)
            all_pos = set(map(OWLNamedIndividual, map(IRI.create, p)))
            all_neg = set(map(OWLNamedIndividual, map(IRI.create, n)))
            lp = PosNegLPStandard(pos=all_pos, neg=all_neg)

            # Drill
            drill_result = run_and_time_prediction(drill, lp, lp, kb, force_iter_bound=concept_limit)
            drill_times.append(drill_result['prediction_time'])
            drill_concepts.append(drill_result.get('concepts_tested', 0))
            drill_train_f1s.append(drill_result['train_f1'])
            drill_test_f1s.append(drill_result['test_f1'])
            
            # Track Drill results
            if tracker:
                tracker.add_result(
                    method='Drill',
                    problem=str_target_concept,
                    fold=1,
                    concept_limit=concept_limit,
                    train_time=drill_train_time,  
                    inference_time=drill_result['prediction_time'],
                    train_f1=drill_result['train_f1'],
                    test_f1=drill_result['test_f1'],
                    concepts_tested=drill_result.get('concepts_tested', 0),
                    prediction=drill_result['prediction']
                )

            # DrillV
            drillv_result = run_and_time_prediction(drillv, lp, lp, kb, force_iter_bound=concept_limit)
            drillv_times.append(drillv_result['prediction_time'])
            drillv_concepts.append(drillv_result.get('concepts_tested', 0))
            drillv_train_f1s.append(drillv_result['train_f1'])
            drillv_test_f1s.append(drillv_result['test_f1'])
        
            # Track DrillV results
            if tracker:
                tracker.add_result(
                    method=f'DrillV_{variant_name}',
                    problem=str_target_concept,
                    fold=1,
                    concept_limit=concept_limit,
                    train_time=drillv_train_time,
                    inference_time=drillv_result['prediction_time'],
                    train_f1=drillv_result['train_f1'],
                    test_f1=drillv_result['test_f1'],
                    concepts_tested=drillv_result.get('concepts_tested', 0),
                    prediction=drillv_result['prediction']
                )        # Get performance statistics after this prediction (if method exists)
        perf_stats = drillv.get_performance_stats() if hasattr(drillv, 'get_performance_stats') else None
        
        # Get V-learning stats if using DrillV_Complex with termination agent
        v_learning_stats = None
        if hasattr(drillv, 'termination_agent') and drillv.termination_agent:
            v_learning_stats = drillv.termination_agent.get_statistics()

        # Print results
        print(f"  Drill (DQN):")
        print(f"    Prediction: {drill_result['prediction']}")
        print(f"    F1: {drill_result['train_f1']:.3f}")
        print(f"    Concepts tested: {drill_result['concepts_tested']}")
        print(f"    Runtime: {drill_result['prediction_time']:.2f} seconds")
        print(f"  DrillV ({variant_name}):")
        print(f"    Prediction: {drillv_result['prediction']}")
        print(f"    F1: {drillv_result['train_f1']:.3f}")
        print(f"    Concepts tested: {drillv_result['concepts_tested']}")
        print(f"    Runtime: {drillv_result['prediction_time']:.2f} seconds")
        if perf_stats:
            print(f"    Optimization: {perf_stats['optimization_type']} (memory efficient: {perf_stats['memory_efficient']})")
        if v_learning_stats:
            print(f"    V-Learning: Total runs={v_learning_stats['total_runs']}, "
                  f"Best ever={v_learning_stats['best_ever_quality']:.3f}, "
                  f"Termination={v_learning_stats['termination_reason']}")
        
            # OCEL
            try:
                ocel_result = run_and_time_prediction(ocel, lp, lp, kb, force_iter_bound=concept_limit)
                ocel_times.append(ocel_result['prediction_time'])
                ocel_train_f1s.append(ocel_result['train_f1'])
                ocel_test_f1s.append(ocel_result['test_f1'])
                
                if tracker:
                    tracker.add_result(
                        method='OCEL',
                        problem=str_target_concept,
                        fold=1,
                        concept_limit=concept_limit,
                        train_time=0,
                        inference_time=ocel_result['prediction_time'],
                        train_f1=ocel_result['train_f1'],
                        test_f1=ocel_result['test_f1'],
                        concepts_tested=ocel_result.get('concepts_tested', 0),
                        prediction=ocel_result['prediction']
                    )
                
                print(f"  OCEL:")
                print(f"    Prediction: {ocel_result['prediction']}")
                print(f"    F1: {ocel_result['train_f1']:.3f}")
                print(f"    Concepts tested: {ocel_result.get('concepts_tested', 0)}")
                print(f"    Runtime: {ocel_result['prediction_time']:.2f} seconds")
            except Exception as e:
                print(f"  OCEL: ✗ FAILED - {str(e)}")
                if tracker:
                    tracker.add_result(
                        method='OCEL',
                        problem=str_target_concept,
                        fold=1,
                        concept_limit=concept_limit,
                        train_time=0,
                        inference_time=0,
                        train_f1=0,
                        test_f1=0,
                        concepts_tested=0,
                        prediction="ERROR"
                    )
            
            # CELOE
            try:
                celoe_result = run_and_time_prediction(celoe, lp, lp, kb, force_iter_bound=concept_limit)
                celoe_times.append(celoe_result['prediction_time'])
                celoe_train_f1s.append(celoe_result['train_f1'])
                celoe_test_f1s.append(celoe_result['test_f1'])
                
                if tracker:
                    tracker.add_result(
                        method='CELOE',
                        problem=str_target_concept,
                        fold=1,
                        concept_limit=concept_limit,
                        train_time=0,
                        inference_time=celoe_result['prediction_time'],
                        train_f1=celoe_result['train_f1'],
                        test_f1=celoe_result['test_f1'],
                        concepts_tested=celoe_result.get('concepts_tested', 0),
                        prediction=celoe_result['prediction']
                    )
                
                print(f"  CELOE:")
                print(f"    Prediction: {celoe_result['prediction']}")
                print(f"    F1: {celoe_result['train_f1']:.3f}")
                print(f"    Concepts tested: {celoe_result.get('concepts_tested', 0)}")
                print(f"    Runtime: {celoe_result['prediction_time']:.2f} seconds")
            except Exception as e:
                print(f"  CELOE: ✗ FAILED - {str(e)}")
                if tracker:
                    tracker.add_result(
                        method='CELOE',
                        problem=str_target_concept,
                        fold=1,
                        concept_limit=concept_limit,
                        train_time=0,
                        inference_time=0,
                        train_f1=0,
                        test_f1=0,
                        concepts_tested=0,
                        prediction="ERROR"
                    )
            
    # Print average prediction times
    avg_drill_time = np.mean(drill_times) if drill_times else 0
    avg_drillv_time = np.mean(drillv_times) if drillv_times else 0
    avg_ocel_time = np.mean(ocel_times) #if ocel_times else 0
    avg_celoe_time = np.mean(celoe_times) #if celoe_times else 0
    avg_tdl_time = np.mean(tdl_times) if tdl_times else 0
    avg_alcsat_time = np.mean(alcsat_times) if alcsat_times else 0
    avg_spell_time = np.mean(spell_times) if spell_times else 0
    avg_nero_time = np.mean(nero_times) if nero_times else 0
    
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print("\nAverage Runtimes:")
    print(f"  Drill (DQN):              {avg_drill_time:.3f} seconds")
    print(f"  DrillV ({variant_name}):  {avg_drillv_time:.3f} seconds")
    print(f"  OCEL:                     {avg_ocel_time:.3f} seconds")
    print(f"  CELOE:                    {avg_celoe_time:.3f} seconds")
    
    if include_non_search:
        print(f"  TDL:                      {avg_tdl_time:.3f} seconds")
        print(f"  ALCSAT:                   {avg_alcsat_time:.3f} seconds")
        print(f"  SPELL:                    {avg_spell_time:.3f} seconds")
        print(f"  NERO:                     {avg_nero_time:.3f} seconds")
    
    print("\nAverage F1 Scores:")
    print(f"  Drill (DQN):              {np.mean(drill_train_f1s):.3f}")
    print(f"  DrillV ({variant_name}):  {np.mean(drillv_train_f1s):.3f}")
    print(f"  OCEL:                     {np.mean(ocel_train_f1s):.3f}")
    print(f"  CELOE:                    {np.mean(celoe_train_f1s):.3f}")
    
    if include_non_search:
        print(f"  TDL:                      {np.mean(tdl_train_f1s):.3f}")
        print(f"  ALCSAT:                   {np.mean(alcsat_train_f1s):.3f}")
        print(f"  SPELL:                    {np.mean(spell_train_f1s):.3f}")
        print(f"  NERO:                     {np.mean(nero_train_f1s):.3f}")
    
    print("\nAverage Concepts Tested:")
    print(f"  Drill (DQN):              {np.mean(drill_concepts):.0f}")
    print(f"  DrillV ({variant_name}):  {np.mean(drillv_concepts):.0f}")
    
    print("\n" + "="*80)
    
  
    # Performance comparison: DrillV vs Drill
    print(f"\nPerformance Comparison (DrillV vs Drill):")
    if avg_drill_time > 0 and avg_drillv_time > 0:
        speedup = avg_drill_time / avg_drillv_time
        print(f"  Time: DrillV is {speedup:.2f}x {'faster' if speedup > 1.0 else 'slower'} than Drill")
    
    if drill_concepts and drillv_concepts:
        avg_drill_concepts = np.mean(drill_concepts)
        avg_drillv_concepts = np.mean(drillv_concepts)
        reduction = (avg_drill_concepts - avg_drillv_concepts) / avg_drill_concepts * 100 if avg_drill_concepts > 0 else 0.0
        print(f"  Concepts: DrillV explores {reduction:+.1f}% {'fewer' if reduction > 0 else 'more'} concepts than Drill")
    
    avg_drill_f1 = np.mean(drill_train_f1s) if drill_train_f1s else 0
    avg_drillv_f1 = np.mean(drillv_train_f1s) if drillv_train_f1s else 0
    f1_diff = avg_drillv_f1 - avg_drill_f1
    print(f"  Quality: DrillV F1 is {f1_diff:+.3f} {'better' if f1_diff > 0 else 'worse'} than Drill")
    
    print("\n" + "="*80)
    
    # Save results to CSV if requested
    if tracker and args.save_results:
        # Extract dataset name from knowledge base path (e.g., KGs/Family/... -> Family)
        kb_parts = args.path_knowledge_base.split('/')
        dataset_name = kb_parts[1] if len(kb_parts) > 1 else 'Unknown'
        
        # Extract LP filename without extension (e.g., lps_difficult.json -> lps_difficult)
        lp_parts = args.path_learning_problem.split('/')
        lp_filename = lp_parts[-1] if lp_parts else 'lps'
        lp_name = lp_filename.replace('.json', '')
        
        # Create directory name
        results_dir = f"results_{dataset_name}"
        
        # Create filename with LP name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"{results_dir}/drill_vs_drillv_{variant_name}_{lp_name}_{timestamp}.csv"
        
        df = tracker.save_to_csv(csv_filename)
        print(f"\nTo visualize results, run:")
        print(f"  python visualize_f1_checkpoints.py --input {csv_filename}")

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
                        default=20)
    parser.add_argument("--path_pretrained_dir", type=str, default=None)

    parser.add_argument("--path_learning_problem", type=str, default='LPs/Family/lps.json',
                        help="Path to a .json file that contains 2 properties 'positive_examples' and "
                             "'negative_examples'. Each of this properties should contain the IRIs of the respective"
                             "instances. e.g. 'some/path/lp.json'")
    parser.add_argument("--num_problems", type=int, default=0,
                        help="Number of problems to evaluate from the learning problem file. 0 means all problems.")
    parser.add_argument("--max_runtime", type=int, default=100, help="Max runtime")
    parser.add_argument("--folds", type=int, default=2, help="Number of folds of cross validation.")
    parser.add_argument("--learner_mode", type=str, default='search',
                        choices=['search', 'all'],
                        help="Which learners to include: 'search' (Drill, DrillV, OCEL, CELOE) or 'all' (includes TDL, ALCSAT, SPELL, NERO)")
    parser.add_argument("--random_seed", type=int, default=1)
    parser.add_argument("--iter_bound", type=int, default=None, help='iter_bound during testing.')
   
    parser.add_argument("--drill_variant", type=str, default='complex',
                        choices=['default', 'minimal', 'standard', 'enhanced', 'complex'],
                        help='DrillV variant to use: default (original with all RL features), '
                             'minimal (simplest 2-layer NN), standard (balanced 3-layer), '
                             'enhanced (standard + curriculum + curiosity), '
                             'complex (4-layer residual with target network) [default: complex]')
    parser.add_argument("--save_results", action='store_true', default=True,
                        help='Save experiment results to CSV file for visualization')
    # DQL related
    parser.add_argument("--num_episode", type=int, default=1, help='Number of trajectories created for a given lp.')
    parser.add_argument("--max_num_of_concepts_tested", type=int, default=10, help='Maximum number of concepts to be tested during learning.')
    parser.add_argument("--epsilon_decay", type=float, default=1.00, help='Epsilon greedy trade off per epoch') # Choose 0.0 for pure exploitation
    parser.add_argument("--max_len_replay_memory", type=int, default=1024,
                        help='Maximum size of the experience replay')
    parser.add_argument("--num_epochs_per_replay", type=int, default=1,
                        help='Number of epochs on experience replay memory')
    parser.add_argument('--num_of_sequential_actions', type=int, default=1, help='Length of the trajectory.')

    # NN related
    parser.add_argument("--learning_rate", type=int, default=.01)

    start(parser.parse_args())
