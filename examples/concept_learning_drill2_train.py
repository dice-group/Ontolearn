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
                   train_f1, test_f1, concepts_tested, prediction):
        """Add a single experiment result."""
        self.results.append({
            'method': method,
            'problem': problem,
            'fold': fold,
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

def run_and_time_prediction(learner, train_lp, test_lp, kb):
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
    
    # Initialize experiment tracker
    tracker = ExperimentTracker() if args.save_results else None

    # Determine which learners to include based on --learner_mode
    include_search_based = args.learner_mode in ['search', 'all']
    include_non_search = args.learner_mode == 'all'

    # Initialize search-based learners (always included in 'search' and 'all' modes)
    print("\nInitializing learners...")
    print(f"Mode: {args.learner_mode} ({'search-based only' if args.learner_mode == 'search' else 'all learners'})")
    
    ocel = OCEL(knowledge_base=kb, quality_func=F1(), max_runtime=args.max_runtime)
    celoe = CELOE(knowledge_base=kb, quality_func=F1(), max_runtime=args.max_runtime)
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
    else:
        tdl = alcsat = spell = nero = None
        print("  ℹ Non-search-based learners excluded (use --learner_mode all to include)")

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
            
            # Track Drill results
            if tracker:
                tracker.add_result(
                    method='Drill',
                    problem=str_target_concept,
                    fold=ith + 1,
                    train_time=drill_train_time,  
                    inference_time=drill_result['prediction_time'],
                    train_f1=drill_result['train_f1'],
                    test_f1=drill_result['test_f1'],
                    concepts_tested=drill_result.get('concepts_tested', 0),
                    prediction=drill_result['prediction']
                )

            # DrillV with trained V-values
            drillv_result = run_and_time_prediction(drillv, train_lp, test_lp, kb)
            drillv_times.append(drillv_result['prediction_time'])
            drillv_concepts.append(drillv_result.get('concepts_tested', 0))
            drillv_train_f1s.append(drillv_result['train_f1'])
            drillv_test_f1s.append(drillv_result['test_f1'])
            fold_drillv_concepts.append(drillv_result.get('concepts_tested', 0))
            
            # Track DrillV results
            if tracker:
                tracker.add_result(
                    method=f'DrillV_{variant_name}',
                    problem=str_target_concept,
                    fold=ith + 1,
                    train_time=drillv_train_time,
                    inference_time=drillv_result['prediction_time'],
                    train_f1=drillv_result['train_f1'],
                    test_f1=drillv_result['test_f1'],
                    concepts_tested=drillv_result.get('concepts_tested', 0),
                    prediction=drillv_result['prediction']
                )
            
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
                print(f"    V-Learning: Total runs={v_learning_stats['total_runs']}, "
                      f"Best ever={v_learning_stats['best_ever_quality']:.3f}, "
                      f"Termination={v_learning_stats['termination_reason']}")
            
            # OCEL
            ocel_result = run_and_time_prediction(ocel, train_lp, test_lp, kb)
            ocel_times.append(ocel_result['prediction_time'])
            ocel_train_f1s.append(ocel_result['train_f1'])
            ocel_test_f1s.append(ocel_result['test_f1'])
            
            if tracker:
                tracker.add_result(
                    method='OCEL',
                    problem=str_target_concept,
                    fold=ith + 1,
                    train_time=0,  # OCEL doesn't require pre-training
                    inference_time=ocel_result['prediction_time'],
                    train_f1=ocel_result['train_f1'],
                    test_f1=ocel_result['test_f1'],
                    concepts_tested=ocel_result.get('concepts_tested', 0),
                    prediction=ocel_result['prediction']
                )
            
            # CELOE
            celoe_result = run_and_time_prediction(celoe, train_lp, test_lp, kb)
            celoe_times.append(celoe_result['prediction_time'])
            celoe_train_f1s.append(celoe_result['train_f1'])
            celoe_test_f1s.append(celoe_result['test_f1'])
            
            if tracker:
                tracker.add_result(
                    method='CELOE',
                    problem=str_target_concept,
                    fold=ith + 1,
                    train_time=0,  # CELOE doesn't require pre-training
                    inference_time=celoe_result['prediction_time'],
                    train_f1=celoe_result['train_f1'],
                    test_f1=celoe_result['test_f1'],
                    concepts_tested=celoe_result.get('concepts_tested', 0),
                    prediction=celoe_result['prediction']
                )
            
            # Non-search-based learners (only if include_non_search is True)
            if include_non_search:
                # TDL
                tdl_result = run_and_time_prediction(tdl, train_lp, test_lp, kb)
                tdl_times.append(tdl_result['prediction_time'])
                tdl_train_f1s.append(tdl_result['train_f1'])
                tdl_test_f1s.append(tdl_result['test_f1'])
                
                if tracker:
                    tracker.add_result(
                        method='TDL',
                        problem=str_target_concept,
                        fold=ith + 1,
                        train_time=0,
                        inference_time=tdl_result['prediction_time'],
                        train_f1=tdl_result['train_f1'],
                        test_f1=tdl_result['test_f1'],
                        concepts_tested=tdl_result.get('concepts_tested', 0),
                        prediction=tdl_result['prediction']
                    )
                
                # ALCSAT
                alcsat_result = run_and_time_prediction(alcsat, train_lp, test_lp, kb)
                alcsat_times.append(alcsat_result['prediction_time'])
                alcsat_train_f1s.append(alcsat_result['train_f1'])
                alcsat_test_f1s.append(alcsat_result['test_f1'])
                
                if tracker:
                    tracker.add_result(
                        method='ALCSAT',
                        problem=str_target_concept,
                        fold=ith + 1,
                        train_time=0,
                        inference_time=alcsat_result['prediction_time'],
                        train_f1=alcsat_result['train_f1'],
                        test_f1=alcsat_result['test_f1'],
                        concepts_tested=alcsat_result.get('concepts_tested', 0),
                        prediction=alcsat_result['prediction']
                    )
                
                # SPELL
                spell_result = run_and_time_prediction(spell, train_lp, test_lp, kb)
                spell_times.append(spell_result['prediction_time'])
                spell_train_f1s.append(spell_result['train_f1'])
                spell_test_f1s.append(spell_result['test_f1'])
                
                if tracker:
                    tracker.add_result(
                        method='SPELL',
                        problem=str_target_concept,
                        fold=ith + 1,
                        train_time=0,
                        inference_time=spell_result['prediction_time'],
                        train_f1=spell_result['train_f1'],
                        test_f1=spell_result['test_f1'],
                        concepts_tested=spell_result.get('concepts_tested', 0),
                        prediction=spell_result['prediction']
                    )
                
                # NERO (requires namespace setup)
                a_prop = list(kb.ontology.object_properties_in_signature())[:1].pop()
                ns = a_prop.iri.get_namespace()
                nero.ns = ns
                nero_result = run_and_time_prediction(nero, train_lp, test_lp, kb)
                nero_times.append(nero_result['prediction_time'])
                nero_train_f1s.append(nero_result['train_f1'])
                nero_test_f1s.append(nero_result['test_f1'])
                
                if tracker:
                    tracker.add_result(
                        method='NERO',
                        problem=str_target_concept,
                        fold=ith + 1,
                        train_time=0,
                        inference_time=nero_result['prediction_time'],
                        train_f1=nero_result['train_f1'],
                        test_f1=nero_result['test_f1'],
                        concepts_tested=nero_result.get('concepts_tested', 0),
                        prediction=nero_result['prediction']
                    )
            
            print(f"  OCEL:")
            print(f"    Prediction: {ocel_result['prediction']}")
            print(f"    Train F1: {ocel_result['train_f1']:.3f} | Test F1: {ocel_result['test_f1']:.3f}")
            print(f"    Prediction time: {ocel_result['prediction_time']:.2f} seconds")
            print(f"  CELOE:")
            print(f"    Prediction: {celoe_result['prediction']}")
            print(f"    Train F1: {celoe_result['train_f1']:.3f} | Test F1: {celoe_result['test_f1']:.3f}")
            print(f"    Prediction time: {celoe_result['prediction_time']:.2f} seconds")
            
            if include_non_search:
                print(f"  TDL:")
                print(f"    Prediction: {tdl_result['prediction']}")
                print(f"    Train F1: {tdl_result['train_f1']:.3f} | Test F1: {tdl_result['test_f1']:.3f}")
                print(f"    Prediction time: {tdl_result['prediction_time']:.2f} seconds")
                print(f"  ALCSAT:")
                print(f"    Prediction: {alcsat_result['prediction']}")
                print(f"    Train F1: {alcsat_result['train_f1']:.3f} | Test F1: {alcsat_result['test_f1']:.3f}")
                print(f"    Prediction time: {alcsat_result['prediction_time']:.2f} seconds")
                print(f"  SPELL:")
                print(f"    Prediction: {spell_result['prediction']}")
                print(f"    Train F1: {spell_result['train_f1']:.3f} | Test F1: {spell_result['test_f1']:.3f}")
                print(f"    Prediction time: {spell_result['prediction_time']:.2f} seconds")
                print(f"  NERO:")
                print(f"    Prediction: {nero_result['prediction']}")
                print(f"    Train F1: {nero_result['train_f1']:.3f} | Test F1: {nero_result['test_f1']:.3f}")
                print(f"    Prediction time: {nero_result['prediction_time']:.2f} seconds")
            
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
    avg_ocel_time = np.mean(ocel_times) if ocel_times else 0
    avg_celoe_time = np.mean(celoe_times) if celoe_times else 0
    avg_tdl_time = np.mean(tdl_times) if tdl_times else 0
    avg_alcsat_time = np.mean(alcsat_times) if alcsat_times else 0
    avg_spell_time = np.mean(spell_times) if spell_times else 0
    avg_nero_time = np.mean(nero_times) if nero_times else 0
    
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print("\nAverage Prediction Times:")
    print(f"  Drill (DQN):              {avg_drill_time:.3f} seconds")
    print(f"  DrillV ({variant_name}):  {avg_drillv_time:.3f} seconds")
    if args.compare_random_v:
        print(f"  DrillV (random V):        {avg_drillv_random_time:.3f} seconds")
    print(f"  OCEL:                     {avg_ocel_time:.3f} seconds")
    print(f"  CELOE:                    {avg_celoe_time:.3f} seconds")
    
    if include_non_search:
        print(f"  TDL:                      {avg_tdl_time:.3f} seconds")
        print(f"  ALCSAT:                   {avg_alcsat_time:.3f} seconds")
        print(f"  SPELL:                    {avg_spell_time:.3f} seconds")
        print(f"  NERO:                     {avg_nero_time:.3f} seconds")
    
    print("\nAverage Test F1 Scores:")
    print(f"  Drill (DQN):              {np.mean(drill_test_f1s):.3f}")
    print(f"  DrillV ({variant_name}):  {np.mean(drillv_test_f1s):.3f}")
    if args.compare_random_v:
        print(f"  DrillV (random V):        {np.mean(drillv_random_test_f1s):.3f}")
    print(f"  OCEL:                     {np.mean(ocel_test_f1s):.3f}")
    print(f"  CELOE:                    {np.mean(celoe_test_f1s):.3f}")
    
    if include_non_search:
        print(f"  TDL:                      {np.mean(tdl_test_f1s):.3f}")
        print(f"  ALCSAT:                   {np.mean(alcsat_test_f1s):.3f}")
        print(f"  SPELL:                    {np.mean(spell_test_f1s):.3f}")
        print(f"  NERO:                     {np.mean(nero_test_f1s):.3f}")
    
    print("\nAverage Concepts Tested:")
    print(f"  Drill (DQN):              {np.mean(drill_concepts):.0f}")
    print(f"  DrillV ({variant_name}):  {np.mean(drillv_concepts):.0f}")
    if args.compare_random_v:
        print(f"  DrillV (random V):        {np.mean(drillv_random_concepts):.0f}")
    
    print("\n" + "="*80)
    
    # Show DrillV performance optimizations statistics (if method exists)
    if hasattr(drillv, 'print_performance_summary'):
        print("\nDrillV Lean Performance Optimizations:")
        drillv.print_performance_summary()
    
    if args.compare_random_v:
        print(f"\nV-Values Importance Analysis:")
        if avg_drillv_time > 0 and avg_drillv_random_time > 0:
            time_diff_percent = ((avg_drillv_random_time - avg_drillv_time) / avg_drillv_time) * 100
            print(f"  Random V-values vs Trained V-values time difference: {time_diff_percent:+.1f}%")
            if abs(time_diff_percent) < 5:
                print("  → V-values have minimal impact on prediction time")
            elif time_diff_percent > 0:
                print("  → Random V-values are slower (less efficient search)")
            else:
                print("  → Random V-values are faster (but potentially less accurate)")
        if hasattr(drillv_random, 'print_performance_summary'):
            print("\nDrillV (Random V-values) Lean Performance Optimizations:")
            drillv_random.print_performance_summary()
                
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
    
    avg_drill_test_f1 = np.mean(drill_test_f1s) if drill_test_f1s else 0
    avg_drillv_test_f1 = np.mean(drillv_test_f1s) if drillv_test_f1s else 0
    f1_diff = avg_drillv_test_f1 - avg_drill_test_f1
    print(f"  Quality: DrillV test F1 is {f1_diff:+.3f} {'better' if f1_diff > 0 else 'worse'} than Drill")
    
    print("\n" + "="*80)
    
    # Save results to CSV if requested
    if tracker and args.save_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"results/drill_vs_drillv_{variant_name}_{timestamp}.csv"
        df = tracker.save_to_csv(csv_filename)
        print(f"\nTo visualize results, run:")
        print(f"  python visualize_drill_drillv_evolution.py --input {csv_filename}")

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

    parser.add_argument("--path_learning_problem", type=str, default='LPs/Family/lps_difficult.json',
                        help="Path to a .json file that contains 2 properties 'positive_examples' and "
                             "'negative_examples'. Each of this properties should contain the IRIs of the respective"
                             "instances. e.g. 'some/path/lp.json'")
    parser.add_argument("--num_problems", type=int, default=2,
                        help="Number of problems to evaluate from the learning problem file. 0 means all problems.")
    parser.add_argument("--max_runtime", type=int, default=10, help="Max runtime")
    parser.add_argument("--folds", type=int, default=5, help="Number of folds of cross validation.")
    parser.add_argument("--learner_mode", type=str, default='search',
                        choices=['search', 'all'],
                        help="Which learners to include: 'search' (Drill, DrillV, OCEL, CELOE) or 'all' (includes TDL, ALCSAT, SPELL, NERO)")
    parser.add_argument("--random_seed", type=int, default=1)
    parser.add_argument("--iter_bound", type=int, default=10_000, help='iter_bound during testing.')
    parser.add_argument("--compare_random_v", action='store_true', 
                        help='Include comparison with random V-values to study V-values importance')
    parser.add_argument("--enable_drillv_optimizations", action='store_true', default=False,
                        help='Enable DrillV performance optimizations (caching, batch processing)')
    
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

    parser.add_argument("--epsilon_decay", type=float, default=0.00, help='Epsilon greedy trade off per epoch') # Choose 0.0 for pure exploitation
    parser.add_argument("--max_len_replay_memory", type=int, default=1024,
                        help='Maximum size of the experience replay')
    parser.add_argument("--num_epochs_per_replay", type=int, default=1,
                        help='Number of epochs on experience replay memory')
    parser.add_argument('--num_of_sequential_actions', type=int, default=1, help='Length of the trajectory.')

    # NN related
    parser.add_argument("--learning_rate", type=int, default=.01)

    start(parser.parse_args())
