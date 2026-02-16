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
from ontolearn.refinement_operators import LengthBasedRefinement, PruneCELBasedRefinement
from ontolearn.learners import Drill, DrillV, OCEL, CELOE, TDL
from ontolearn.learners import EvoLearner, ALCSAT, SPELL, NERO
from ontolearn.metrics import F1
from ontolearn.heuristics import CeloeBasedReward
from owlapy.owl_individual import OWLNamedIndividual, IRI
from owlapy.render import DLSyntaxObjectRenderer
from ontolearn.vocell import VOCELL

# Import DrillV variants
import sys
sys.path.insert(0, '.')
from drillv_variants import DrillV_Minimal, DrillV_Standard, DrillV_Enhanced, DrillV_Complex
from prunecel_wrapper import PruneCELWrapper, check_prunecel_available


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
    
    # Handle PruneCEL wrapper specially (it returns a string wrapper)
    from prunecel_wrapper import PruneCELWrapper
    if isinstance(learner, PruneCELWrapper):
        # For PruneCEL, use the extracted F1 scores from its output
        # Note: PruneCEL computes F1 on training data, so we use it for both train and test
        # (test F1 would require running PruneCEL separately on test fold)

        pred_time = learner.last_runtime
        concepts_tested = learner.number_of_tested_concepts
        # return {
        #     "prediction": str(pred),
        #     "train_f1": learner._last_train_f1,
        #     "test_f1": learner._last_train_f1,  # Using train F1 as approximation
        #     "prediction_time": pred_time,
        #     "concepts_tested": getattr(learner, '_number_of_tested_concepts', 0)
        # }
    
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
    tracker = None

    # Determine which learners to include based on --learner_mode
    include_non_search = args.learner_mode == 'all'

    # Initialize search-based learners (always included in 'search' and 'all' modes)
    print("\nInitializing learners...")
    print(f"Mode: {args.learner_mode} ({'search-based only' if args.learner_mode == 'search' else 'all learners'})")
    
    ocel = OCEL(knowledge_base=kb, quality_func=F1(), max_runtime=args.max_runtime)
    celoe = CELOE(knowledge_base=kb, quality_func=F1(), max_runtime=args.max_runtime)
    print("  ✓ OCEL initialized (search-based)")
    print("  ✓ CELOE initialized (search-based)")
    
    # Initialize PruneCEL if requested and available
    prunecel = None
    if args.use_prunecel:
        if not args.prunecel_jar or not args.prunecel_sparql_url:
            print("  ⚠ PruneCEL requested but missing --prunecel_jar or --prunecel_sparql_url")
            print("    Skipping PruneCEL initialization")
        elif check_prunecel_available(args.prunecel_jar):
            try:
                prunecel = PruneCELWrapper(
                    jar_path=args.prunecel_jar,
                    sparql_url=args.prunecel_sparql_url,
                    knowledge_base=kb,
                    max_runtime=args.max_runtime,
                    recursive=args.prunecel_recursive,
                    skip_none=args.prunecel_skip_none
                )
                print(f"  ✓ PruneCEL initialized (search-based, Java)")
                print(f"    JAR: {args.prunecel_jar}")
                print(f"    SPARQL: {args.prunecel_sparql_url}")
                print(f"    Extensions: R={args.prunecel_recursive}, S={args.prunecel_skip_none}")
            except Exception as e:
                print(f"  ⚠ Failed to initialize PruneCEL: {e}")
                print("    Continuing without PruneCEL")
                prunecel = None
        else:
            print("  ⚠ PruneCEL not available. Run ./setup_prunecel.sh to install.")
            print("    Continuing without PruneCEL")
    
    # Initialize VOCELL if requested
    vocell = None
    if args.use_vocell:
        if not args.vocell_sparql_url:
            print("  ⚠ VOCELL requested but missing --vocell_sparql_url")
            print("    Skipping VOCELL initialization")
        else:
            try:
                operator = PruneCELBasedRefinement(
                    knowledge_base=kb,
                    sparql_endpoint='http://localhost:3030/family/sparql'
                    # max_concepts=100
                )
                vocell = VOCELL(
                    kb=kb,
                    operator=operator,
                    time_limit=args.max_runtime,
                    beam_width=5,
                    max_depth=20,
                    # use_negation=False,
                    # use_skip=True,
                    # max_concepts=10,
                    # use_termination=args.vocell_termination,
                    # verbose=1,
                )
                term_tag = "V-learning ON" if args.vocell_termination else "pure search"
                print(f"  ✓ VOCELL initialized (PruneCEL-S, {term_tag})")
                print(f"    SPARQL: {args.vocell_sparql_url}")
            except Exception as e:
                print(f"  ⚠ Failed to initialize VOCELL: {e}")
                print("    Continuing without VOCELL")
                vocell = None
    
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
        iter_bound=args.iter_bound,
        # max_num_of_concepts_tested=5,
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
        refinement_operator=PruneCELBasedRefinement(knowledge_base=kb, sparql_endpoint=args.vocell_sparql_url),#LengthBasedRefinement(knowledge_base=kb),
        quality_func=F1(),
        reward_func=CeloeBasedReward(),
        epsilon_decay=args.epsilon_decay,
        learning_rate=args.learning_rate,
        verbose=0,
        num_of_sequential_actions=args.num_of_sequential_actions,
        num_episode=args.num_episode,
        iter_bound=args.iter_bound,
        # max_num_of_concepts_tested=5,
        max_runtime=args.max_runtime
    )

     # Train and time both models
    # if args.path_pretrained_dir:
    #     print("Loading pretrained Drill agent...")
    #     drill.load(directory="pretrained_drill")
    #     print("Loading pretrained DrillV agent...")
    #     drillv.load(directory="pretrained_drillv")
    # else:
    #     print("Training Drill agent...")
    #     drill_train_time = run_and_time_training(drill, train_args, directory="pretrained_drill")
    #     print("Training DrillV agent...")
    #     drillv_train_time = run_and_time_training(drillv, train_args, directory="pretrained_drillv")
    #     print(f"DrillV training time: {drillv_train_time:.2f} seconds\n")
    #     print(f"Drill training time: {drill_train_time:.2f} seconds\n")
      
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
    prunecel_times = []
    tdl_times = []
    alcsat_times = []
    spell_times = []
    nero_times = []
    vocell_times = []
    # Collect concepts tested counts
    drill_concepts = []
    drillv_concepts = []
    drillv_random_concepts = []
    prunecel_concepts = []
    vocell_concepts = []
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
    prunecel_train_f1s = []
    prunecel_test_f1s = []
    tdl_train_f1s = []
    tdl_test_f1s = []
    alcsat_train_f1s = []
    alcsat_test_f1s = []
    spell_train_f1s = []
    spell_test_f1s = []
    nero_train_f1s = []
    nero_test_f1s = []
    vocell_train_f1s = []
    vocell_test_f1s = []
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
        fold_vocell_concepts = []

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
                    train_time=0,#drill_train_time,  
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
                    train_time=0,#drillv_train_time,
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
            
            # PruneCEL (if available)
            if prunecel is not None:
                prunecel_result = run_and_time_prediction(prunecel, train_lp, test_lp, kb)
                prunecel_times.append(prunecel_result['prediction_time'])
                prunecel_train_f1s.append(prunecel_result['train_f1'])
                prunecel_test_f1s.append(prunecel_result['test_f1'])
                prunecel_concepts.append(prunecel_result.get('concepts_tested', 0))
                
                if tracker:
                    tracker.add_result(
                        method='PruneCEL',
                        problem=str_target_concept,
                        fold=ith + 1,
                        train_time=0,  # PruneCEL doesn't require pre-training
                        inference_time=prunecel_result['prediction_time'],
                        train_f1=prunecel_result['train_f1'],
                        test_f1=prunecel_result['test_f1'],
                        concepts_tested=prunecel_result.get('concepts_tested', 0),
                        prediction=prunecel_result['prediction']
                    )
            
            # VOCELL (if available) — PruneCEL-S + optional V-learning
            if vocell is not None:
                # Pass lp_name so V-learning agent is shared across folds
                # of the same target concept
                dl_render = DLSyntaxObjectRenderer()
                start_time = time.time()
                best, vocell_nc = vocell.fit(train_lp, lp_name=str_target_concept, max_runtime=10)
                vocell_pred_time = time.time() - start_time
                
                if isinstance(best, tuple):
                    best = best[0]  
                    vocell_nc = best[-1]

                vocell_pred_str = dl_render.render(best) if best else "None"
                # Train F1: use the SPARQL-scored value from the search
                vocell_train_f1 = compute_f1_score(individuals=frozenset({i for i in kb.individuals(best)}),
                                pos=train_lp.pos, neg=train_lp.neg)
                # Test F1: evaluate the concept on the held-out fold
                vocell_test_f1 = compute_f1_score(individuals=frozenset({i for i in kb.individuals(best)}),
                                pos=test_lp.pos, neg=test_lp.neg)
                vocell_times.append(vocell_pred_time)
                vocell_concepts.append(vocell_nc)
                vocell_train_f1s.append(vocell_train_f1)
                vocell_test_f1s.append(vocell_test_f1)
                fold_vocell_concepts.append(vocell_nc)
                
                if tracker:
                    tracker.add_result(
                        method='VOCELL',
                        problem=str_target_concept,
                        fold=ith + 1,
                        train_time=0,  # VOCELL doesn't require pre-training
                        inference_time=vocell_pred_time,
                        train_f1=vocell_train_f1,
                        test_f1=vocell_test_f1,
                        concepts_tested=vocell_nc,
                        prediction=vocell_pred_str
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
            
            # print(f"  OCEL:")
            # print(f"    Prediction: {ocel_result['prediction']}")
            # print(f"    Train F1: {ocel_result['train_f1']:.3f} | Test F1: {ocel_result['test_f1']:.3f}")
            # print(f"    Prediction time: {ocel_result['prediction_time']:.2f} seconds")
            # print(f"  CELOE:")
            # print(f"    Prediction: {celoe_result['prediction']}")
            # print(f"    Train F1: {celoe_result['train_f1']:.3f} | Test F1: {celoe_result['test_f1']:.3f}")
            # print(f"    Prediction time: {celoe_result['prediction_time']:.2f} seconds")
            
            if prunecel is not None:
                print(f"  PruneCEL:")
                print(f"    Prediction: {prunecel_result['prediction']}")
                print(f"    Train F1: {prunecel_result['train_f1']:.3f} | Test F1: {prunecel_result['test_f1']:.3f}")
                print(f"    Concepts tested: {prunecel_result['concepts_tested']}")
                print(f"    Prediction time: {prunecel_result['prediction_time']:.2f} seconds")
            
            if vocell is not None:
                # early_tag = " [EARLY STOP]" if vocell.stopped_early else ""
                # print(f"  VOCELL:{early_tag}")
                print(f"    Prediction: {vocell_pred_str}")
                print(f"    Train F1: {vocell_train_f1:.3f} | Test F1: {vocell_test_f1:.3f}")
                print(f"    Concepts scored: {vocell_nc}")
                print(f"    Prediction time: {vocell_pred_time:.2f} seconds")
            
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

        # # VOCELL V-learning trend
        # if vocell is not None and len(fold_vocell_concepts) > 1:
        #     print(f"\nVOCELL V-Learning Analysis for '{str_target_concept}':")
        #     print(f"Concepts per fold: {fold_vocell_concepts}")
        #     first_fold = fold_vocell_concepts[0]
        #     last_fold = fold_vocell_concepts[-1]
        #     if first_fold > 0:
        #         improvement = ((first_fold - last_fold) / first_fold) * 100
        #         print(f"Learning trend: {first_fold} → {last_fold} ({improvement:+.1f}% efficiency)")
        #         if vocell.termination_agent:
        #             stats = vocell.termination_agent.get_statistics()
        #             print(f"Agent runs={stats['total_runs']}, "
        #                   f"best_ever={stats['best_ever_quality']:.3f}")


    # Print average prediction times
    avg_drill_time = np.mean(drill_times) if drill_times else 0
    avg_drillv_time = np.mean(drillv_times) if drillv_times else 0
    avg_ocel_time = np.mean(ocel_times) if ocel_times else 0
    avg_celoe_time = np.mean(celoe_times) if celoe_times else 0
    avg_prunecel_time = np.mean(prunecel_times) if prunecel_times else 0
    avg_tdl_time = np.mean(tdl_times) if tdl_times else 0
    avg_alcsat_time = np.mean(alcsat_times) if alcsat_times else 0
    avg_spell_time = np.mean(spell_times) if spell_times else 0
    avg_nero_time = np.mean(nero_times) if nero_times else 0
    avg_vocell_time = np.mean(vocell_times) if vocell_times else 0
    
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print("\nAverage Prediction Times:")
    print(f"  Drill (DQN):              {avg_drill_time:.3f} seconds")
    print(f"  DrillV ({variant_name}):  {avg_drillv_time:.3f} seconds")
    
    print(f"  OCEL:                     {avg_ocel_time:.3f} seconds")
    print(f"  CELOE:                    {avg_celoe_time:.3f} seconds")
    if prunecel is not None:
        print(f"  PruneCEL:                 {avg_prunecel_time:.3f} seconds")
    if vocell is not None:
        print(f"  VOCELL:                   {avg_vocell_time:.3f} seconds")
    
    if include_non_search:
        print(f"  TDL:                      {avg_tdl_time:.3f} seconds")
        print(f"  ALCSAT:                   {avg_alcsat_time:.3f} seconds")
        print(f"  SPELL:                    {avg_spell_time:.3f} seconds")
        print(f"  NERO:                     {avg_nero_time:.3f} seconds")
    
    print("\nAverage Test F1 Scores:")
    print(f"  Drill (DQN):              {np.mean(drill_test_f1s):.3f}")
    print(f"  DrillV ({variant_name}):  {np.mean(drillv_test_f1s):.3f}")
    
    print(f"  OCEL:                     {np.mean(ocel_test_f1s):.3f}")
    print(f"  CELOE:                    {np.mean(celoe_test_f1s):.3f}")
    if prunecel is not None:
        print(f"  PruneCEL:                 {np.mean(prunecel_test_f1s):.3f}")
    if vocell is not None and vocell_test_f1s:
        print(f"  VOCELL:                   {np.mean(vocell_test_f1s):.3f}")
    
    if include_non_search:
        print(f"  TDL:                      {np.mean(tdl_test_f1s):.3f}")
        print(f"  ALCSAT:                   {np.mean(alcsat_test_f1s):.3f}")
        print(f"  SPELL:                    {np.mean(spell_test_f1s):.3f}")
        print(f"  NERO:                     {np.mean(nero_test_f1s):.3f}")
    
    print("\nAverage Concepts Tested:")
    print(f"  Drill (DQN):              {np.mean(drill_concepts):.0f}")
    print(f"  DrillV ({variant_name}):  {np.mean(drillv_concepts):.0f}")
    if prunecel is not None and prunecel_concepts:
        print(f"  PruneCEL:                 {np.mean(prunecel_concepts):.0f}")
    if vocell is not None and vocell_concepts:
        print(f"  VOCELL:                   {np.mean(vocell_concepts):.0f}")
    
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
    
    avg_drill_test_f1 = np.mean(drill_test_f1s) if drill_test_f1s else 0
    avg_drillv_test_f1 = np.mean(drillv_test_f1s) if drillv_test_f1s else 0
    f1_diff = avg_drillv_test_f1 - avg_drill_test_f1
    print(f"  Quality: DrillV test F1 is {f1_diff:+.3f} {'better' if f1_diff > 0 else 'worse'} than Drill")
    
    print("\n" + "="*80)
    
    # Save results to CSV if requested
    if tracker and args.save_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"results/drill_vs_drillv_{variant_name}_{timestamp}_{args.max_runtime}.csv"
        df = tracker.save_to_csv(csv_filename)
        print(f"\nTo visualize results, run:")
        print(f"  python visualize_drill_drillv_evolution.py --input {csv_filename}")

if __name__ == '__main__':
    parser = ArgumentParser()
    # General
    parser.add_argument("--path_knowledge_base", type=str,
                        default='KGs/Family/family.owl')
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
    parser.add_argument("--num_problems", type=int, default=5,
                        help="Number of problems to evaluate from the learning problem file. 0 means all problems.")
    parser.add_argument("--max_runtime", type=int, default=10, help="Max runtime")
    parser.add_argument("--folds", type=int, default=2, help="Number of folds of cross validation.")
    parser.add_argument("--learner_mode", type=str, default='search',
                        choices=['search', 'all'],
                        help="Which learners to include: 'search' (Drill, DrillV, OCEL, CELOE) or 'all' (includes TDL, ALCSAT, SPELL, NERO)")
    parser.add_argument("--random_seed", type=int, default=1)
    parser.add_argument("--iter_bound", type=int, default=10_000, help='iter_bound during testing.')
   
    parser.add_argument("--drill_variant", type=str, default='complex',
                        choices=['default', 'minimal', 'standard', 'enhanced', 'complex'],
                        help='DrillV variant to use: default (original with all RL features), '
                             'minimal (simplest 2-layer NN), standard (balanced 3-layer), '
                             'enhanced (standard + curriculum + curiosity), '
                             'complex (4-layer residual with target network) [default: complex]')
    parser.add_argument("--save_results", action='store_true', default=True,
                        help='Save experiment results to CSV file for visualization')
    
    # PruneCEL related
    parser.add_argument("--use_prunecel", action='store_true', default=False,
                        help='Include PruneCEL as a baseline (requires Java and compiled JAR)')
    parser.add_argument("--prunecel_jar", type=str, default="PruneCEL/target/prune-cel-0.0.1-SNAPSHOT.jar",
                        help='Path to compiled PruneCEL JAR file')
    parser.add_argument("--prunecel_sparql_url", type=str, default="http://localhost:3030/family/sparql",
                        help='SPARQL endpoint URL with loaded knowledge base')
    parser.add_argument("--prunecel_recursive", action='store_true', default=True,
                        help='Enable PruneCEL recursive extension (-R)')
    parser.add_argument("--prunecel_skip_none", action='store_true', default=True,
                        help='Enable PruneCEL skip-none extension (-S)')
    
    # VOCELL related
    parser.add_argument("--use_vocell", action='store_true', default=True,
                        help='Include VOCELL')
    parser.add_argument("--vocell_sparql_url", type=str, default="http://localhost:3030/family/sparql",
                        help='SPARQL endpoint URL for VOCELL')
    parser.add_argument("--vocell_termination", action='store_true', default=False,
                        help='Enable V-learning termination agent in VOCELL')
    parser.add_argument("--no_vocell_termination", action='store_false', dest='vocell_termination',
                        help='Disable V-learning termination agent in VOCELL (pure PruneCEL-S search)')
    
    # DQL related
    parser.add_argument("--num_episode", type=int, default=1, help='Number of trajectories created for a given lp.')

    parser.add_argument("--epsilon_decay", type=float, default=1.00, help='Epsilon greedy trade off per epoch') # Choose 0.0 for pure exploitation
    parser.add_argument("--max_len_replay_memory", type=int, default=1024,
                        help='Maximum size of the experience replay')
    parser.add_argument("--num_epochs_per_replay", type=int, default=1,
                        help='Number of epochs on experience replay memory')
    parser.add_argument('--num_of_sequential_actions', type=int, default=1, help='Length of the trajectory.')

    # NN related
    parser.add_argument("--learning_rate", type=int, default=.01)

    start(parser.parse_args())
