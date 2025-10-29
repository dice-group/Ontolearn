#!/usr/bin/env python3
"""
Convergence Analysis: V-Learning vs Q-Learning vs Random V-Values
================================================================

This script compares how quickly different learning approaches converge to 
optimal concept solutions on the Family dataset.

Comparison Methods:
1. Drill (Q-Learning) - Traditional Deep Q-Learning
2. DrillV (V-Learning) - Deep V-Learning with trained values  
3. DrillV (Random V) - Deep V-Learning with random values

Metrics Analyzed:
- Time to first good solution (quality > 0.8)
- Time to optimal solution (quality = 1.0)
- Quality progression over time
- Number of concepts tested over time
- Search efficiency

Visualization:
- Quality vs Time plots
- Concepts tested vs Time plots  
- Convergence rate comparison
- Success rate analysis
"""

import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from argparse import ArgumentParser
from sklearn.model_selection import StratifiedKFold
from collections import defaultdict
import sys
import os

# Add path to access ontolearn
sys.path.append('/home/dice/Downloads/Ontolearn')

from ontolearn.utils.static_funcs import compute_f1_score
from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.learning_problem import PosNegLPStandard
from ontolearn.refinement_operators import LengthBasedRefinement
from ontolearn.learners import Drill, DrillV
from ontolearn.metrics import F1
from ontolearn.heuristics import CeloeBasedReward
from owlapy.owl_individual import OWLNamedIndividual, IRI
from owlapy.render import DLSyntaxObjectRenderer

class ConvergenceTracker:
    """Track convergence metrics during learning."""
    
    def __init__(self, learner_name):
        self.learner_name = learner_name
        self.start_time = None
        self.quality_timeline = []
        self.time_timeline = []
        self.concepts_tested_timeline = []
        self.first_good_solution_time = None  # Time to quality > 0.8
        self.optimal_solution_time = None     # Time to quality = 1.0
        self.final_quality = 0.0
        self.total_concepts_tested = 0
        
    def start_tracking(self):
        """Start tracking convergence."""
        self.start_time = time.time()
        self.quality_timeline = []
        self.time_timeline = []
        self.concepts_tested_timeline = []
        self.first_good_solution_time = None
        self.optimal_solution_time = None
        
    def update(self, current_quality, concepts_tested, real_time_quality=None):
        """Update tracking with current metrics."""
        if self.start_time is None:
            return
            
        current_time = time.time() - self.start_time
        
        # Record timeline
        self.time_timeline.append(current_time)
        self.quality_timeline.append(current_quality)  # Best quality found so far
        self.concepts_tested_timeline.append(concepts_tested)
        
        # Check for milestones
        if current_quality > 0.8 and self.first_good_solution_time is None:
            self.first_good_solution_time = current_time
            
        if current_quality >= 1.0 and self.optimal_solution_time is None:
            self.optimal_solution_time = current_time
            
    def finish_tracking(self, final_quality, total_concepts):
        """Finish tracking and record final metrics."""
        self.final_quality = final_quality
        self.total_concepts_tested = total_concepts
        
    def get_summary(self):
        """Get summary statistics."""
        return {
            'learner': self.learner_name,
            'first_good_time': self.first_good_solution_time,
            'optimal_time': self.optimal_solution_time,
            'final_quality': self.final_quality,
            'total_concepts': self.total_concepts_tested,
            'efficiency': self.final_quality / max(self.total_concepts_tested, 1)
        }

class TrackingDrill(Drill):
    """Drill with convergence tracking."""
    
    def __init__(self, *args, tracker=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.tracker = tracker
        
    def fit(self, learning_problem, max_runtime=None):
        if self.tracker:
            self.tracker.start_tracking()
        
        # Call parent fit with custom tracking
        result = super().fit(learning_problem, max_runtime)
        
        if self.tracker:
            best_hyp = self.best_hypotheses(n=1, return_node=True)
            final_quality = best_hyp.quality if best_hyp else 0.0
            self.tracker.finish_tracking(final_quality, self._number_of_tested_concepts)
            
        return result
    
    def update_search(self, next_states, values):
        """Override to track progress."""
        result = super().update_search(next_states, values)
        
        # Track current best quality and also current iteration quality
        if self.tracker and hasattr(self, '_number_of_tested_concepts'):
            best_hyp = self.best_hypotheses(n=1, return_node=True)
            current_best_quality = best_hyp.quality if best_hyp else 0.0
            
            # Get quality of current states being evaluated
            current_qualities = [state.quality for state in next_states if state.quality is not None]
            current_iter_quality = max(current_qualities) if current_qualities else 0.0
            
            self.tracker.update(current_best_quality, self._number_of_tested_concepts, current_iter_quality)
            
        return result

class TrackingDrillV(DrillV):
    """DrillV with convergence tracking."""
    
    def __init__(self, *args, tracker=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.tracker = tracker
        
    def fit(self, learning_problem, max_runtime=None):
        if self.tracker:
            self.tracker.start_tracking()
        
        # Call parent fit with custom tracking
        result = super().fit(learning_problem, max_runtime)
        
        if self.tracker:
            best_hyp = self.best_hypotheses(n=1, return_node=True)
            final_quality = best_hyp.quality if best_hyp else 0.0
            self.tracker.finish_tracking(final_quality, self._number_of_tested_concepts)
            
        return result
    
    def update_search(self, next_states, values):
        """Override to track progress."""
        result = super().update_search(next_states, values)
        
        # Track current best quality and also current iteration quality
        if self.tracker and hasattr(self, '_number_of_tested_concepts'):
            best_hyp = self.best_hypotheses(n=1, return_node=True)
            current_best_quality = best_hyp.quality if best_hyp else 0.0
            
            # Get quality of current states being evaluated
            current_qualities = [state.quality for state in next_states if state.quality is not None]
            current_iter_quality = max(current_qualities) if current_qualities else 0.0
            
            self.tracker.update(current_best_quality, self._number_of_tested_concepts, current_iter_quality)
            
        return result

def run_convergence_experiment(learner, tracker, train_lp, max_runtime=30):
    """Run a single convergence experiment."""
    print(f"Running {tracker.learner_name} convergence experiment...")
    
    start_time = time.time()
    try:
        result = learner.fit(train_lp, max_runtime=max_runtime)
        runtime = time.time() - start_time
        
        best_hyp = learner.best_hypotheses(n=1, return_node=True)
        final_quality = best_hyp.quality if best_hyp else 0.0
        
        return {
            'success': True,
            'runtime': runtime,
            'final_quality': final_quality,
            'concepts_tested': learner._number_of_tested_concepts,
            'tracker': tracker
        }
    except Exception as e:
        print(f"Error in {tracker.learner_name}: {e}")
        return {
            'success': False,
            'error': str(e),
            'tracker': tracker
        }

def create_convergence_plots(trackers, save_dir="convergence_plots"):
    """Create visualization plots for convergence analysis."""
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Plot 1: Quality vs Time
    plt.figure(figsize=(12, 8))
    
    for tracker in trackers:
        if tracker.time_timeline and tracker.quality_timeline:
            plt.plot(tracker.time_timeline, tracker.quality_timeline, 
                    linewidth=2, marker='o', markersize=4, alpha=0.8,
                    label=f'{tracker.learner_name} (Final: {tracker.final_quality:.3f})')
    
    plt.xlabel('Time (seconds)', fontsize=12)
    plt.ylabel('Best Quality Found', fontsize=12)
    plt.title('Convergence Analysis: Quality vs Time\nFamily Dataset (30s runtime)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.05)
    
    # Add milestone lines
    plt.axhline(y=0.8, color='orange', linestyle='--', alpha=0.7, label='Good Solution (0.8)')
    plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Optimal Solution (1.0)')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/quality_vs_time.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{save_dir}/quality_vs_time.pdf', bbox_inches='tight')
    print(f"Saved: {save_dir}/quality_vs_time.png")
    
    # Plot 2: Concepts Tested vs Time
    plt.figure(figsize=(12, 8))
    
    for tracker in trackers:
        if tracker.time_timeline and tracker.concepts_tested_timeline:
            plt.plot(tracker.time_timeline, tracker.concepts_tested_timeline,
                    linewidth=2, marker='s', markersize=4, alpha=0.8,
                    label=f'{tracker.learner_name} (Total: {tracker.total_concepts_tested})')
    
    plt.xlabel('Time (seconds)', fontsize=12)
    plt.ylabel('Number of Concepts Tested', fontsize=12)
    plt.title('Search Efficiency: Concepts Tested vs Time\nFamily Dataset (30s runtime)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/concepts_vs_time.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{save_dir}/concepts_vs_time.pdf', bbox_inches='tight')
    print(f"Saved: {save_dir}/concepts_vs_time.png")
    
    # Plot 3: Convergence Rate Comparison
    plt.figure(figsize=(10, 6))
    
    learners = []
    first_good_times = []
    optimal_times = []
    final_qualities = []
    efficiencies = []
    
    for tracker in trackers:
        summary = tracker.get_summary()
        learners.append(summary['learner'])
        first_good_times.append(summary['first_good_time'] or 30)  # Use max time if not reached
        optimal_times.append(summary['optimal_time'] or 30)
        final_qualities.append(summary['final_quality'])
        efficiencies.append(summary['efficiency'])
    
    x = np.arange(len(learners))
    width = 0.35
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Convergence times
    ax1.bar(x - width/2, first_good_times, width, label='Time to Good Solution (>0.8)', alpha=0.8)
    ax1.bar(x + width/2, optimal_times, width, label='Time to Optimal Solution (1.0)', alpha=0.8)
    
    ax1.set_xlabel('Learning Method', fontsize=12)
    ax1.set_ylabel('Time (seconds)', fontsize=12)
    ax1.set_title('Convergence Speed Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(learners, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Final quality and efficiency
    ax2_twin = ax2.twinx()
    
    bars1 = ax2.bar(x - width/2, final_qualities, width, label='Final Quality', alpha=0.8, color='green')
    bars2 = ax2_twin.bar(x + width/2, efficiencies, width, label='Efficiency (Quality/Concepts)', alpha=0.8, color='orange')
    
    ax2.set_xlabel('Learning Method', fontsize=12)
    ax2.set_ylabel('Final Quality', fontsize=12, color='green')
    ax2_twin.set_ylabel('Efficiency', fontsize=12, color='orange')
    ax2.set_title('Solution Quality and Efficiency', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(learners, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (qual, eff) in enumerate(zip(final_qualities, efficiencies)):
        ax2.text(i - width/2, qual + 0.02, f'{qual:.3f}', ha='center', va='bottom', fontweight='bold')
        ax2_twin.text(i + width/2, eff + 0.0001, f'{eff:.4f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/convergence_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{save_dir}/convergence_comparison.pdf', bbox_inches='tight')
    print(f"Saved: {save_dir}/convergence_comparison.png")
    
    # Plot 4: Combined Dashboard with Quality vs Runtime
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # A) Current Best Quality over Time (Cumulative Best)
    for tracker in trackers:
        if tracker.time_timeline and tracker.quality_timeline:
            ax1.plot(tracker.time_timeline, tracker.quality_timeline, 
                    linewidth=2, marker='o', markersize=3, alpha=0.8, label=tracker.learner_name)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Best Quality Found So Far')
    ax1.set_title('A) Cumulative Best Quality Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0.8, color='orange', linestyle='--', alpha=0.5, label='Good Solution')
    ax1.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Optimal Solution')
    ax1.set_ylim(0, 1.05)
    
    # B) Quality over Time (F1-Score progression)
    for tracker in trackers:
        if tracker.time_timeline and tracker.quality_timeline:
            # Create a simple quality progression plot
            ax2.plot(tracker.time_timeline, tracker.quality_timeline,
                    linewidth=2, marker='o', markersize=3, alpha=0.8, label=tracker.learner_name)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('F1-Score Quality')
    ax2.set_title('B) F1-Score Quality Over Runtime')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0.8, color='orange', linestyle='--', alpha=0.5)
    ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.5)
    ax2.set_ylim(0, 1.05)
    
    # C) Search Progress (concepts tested)
    for tracker in trackers:
        if tracker.time_timeline and tracker.concepts_tested_timeline:
            ax3.plot(tracker.time_timeline, tracker.concepts_tested_timeline,
                    linewidth=2, marker='s', markersize=3, alpha=0.8, label=tracker.learner_name)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Concepts Tested')
    ax3.set_title('C) Search Progress')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # D) Final Performance Summary
    final_qualities = [tracker.final_quality for tracker in trackers]
    concepts_tested = [tracker.total_concepts_tested for tracker in trackers]
    learner_names = [tracker.learner_name for tracker in trackers]
    
    # Create a grouped bar chart
    x = np.arange(len(learner_names))
    width = 0.35
    
    # Normalize concepts tested to 0-1 scale for comparison
    max_concepts = max(concepts_tested) if concepts_tested else 1
    normalized_concepts = [c/max_concepts for c in concepts_tested]
    
    bars1 = ax4.bar(x - width/2, final_qualities, width, label='Final Quality', alpha=0.8, color='green')
    bars2 = ax4.bar(x + width/2, normalized_concepts, width, label='Search Effort (normalized)', alpha=0.8, color='orange')
    
    ax4.set_xlabel('Learning Method')
    ax4.set_ylabel('Score')
    ax4.set_title('D) Final Performance vs Search Effort')
    ax4.set_xticks(x)
    ax4.set_xticklabels([name.replace(' ', '\n') for name in learner_names], fontsize=9)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (qual, concepts) in enumerate(zip(final_qualities, concepts_tested)):
        ax4.text(i - width/2, qual + 0.02, f'{qual:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
        ax4.text(i + width/2, normalized_concepts[i] + 0.02, f'{concepts}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    plt.suptitle('Convergence Analysis Dashboard: V-Learning vs Q-Learning vs Random V-Values\nFamily Dataset (30s runtime)', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.savefig(f'{save_dir}/convergence_dashboard.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{save_dir}/convergence_dashboard.pdf', bbox_inches='tight')
    print(f"Saved: {save_dir}/convergence_dashboard.png")
    
    print(f"\nAll plots saved to: {save_dir}/")

def main(args):
    """Main convergence analysis function."""
    
    print("🚀 Starting Convergence Analysis: V-Learning vs Q-Learning vs Random V-Values")
    print("=" * 80)
    
    # Load knowledge base and embeddings
    kb = KnowledgeBase(path=args.path_knowledge_base)
    
    # Load learning problems
    with open(args.path_learning_problem, "r", encoding="utf-8") as json_file:
        data = json.load(json_file)
    problems = data.get("problems", {})
    
    all_results = []
    
    # Process each learning problem
    for problem_idx, (str_target_concept, examples) in enumerate(problems.items()):
        if problem_idx >= args.max_problems:
            break
            
        print(f"\n📊 Problem {problem_idx + 1}: {str_target_concept}")
        print("-" * 60)
        
        p = examples['positive_examples']
        n = examples['negative_examples']
        
        # Create cross-validation splits
        kf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.random_seed)
        X = np.array(p + n)
        Y = np.array([1.0 for _ in p] + [0.0 for _ in n])
        
        for fold_idx, (train_index, test_index) in enumerate(kf.split(X, Y)):
            if fold_idx >= 1:  # Only use first fold for speed
                break
                
            print(f"\n🔄 Fold {fold_idx + 1}")
            
            # Create training learning problem
            train_pos = {pos_individual for pos_individual in X[train_index][Y[train_index] == 1]}
            train_neg = {neg_individual for neg_individual in X[train_index][Y[train_index] == 0]}
            train_lp = PosNegLPStandard(
                pos=set(map(OWLNamedIndividual, map(IRI.create, train_pos))),
                neg=set(map(OWLNamedIndividual, map(IRI.create, train_neg)))
            )
            
            # Initialize learners with tracking
            trackers = [
                ConvergenceTracker("Drill (Q-Learning)"),
                ConvergenceTracker("DrillV (V-Learning)"), 
                ConvergenceTracker("DrillV (Random V)")
            ]
            
            learners = [
                TrackingDrill(
                    knowledge_base=kb,
                    path_embeddings=args.path_embeddings,
                    refinement_operator=LengthBasedRefinement(knowledge_base=kb),
                    quality_func=F1(),
                    reward_func=CeloeBasedReward(),
                    verbose=0,
                    iter_bound=args.iter_bound,
                    tracker=trackers[0]
                ),
                TrackingDrillV(
                    knowledge_base=kb,
                    path_embeddings=args.path_embeddings,
                    refinement_operator=LengthBasedRefinement(knowledge_base=kb),
                    quality_func=F1(),
                    reward_func=CeloeBasedReward(),
                    verbose=0,
                    iter_bound=args.iter_bound,
                    use_random_v_values=False,
                    tracker=trackers[1]
                ),
                TrackingDrillV(
                    knowledge_base=kb,
                    path_embeddings=args.path_embeddings,
                    refinement_operator=LengthBasedRefinement(knowledge_base=kb),
                    quality_func=F1(),
                    reward_func=CeloeBasedReward(),
                    verbose=0,
                    iter_bound=args.iter_bound,
                    use_random_v_values=True,
                    tracker=trackers[2]
                )
            ]
            
            # Load pretrained models
            print("Loading pretrained models...")
            learners[0].load(directory="pretrained_drill")
            learners[1].load(directory="pretrained_drillv")
            # learners[2] uses random V-values, no loading needed
            
            # Apply optimizations
            for learner in learners[1:]:  # DrillV learners
                learner.optimize_for_performance()
            
            # Run convergence experiments
            results = []
            for learner, tracker in zip(learners, trackers):
                result = run_convergence_experiment(learner, tracker, train_lp, args.max_runtime)
                results.append(result)
                
                if result['success']:
                    print(f"  ✓ {tracker.learner_name}: Quality={result['final_quality']:.3f}, "
                          f"Concepts={result['concepts_tested']}, Time={result['runtime']:.1f}s")
                else:
                    print(f"  ✗ {tracker.learner_name}: Failed - {result['error']}")
            
            # Store results for this problem/fold
            all_results.append({
                'problem': str_target_concept,
                'fold': fold_idx,
                'trackers': trackers,
                'results': results
            })
            
            # Create plots for this specific problem
            if args.plot_individual:
                save_dir = f"convergence_plots/problem_{problem_idx+1}_fold_{fold_idx+1}"
                create_convergence_plots(trackers, save_dir)
    
    # Aggregate results and create summary plots
    print("\n📈 Creating Summary Visualizations...")
    print("=" * 50)
    
    # Combine all trackers for overall analysis
    all_trackers = []
    for result_set in all_results:
        all_trackers.extend(result_set['trackers'])
    
    # Group trackers by method
    drill_trackers = [t for t in all_trackers if "Q-Learning" in t.learner_name]
    drillv_trackers = [t for t in all_trackers if "V-Learning" in t.learner_name and "Random" not in t.learner_name]
    random_trackers = [t for t in all_trackers if "Random V" in t.learner_name]
    
    # Create average tracker for each method
    avg_trackers = []
    for method_trackers, method_name in [(drill_trackers, "Drill (Q-Learning)"), 
                                         (drillv_trackers, "DrillV (V-Learning)"),
                                         (random_trackers, "DrillV (Random V)")]:
        if method_trackers:
            avg_tracker = ConvergenceTracker(method_name)
            # Average the metrics
            avg_tracker.first_good_solution_time = np.mean([t.first_good_solution_time or 30 for t in method_trackers])
            avg_tracker.optimal_solution_time = np.mean([t.optimal_solution_time or 30 for t in method_trackers])
            avg_tracker.final_quality = np.mean([t.final_quality for t in method_trackers])
            avg_tracker.total_concepts_tested = int(np.mean([t.total_concepts_tested for t in method_trackers]))
            
            # For timeline, use the longest timeline
            max_timeline_tracker = max(method_trackers, key=lambda t: len(t.time_timeline))
            avg_tracker.time_timeline = max_timeline_tracker.time_timeline
            avg_tracker.quality_timeline = max_timeline_tracker.quality_timeline
            avg_tracker.concepts_tested_timeline = max_timeline_tracker.concepts_tested_timeline
            
            avg_trackers.append(avg_tracker)
    
    # Create summary plots
    create_convergence_plots(avg_trackers, "convergence_plots/summary")
    
    # Print summary statistics
    print("\n📊 CONVERGENCE ANALYSIS SUMMARY")
    print("=" * 60)
    
    for tracker in avg_trackers:
        summary = tracker.get_summary()
        print(f"\n{summary['learner']}:")
        print(f"  Final Quality: {summary['final_quality']:.3f}")
        print(f"  Time to Good Solution (>0.8): {summary['first_good_time']:.1f}s" if summary['first_good_time'] else "  Time to Good Solution: Not reached")
        print(f"  Time to Optimal Solution (1.0): {summary['optimal_time']:.1f}s" if summary['optimal_time'] else "  Time to Optimal Solution: Not reached")
        print(f"  Concepts Tested: {summary['total_concepts']}")
        print(f"  Efficiency: {summary['efficiency']:.4f}")
    
    # Comparative analysis
    print(f"\n🏆 WINNER ANALYSIS:")
    best_quality = max(t.final_quality for t in avg_trackers)
    fastest_good = min((t.first_good_solution_time or 30) for t in avg_trackers)
    fastest_optimal = min((t.optimal_solution_time or 30) for t in avg_trackers)
    most_efficient = max(t.get_summary()['efficiency'] for t in avg_trackers)
    
    for tracker in avg_trackers:
        summary = tracker.get_summary()
        wins = []
        if tracker.final_quality == best_quality:
            wins.append("Best Quality")
        if (tracker.first_good_solution_time or 30) == fastest_good:
            wins.append("Fastest Good Solution")
        if (tracker.optimal_solution_time or 30) == fastest_optimal:
            wins.append("Fastest Optimal Solution")
        if summary['efficiency'] == most_efficient:
            wins.append("Most Efficient")
            
        if wins:
            print(f"  🥇 {tracker.learner_name}: {', '.join(wins)}")
    
    print(f"\n✅ Analysis complete! Check 'convergence_plots/' for visualizations.")

if __name__ == '__main__':
    parser = ArgumentParser(description="Convergence Analysis: V-Learning vs Q-Learning vs Random V-Values")
    
    # Dataset arguments
    parser.add_argument("--path_knowledge_base", type=str,
                        default='KGs/Family/family-benchmark_rich_background.owl')
    parser.add_argument("--path_embeddings", type=str,
                        default='Experiments/embeddings/Keci_entity_embeddings.csv')
    parser.add_argument("--path_learning_problem", type=str, 
                        default='LPs/Family/lps.json')
    
    # Experiment parameters
    parser.add_argument("--max_runtime", type=int, default=30, 
                        help="Maximum runtime per experiment (seconds)")
    parser.add_argument("--iter_bound", type=int, default=10000, 
                        help="Maximum iterations per experiment")
    parser.add_argument("--folds", type=int, default=2, 
                        help="Number of cross-validation folds")
    parser.add_argument("--max_problems", type=int, default=3, 
                        help="Maximum number of problems to analyze")
    parser.add_argument("--random_seed", type=int, default=42)
    
    # Visualization options  
    parser.add_argument("--plot_individual", action='store_true',
                        help="Create plots for individual problems")
    
    main(parser.parse_args())