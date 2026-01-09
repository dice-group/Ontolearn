"""
Visualize F1 Score Evolution at Different Concept Limits

Creates publication-quality plots comparing OCEL, CELOE, Drill, and DrillV_complex.
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from argparse import ArgumentParser
from pathlib import Path

# Set publication-quality defaults
plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'figure.titlesize': 18,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'text.usetex': False,
    'axes.linewidth': 1.5,
    'grid.linewidth': 0.8,
    'lines.linewidth': 2.5,
    'lines.markersize': 8
})


def create_checkpoint_plots(df, output_dir='results'):
    """Create publication-quality comparison plots for research paper."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Filter to only the 4 learners of interest
    learners_of_interest = ['OCEL', 'CELOE', 'Drill', 'DrillV_complex']
    df_filtered = df[df['method'].isin(learners_of_interest)].copy()
    
    # Rename for cleaner labels
    df_filtered['method'] = df_filtered['method'].replace({
        'DrillV_complex': 'Drill2'
    })
    
    # Color scheme and markers for the 4 learners
    colors = {
        'OCEL': '#1f77b4',      # Blue
        'CELOE': '#ff7f0e',     # Orange
        'Drill': '#2ca02c',     # Green
        'Drill2': '#d62728'     # Red
    }
    
    markers = {
        'OCEL': 'o',
        'CELOE': 's',
        'Drill': '^',
        'Drill2': 'D'
    }
    
    linestyles = {
        'OCEL': '-',
        'CELOE': '--',
        'Drill': '-.',
        'Drill2': ':'
    }
    
    # ===================================================================
    # Plot 1: F1 Score vs Number of Concepts Tested
    # ===================================================================
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Compute mean and std across problems for each method and concept limit
    summary = df_filtered.groupby(['method', 'concept_limit'])['test_f1'].agg(['mean', 'std', 'count']).reset_index()
    
    methods = ['OCEL', 'CELOE', 'Drill', 'Drill2']
    for method in methods:
        method_data = summary[summary['method'] == method].sort_values('concept_limit')
        
        if len(method_data) == 0:
            continue
        
        x = method_data['concept_limit']
        y_mean = method_data['mean']
        y_std = method_data['std'].fillna(0)
        
        ax.plot(x, y_mean, label=method, color=colors[method], 
               linestyle=linestyles[method], marker=markers[method], 
               linewidth=3, markersize=10, markeredgewidth=1.5, markeredgecolor='white')
        ax.fill_between(x, y_mean - y_std, y_mean + y_std, 
                       color=colors[method], alpha=0.15)
    
    ax.set_xlabel('Number of Concepts Tested', fontweight='bold')
    ax.set_ylabel('F1 Score', fontweight='bold')
    ax.set_title('F1 Score vs Number of Concepts Tested', fontweight='bold', pad=20)
    ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_ylim([0, 1.05])
    plt.tight_layout()
    
    filename = output_path / 'f1_vs_concepts.pdf'
    plt.savefig(filename, dpi=300, bbox_inches='tight', format='pdf')
    print(f"✓ Saved: {filename}")
    
    filename_png = output_path / 'f1_vs_concepts.png'
    plt.savefig(filename_png, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filename_png}")
    plt.close()
    
    # ===================================================================
    # Plot 2: F1 Score vs Runtime
    # ===================================================================
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Aggregate by method and concept_limit
    time_f1_summary = df_filtered.groupby(['method', 'concept_limit']).agg({
        'inference_time': 'mean',
        'test_f1': 'mean'
    }).reset_index()
    
    # Filter out concept_limit = 0
    time_f1_summary = time_f1_summary[time_f1_summary['concept_limit'] > 0]
    
    # Get all unique runtime values across all methods for interpolation
    all_runtimes = sorted(time_f1_summary['inference_time'].unique())
    
    # Create common x-axis points for interpolation starting from 0
    if len(all_runtimes) > 1:
        x_interp = np.linspace(0, max(all_runtimes), 100)
    else:
        x_interp = np.array([0] + list(all_runtimes))
    
    for method in methods:
        method_data = time_f1_summary[time_f1_summary['method'] == method].sort_values('inference_time')
        
        if len(method_data) == 0:
            continue
        
        x = method_data['inference_time'].values
        y = method_data['test_f1'].values
        
        # Prepend (0, 0) to start all learners from origin
        x_with_origin = np.concatenate([[0], x])
        y_with_origin = np.concatenate([[0], y])
        
        # Interpolate if we have enough points
        if len(x_with_origin) > 1:
            # Use linear interpolation
            y_interp = np.interp(x_interp, x_with_origin, y_with_origin)
            ax.plot(x_interp, y_interp, label=method, color=colors[method],
                   linestyle=linestyles[method], linewidth=3, alpha=0.8)
        
        # Plot original data points as markers (without the origin point)
        ax.scatter(x, y, color=colors[method], marker=markers[method],
                  s=150, edgecolors='white', linewidths=2, zorder=5)
    
    ax.set_xlabel('Runtime (seconds)', fontweight='bold')
    ax.set_ylabel('F1 Score', fontweight='bold')
    ax.set_title('F1 Score vs Runtime', fontweight='bold', pad=20)
    ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_ylim([0, 1.05])
    plt.tight_layout()
    
    filename = output_path / 'f1_vs_runtime.pdf'
    plt.savefig(filename, dpi=300, bbox_inches='tight', format='pdf')
    print(f"✓ Saved: {filename}")
    
    filename_png = output_path / 'f1_vs_runtime.png'
    plt.savefig(filename_png, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filename_png}")
    plt.close()
    
    # ===================================================================
    # Plot 3: Runtime vs Number of Concepts Tested
    # ===================================================================
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Aggregate runtime by method and concept_limit
    time_summary = df_filtered.groupby(['method', 'concept_limit'])['inference_time'].mean().reset_index()
    
    for method in methods:
        method_data = time_summary[time_summary['method'] == method].sort_values('concept_limit')
        
        if len(method_data) == 0:
            continue
        
        x = method_data['concept_limit']
        y = method_data['inference_time']
        
        ax.plot(x, y, label=method, color=colors[method],
               linestyle=linestyles[method], marker=markers[method],
               linewidth=3, markersize=10, markeredgewidth=1.5, markeredgecolor='white')
    
    ax.set_xlabel('Number of Concepts Tested', fontweight='bold')
    ax.set_ylabel('Runtime (seconds)', fontweight='bold')
    ax.set_title('Runtime vs Number of Concepts Tested', fontweight='bold', pad=20)
    ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    plt.tight_layout()
    
    filename = output_path / 'runtime_vs_concepts.pdf'
    plt.savefig(filename, dpi=300, bbox_inches='tight', format='pdf')
    print(f"✓ Saved: {filename}")
    
    filename_png = output_path / 'runtime_vs_concepts.png'
    plt.savefig(filename_png, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filename_png}")
    plt.close()
    
    print("\n" + "="*80)
    print("All plots saved successfully in PDF and PNG formats!")
    print("="*80)


def main():
    parser = ArgumentParser(description='Visualize F1 evolution at concept checkpoints')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to CSV file with results')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save plots (default: same directory as input CSV)')
    
    args = parser.parse_args()
    
    # If no output_dir specified, use the directory of the input CSV file
    if args.output_dir is None:
        input_path = Path(args.input)
        args.output_dir = str(input_path.parent)
    
    # Load data
    print(f"Loading data from: {args.input}")
    df = pd.read_csv(args.input)
    
    print(f"Data shape: {df.shape}")
    print(f"Methods: {sorted(df['method'].unique())}")
    print(f"Problems: {len(df['problem'].unique())}")
    print(f"Concept limits: {sorted(df['concept_limit'].unique())}")
    
    print(f"Output directory: {args.output_dir}")
    
    # Create plots
    print("\nCreating plots...")
    create_checkpoint_plots(df, args.output_dir)


if __name__ == '__main__':
    main()
