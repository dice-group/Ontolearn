import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_jaccard_vs_cache_size(data, name_reasoner):

    datasets = data["dataset"].unique()

    # Plot each dataset's Cache Size vs Avg Jaccard
    for dataset_name in datasets:
        # Filter data for the specific dataset
        subset = data[data["dataset"] == dataset_name]
        
        # Plot cache size vs avg jaccard for this dataset
        plt.plot(subset["cache_size"], subset["avg_jaccard"], marker='o', label=dataset_name)

    # Label the plot
    plt.xlabel('Cache Size')
    plt.ylabel('Average Jaccard Similarity')

    # plt.title('Cache Size vs Avg Jaccard Similarity for Each Dataset')
    plt.legend()
    plt.grid()
    plt.savefig(f'caching_results/jaccard_vs_cache_size_plot_{name_reasoner}.pdf', format='pdf')

    plt.show()

def plot_RT_vs_RT_cache(data, name_reasoner):

    datasets = data["dataset"].unique()

    for dataset_name in datasets:

        # Filter data for the specific dataset
        subset = data[data["dataset"] == dataset_name]
        
        # Define x and y values for the plot
        x = subset["cache_size"]
        y1 = subset["RT_cache"]  # Runtime with cache
        y2 = subset["RT"]        # Runtime without cache

        # Create a new figure for each dataset
        plt.figure()
        
        # Plot the data
        plt.plot(x, y1, '-b', label='Runtime with Cache')
        plt.plot(x, y2, '-r', label='Runtime without Cache')
        
        # Add legend
        plt.legend()
        
        # Label the axes and title
        plt.xlabel('Cache Size')
        plt.ylabel('Runtime(s)')
        plt.grid()
        # plt.title(f'Cache Size vs. Runtime for {dataset_name}')
        plt.savefig(f'caching_results/runtime_plot_{name_reasoner}.pdf', format='pdf')

        # Show the plot
        plt.show()

def plot_scale_factor(data, name_reasoner):
    # Ensure 'RT' and 'RT_cache' are numeric
    data["RT"] = pd.to_numeric(data["RT"], errors='coerce')
    data["RT_cache"] = pd.to_numeric(data["RT_cache"], errors='coerce')

    # Get unique datasets
    datasets = data["dataset"].unique()

    # Plot speedup factor for each dataset on the same plot
    for dataset_name in datasets:
        # Filter data for the specific dataset
        subset = data[data["dataset"] == dataset_name]
        
        # Calculate speedup factor
        speedup_factor = subset["RT"] / subset["RT_cache"]
        
        # Plot speedup factor vs cache size for this dataset
        plt.plot(subset["cache_size"], speedup_factor, marker='o', label=dataset_name)

    # Label the axes and add a title
    plt.xlabel('Cache Size')
    plt.ylabel('Speedup Factor(RT / RT_cache)')
    
    # Add legend to identify each dataset
    plt.legend()
    plt.grid()
    
    # Save the plot as a PDF
    plt.savefig(f'caching_results/scale_factor_plot_{name_reasoner}.pdf', format='pdf')
    
    # Show the plot
    plt.show()

def bar_plot_all_data(data, cache_size, name_reasoner):
    # Plotting
    grouped_df = data.groupby(['dataset', 'Type', 'cache_size']).agg({
        'time_ebr': 'mean',
        'time_cache': 'mean',
        'Jaccard': 'mean'
    }).reset_index()

    grouped_df = grouped_df[grouped_df["cache_size"]==cache_size]

    df = grouped_df
    fig, ax = plt.subplots(figsize=(10, 6))

    # Creating unique labels for each combination of dataset, Type, and cache_size
    df['label'] = df['dataset'] + "-" + df['Type'] 
    # Set positions and width for bars
    x = np.arange(len(df['label']))
    width = 0.35

    # Plot time_ebr and time_cache side-by-side for each label
    ax.bar(x - width/2, df['time_ebr'], width, label='RT without Cache', color='skyblue')
    ax.bar(x + width/2, df['time_cache'], width, label='RT With Cache', color='salmon')

    # Labels and titles
    # ax.set_xlabel('Instance Type and Cache Size')
    ax.set_ylabel('Running Time (seconds)')
    ax.set_title(f'Running Time Comparison With and Without Cache by Instance Type (cache size = {cache_size})')
    ax.set_xticks(x)
    ax.set_xticklabels(df['label'], rotation=35, ha='right')
    ax.legend()
    # Show plot
    plt.tight_layout()
    plt.savefig(f'caching_results/bar_plot_{name_reasoner}.pdf', format='pdf')
    plt.show()


def bar_plot_separate_data(data, cache_size, name_reasoner):

    grouped_df = data.groupby(['dataset', 'Type', 'cache_size']).agg({
        'time_ebr': 'mean',
        'time_cache': 'mean',
        'Jaccard': 'mean'
    }).reset_index()

    grouped_df = grouped_df[grouped_df["cache_size"]==cache_size]

    df = grouped_df

    datasets = df['dataset'].unique()

    # Plot for each dataset separately
    for dataset in datasets:
        subset = df[df['dataset'] == dataset]
        
        # Creating unique labels for each Type and cache_size combination
        subset['label'] = subset['Type'] 
        # Set positions and width for bars
        x = np.arange(len(subset['label']))
        width = 0.35

        # Initialize the figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot time_ebr and time_cache side-by-side for each label
        ax.bar(x - width/2, subset['time_ebr'], width, label='RT without cache', color='skyblue')
        ax.bar(x + width/2, subset['time_cache'], width, label='RT with Cache', color='salmon')

        # Labels and titles
        ax.set_xlabel('Instance Type')
        ax.set_ylabel('Running Time (seconds)')
        ax.set_title(f'Running Time Comparison With and Without Cache for {dataset}')
        ax.set_xticks(x)
        ax.set_xticklabels(subset['label'], rotation=35, ha='right')
        ax.legend()
        # Adjust layout and display
        plt.tight_layout()
        plt.savefig(f'caching_results/bar_plot_{dataset}_{name_reasoner}.pdf', format='pdf')
        plt.show()