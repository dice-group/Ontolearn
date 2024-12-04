import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import json

def plot_results(data, title="CEL on DBpedia with ", model="TDL") -> None:
    colors = ["tab:green", "tab:red"]
    x_labels = ["$F_1$", "$Runtime\ (sec.)$"]
    # Create the figure and axes
    fig, axes = plt.subplots(1, 2, figsize=(15, 10), sharey=False)  # 2 rows, 3 columns
    
    # Flatten the axes array for easier iteration
    axes = axes.flatten()

    # Plot the histograms
    for i, plot_data in enumerate(data):
        sns.histplot(plot_data, ax=axes[i], color=colors[i], kde=True, bins=20)  # kde=True adds a density curve
        #axes[i].set_title(f'${col}$',fontsize=25)
        axes[i].set_xlabel(x_labels[i], fontsize=25)
        axes[i].set_ylabel("$Counts$", fontsize=25)
        axes[i].tick_params(axis='both', which='major', labelsize=22)
    fig.suptitle(title+model, fontsize=30, fontweight="bold")
    # Adjust layout
    plt.tight_layout()
    fig.savefig(f'{model}-results.pdf', bbox_inches='tight')
    #plt.show()
    
if __name__ == "__main__":
    with open("./large_scale_cel_results_tdl.json") as f:
        data = json.load(f)
    data = [data["f1"]["values"], data["runtime"]["values"]]
    plot_results(data, title="CEL on DBpedia with ", model="TDL")