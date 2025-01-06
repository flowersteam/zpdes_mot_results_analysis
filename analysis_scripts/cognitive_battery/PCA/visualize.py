from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, leaves_list
import seaborn as sns
from pingouin import ttest
import pandas as pd


def get_scatter_of_indiv_2D(df_pca, path):
    # Create a scatter plot of the first two principal components
    plt.figure(figsize=(8, 6))
    plt.scatter(df_pca['PC1'], df_pca['PC2'], marker='o', c='b', alpha=0.7)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA - Scatter Plot of Principal Components 1 and 2')
    plt.grid()
    # Show or save the plot
    plt.savefig(f"{path}/figures/scatter_plots.png")


def create_correlation_circle(variable_contributions, variable_names, path):
    # Create a figure
    fig, ax = plt.subplots(figsize=(8, 8))
    # Define a colormap for arrows
    cmap = plt.get_cmap('tab10')
    # Plot the correlation circle (arrows) with unique colors and labels
    for i, var_name in enumerate(variable_names):
        arrow_length = np.sqrt(variable_contributions[i, 0] ** 2 + variable_contributions[i, 1] ** 2)
        arrow_x = variable_contributions[i, 0] / arrow_length
        arrow_y = variable_contributions[i, 1] / arrow_length
        color = cmap(i % 10)  # Use modulo to cycle through colors
        plt.arrow(0, 0, arrow_x, arrow_y, color=color, alpha=0.7, label=var_name)
    # Add a circle of radius 1
    circle = plt.Circle((0, 0), 1, fill=False, color='black', linestyle='dotted', alpha=0.5)
    ax.add_artist(circle)
    # Add a legend on the right of the plot
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid()
    plt.axhline(0, color='black', lw=0.5)
    plt.axvline(0, color='black', lw=0.5)
    # Set axis limits to show the circle
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    # Show or save the plot
    plt.savefig(f"{path}/figures/correlation_circle.png")


def get_scree_plot(pca_object, n_components, path):
    # Generate Scree Plot
    # Previous method to handle nb components to keep:
    # n_components_keep = sum(pca_object.explained_variance_ > 1)
    n_components_keep = sum(pca_object.explained_variance_ratio_ > (1 / n_components))
    variance_explained_kaiser = sum(pca_object.explained_variance_ratio_[:n_components_keep]) * 100
    plt.figure(figsize=(10, 6))
    # Create bar plot for explained variance
    plt.bar(range(1, n_components + 1), pca_object.explained_variance_ratio_,
            alpha=0.7,
            facecolor='blue',  # Set the fill color to white
            edgecolor='black',  # Set the border color to black
            linewidth=0.8,  # Set the width of the border
            label='Variance Explained by Each PC')
    # Add line plot on top of the bar plot
    plt.plot(range(1, n_components + 1), pca_object.explained_variance_ratio_, marker='o', color='red',
             label='Cumulative Variance Explained')
    # Add a vertical line for the Kaiser Criterion
    plt.vlines(x=n_components_keep, ymin=0, ymax=1.1 * pca_object.explained_variance_ratio_[0], colors='black',
               linestyles='dashed', label='Variance Threshold')
    plt.title(f'Scree Plot - \n Variance explained by kept components: {variance_explained_kaiser:.2f} ')
    plt.xticks([i+1 for i in range(n_components)])
    plt.xlabel('Principal Component')
    plt.ylabel('Variance Explained')
    plt.legend()
    plt.savefig(f"{path}/figures/scree_plots.png")


def get_heat_and_dendro_contrib(variable_contributions, variable_names, n_components, path):
    # Assuming you have variable_contributions, a 2D numpy array with the variable contributions to components
    # Also, make sure you have the original variable names in a list, e.g., variable_names
    # Keep only the first 10 components
    # variable_contributions = variable_contributions[:, :10]
    # Create a linkage matrix for hierarchical clustering
    linkage_matrix = linkage(variable_contributions, method='single')
    # Perform hierarchical clustering and get the reordered indices
    order = leaves_list(linkage_matrix)
    # Reorder the variable names based on clustering
    ordered_variable_names = [variable_names[i] for i in order]
    # Create a larger figure for the heatmap
    fig, ax1 = plt.subplots(figsize=(12, 8))
    # Create a heatmap of the reordered variable contributions with a custom colormap
    cmap = LinearSegmentedColormap.from_list("custom_colormap", [(0, 'blue'), (0.5, 'white'), (1, 'red')])
    cax = ax1.matshow(variable_contributions[order, :], cmap=cmap, aspect='auto', vmin=-1, vmax=1,
                      extent=(-0.5, n_components - 0.5, len(variable_names) - 0.5, -0.5))
    # Invert the y-axis to match the dendrogram
    ax1.invert_yaxis()
    ax1.set_yticks(np.arange(len(ordered_variable_names)))
    ax1.set_yticklabels(ordered_variable_names)
    ax1.set_xticks(np.arange(n_components))
    ax1.set_xticklabels([f'PC{i + 1}' for i in range(n_components)], rotation=0)
    # Add text labels with values in black
    for i in range(len(variable_names)):
        for j in range(n_components):
            value = variable_contributions[order, :][i, j]
            ax1.text(j, i, f'{value:.2f}', ha='center', va='center', color='black', fontsize=10, fontweight='bold')
    # Add a color bar for reference
    cbar = fig.colorbar(cax, ax=ax1, ticks=[-1, 0, 1])
    cbar.set_label('Loading')
    plt.title('Variable Loadings for Principal Components (First 10 Components) - Heatmap')
    fig.savefig(f"{path}/figures/heatmap.png")
    plt.close()
    # Create a separate figure for the dendrogram
    fig2, ax2 = plt.subplots(figsize=(3, 8))
    # Create a dendrogram for hierarchical clustering with custom labels
    dendrogram(linkage_matrix,
               ax=ax2,
               orientation='left',
               # color_threshold=0.5 * np.max(linkage_matrix[:, 2]),
               distance_sort='descending',
               show_leaf_counts=True,
               leaf_font_size=20,
               color_threshold=0
               )
    ax2.set_yticklabels(ordered_variable_names)
    plt.title('Hierarchical Clustering Dendrogram')
    fig2.set_size_inches(10, 8)
    fig2.tight_layout()
    fig2.savefig(f"{path}/figures/dendro.png")


def plot_loading_matrix(loadings_df, variable_names, n_components_to_keep, n_components, path, name):
    # Assuming you have variable_contributions, a 2D numpy array with the variable contributions to components
    # Also, make sure you have the original variable names in a list, e.g., variable_names
    # Keep only the first 10 components
    # variable_contributions = variable_contributions[:, :10]
    # Create a linkage matrix for hierarchical clustering
    loadings = loadings_df.values
    linkage_matrix = linkage(loadings, method='single')
    # Perform hierarchical clustering and get the reordered indices
    order = leaves_list(linkage_matrix)
    # order = [9, 8, 21, 19, 15, 18, 16, 17, 20, 22, 4, 3, 2, 7, 6, 5, 10, 12, 13, 14, 11, 0, 1, 23]
    order = [i for i in range(n_components)]
    order.reverse()
    # Reorder the variable names based on clustering
    ordered_variable_names = [variable_names[i] for i in order]
    # Create a larger figure for the heatmap
    fig, ax1 = plt.subplots(figsize=(12, 8))
    # Create a heatmap of the reordered variable contributions with a custom colormap
    # cmap = LinearSegmentedColormap.from_list("custom_colormap", [(0, 'blue'), (0.5, 'white'), (1, 'red')])
    cax = ax1.matshow(loadings[order, :n_components_to_keep], cmap="PuOr_r", aspect='auto',
                      extent=(-0.5, n_components_to_keep - 0.5, len(variable_names) - 0.5, -0.5))
    # Invert the y-axis to match the dendrogram
    ax1.invert_yaxis()
    ax1.set_yticks(np.arange(len(ordered_variable_names)))
    ax1.set_yticklabels(ordered_variable_names)
    ax1.set_xticks(np.arange(n_components_to_keep))
    ax1.set_xticklabels([f'PC{i + 1}' for i in range(n_components_to_keep)], rotation=0)
    # Add text labels with values in black
    for i in range(len(variable_names)):
        for j in range(n_components_to_keep):
            value = loadings[order, :][i, j]
            ax1.text(j, i, f'{value:.2f}', ha='center', va='center', color='black', fontsize=10, fontweight='bold')
    # Add a color bar for reference
    cbar = fig.colorbar(cax, ax=ax1, ticks=[-1, 0, 1])
    cbar.set_label('Loading')
    plt.title(f'Variable Loadings for Principal Components (First {n_components_to_keep} Components) - Heatmap')
    fig.tight_layout()
    fig.savefig(f"{path}/figures/heatmap_{name}.png")
    plt.close()


def plot_correlation_matrix(df_transformed, path, name="main"):
    correlation_matrix = df_transformed.loc[:, 'PC1':'PC6'].corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title(f'Correlation Matrix')
    plt.savefig(f"{path}/figures/{name}_model_correlations.png")


def plot_diff_on_PC(pre_baseline, post_baseline, pre_zpdes, post_zpdes, PC_name, study, path, name="main"):
    # Merge the two dataframes
    plt.close()
    fig, ax = plt.subplots()
    # t_statistic_baseline, p_value_baseline = ttest_rel(pre_baseline[PC_name], post_baseline[PC_name])
    # t_statistic_zpdes, p_value_zpdes = ttest_rel(pre_zpdes[PC_name], post_zpdes[PC_name])
    # t_statistic_pre, p_value_pre = ttest_ind(pre_zpdes[PC_name], pre_baseline[PC_name])
    # t_statistic_post, p_value_post = ttest_ind(post_zpdes[PC_name], post_baseline[PC_name])
    diff_baseline = ttest(pre_baseline[PC_name], post_baseline[PC_name], paired=True)
    diff_zpdes = ttest(pre_zpdes[PC_name], post_zpdes[PC_name], paired=True)
    diff_pre = ttest(pre_zpdes[PC_name], pre_baseline[PC_name], paired=False)
    diff_post = ttest(post_zpdes[PC_name], post_baseline[PC_name], paired=False)
    save = pd.DataFrame.from_dict({"t_diff_zpdes": [diff_zpdes["T"].values[0]],
                                   "p_diff_zpdes": [diff_zpdes["p-val"].values[0]],
                                   "d_diff_zpdes": [diff_zpdes["cohen-d"].values[0]],
                                   "BF_diff_zpdes": [diff_zpdes["BF10"].values[0]],
                                   "t_diff_baseline": [diff_baseline["T"].values[0]],
                                   "p_diff_baseline": [diff_baseline["p-val"].values[0]],
                                   "d_diff_baseline": [diff_baseline["cohen-d"].values[0]],
                                   "BF_diff_baseline": [diff_baseline["BF10"].values[0]],
                                   "t_diff_pre": [diff_pre["T"].values[0]],
                                   "p_diff_pre": [diff_pre["p-val"].values[0]],
                                   "d_diff_pre": [diff_pre["cohen-d"].values[0]],
                                   "BF_diff_pre": [diff_pre["BF10"].values[0]],
                                   "t_diff_post": [diff_post["T"].values[0]],
                                   "p_diff_post": [diff_post["p-val"].values[0]],
                                   "d_diff_post": [diff_post["cohen-d"].values[0]],
                                   "BF_diff_post": [diff_post["BF10"].values[0]],
                                   }
                                  )
    save.to_csv(f"{path}/ttests_data/{PC_name}-{study}-{name}.csv")
    plt.scatter(pre_baseline[PC_name], post_baseline[PC_name], alpha=0.7, c='blue', label="baseline")
    plt.scatter(pre_zpdes[PC_name], post_zpdes[PC_name], alpha=0.7, c='red', label="zpdes")
    plt.plot([i for i in range(-6, 6)], [i for i in range(-6, 6)])
    plt.xlabel('Pre')
    plt.ylabel('Post')
    plt.title(f'{PC_name}')
    plt.grid(True)
    plt.legend()
    fig.savefig(f"{path}/figures/{PC_name}-{study}-{name}-diffplot.png")


def get_all_PCA_figures(df_pca, path, DR_type, model, n_components, variable_contributions, columns_for_pca,
                        average_loadings, main_model_loadings):
    get_scatter_of_indiv_2D(df_pca, path)
    if DR_type == "PCA":
        get_scree_plot(model, n_components, path)
    create_correlation_circle(variable_contributions, columns_for_pca, path)
    get_heat_and_dendro_contrib(variable_contributions, columns_for_pca, n_components, path)
    n_components_to_keep = 6
    plot_loading_matrix(average_loadings, columns_for_pca, n_components_to_keep, n_components, path, name="average")
    plot_loading_matrix(main_model_loadings, columns_for_pca, n_components_to_keep, n_components, path, name="main")
    plot_correlation_matrix(df_pca, path, name="main")
