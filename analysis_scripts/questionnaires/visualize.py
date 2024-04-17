import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.patches as mpatches
import arviz as az

from analysis_scripts.questionnaires.utils import filter_condition, get_mean_std

# Load the colors we chose to make all figures visually similar
with open('analysis_scripts/cognitive_battery/hierarchical_bayesian_models/config/visual_features_config.json', 'r') as file:
    data_colors = json.load(file)

# Create global variables for the colors
COLOR_ZPDES, COLOR_BASELINE = data_colors['COLOR_ZPDES'], data_colors['COLOR_BASELINE']


def descriptive_visualization(study, conditions, questionnaire_name):
    df = pd.read_csv(f'data/{study}/questionnaires/{questionnaire_name}/{questionnaire_name}.csv')
    path_to_store = f'outputs/{study}/questionnaires/{questionnaire_name}/figures'
    Path(path_to_store).mkdir(parents=True, exist_ok=True)

    # Run a bunch of function to explore questionnaires results
    save_boxplots(df, conditions, questionnaire_name, path_to_store, additional_tag='all-', reverse=True)
    save_boxplots(df[df['condition'] == "zpdes"], conditions, questionnaire_name, path_to_store,
                  additional_tag="zpdes-all-")
    save_boxplots(df[df['condition'] == "baseline"], conditions, questionnaire_name, path_to_store,
                  additional_tag="baseline-all-")
    df_tlx_baseline = filter_condition(df, 'baseline')
    df_tlx_zpdes = filter_condition(df, 'zpdes')
    display_cols_value(df_tlx_baseline, df_tlx_zpdes, questionnaire_name, study, path_to_store)


def display_cols_value(df_baseline, df_zpdes, dir, study, path_to_store):
    for col in df_baseline.columns:
        if col != 'session_id' and col != 'condition' and col != 'id_participant':
            plot_scatter_serie(df_baseline, df_zpdes, col, dir, study, path_to_store)


def plot_scatter_serie(df_baseline, df_zpdes, col, dir, study, path_to_store):
    # Plot
    df_baseline_mean, df_baseline_std, df_zpdes_mean, df_zpdes_std = get_mean_std(df_baseline, df_zpdes)
    # Plot the participant level:
    tmp_df_dict = {"baseline": df_baseline, "zpdes": df_zpdes}
    for group, df in tmp_df_dict.items():
        for p in df['id_participant'].unique():
            participant_values = df[df["id_participant"] == p][col]
            color = COLOR_ZPDES if group == "zpdes" else COLOR_BASELINE
            base_shift = 0.2
            shift = base_shift if group == "zpdes" else -base_shift
            plt.scatter(df_baseline_mean[col].index.values + shift, participant_values, s=1.5, color=color, alpha=1)
    # Plot the group level statistics:
    shift = 0
    # plt.errorbar(df_baseline_mean[col].index.values - shift, df_baseline_mean[col], yerr=df_baseline_std[col], label='Baseline',
    #              color=COLOR_BASELINE, elinewidth=3, linewidth=3, alpha=0.9)
    plt.plot(df_baseline_mean[col].index.values - shift, df_baseline_mean[col], color=COLOR_BASELINE, alpha=1,
             marker="s", markersize=4, label='Baseline')
    # plt.errorbar(df_zpdes_mean[col].index.values + shift, df_zpdes_mean[col], yerr=df_zpdes_std[col], label='Zpdes',
    #              color=COLOR_ZPDES, elinewidth=3, linewidth=3, alpha=0.9)
    plt.plot(df_zpdes_mean[col].index.values + shift, df_zpdes_mean[col], color=COLOR_ZPDES, alpha=1, marker="s",
             markersize=4, label='ZPDES')
    # plt.legend(loc="center left", bbox_to_anchor=(1.3, 0.5))
    plt.title(col)
    Path(f"{path_to_store}/scatter_evolution").mkdir(parents=True, exist_ok=True)
    plt.savefig(f"{path_to_store}/scatter_evolution/{col}.png", bbox_inches='tight')
    plt.close()


def save_boxplots(df, conditions, instrument, path, additional_tag="", reverse=False):
    if not df.empty:
        for col in conditions:
            if not reverse:
                df.boxplot(column=col, by=['condition', 'session_id'])
            else:
                df.boxplot(column=col, by=['session_id', 'condition'])
            plt.xticks(rotation=45)
            plt.tight_layout()
            Path(f"{path}/boxplots/").mkdir(parents=True, exist_ok=True)
            plt.savefig(f"{path}/boxplots/{additional_tag}{col}_boxplot.png")
            plt.close()


def get_proportion_plots(questionnaire_name, study, dir, conditions, xtickslabels, possible_values, bpwidth=0.24,
                         yticks=[], ytickslabels=[], colors_barplots=[], legend_labels=[], plot_title=False):
    # First retrieve df:
    df = pd.read_csv(f'data/{study}/questionnaires/{questionnaire_name}/{questionnaire_name}.csv')
    df[conditions] = df[conditions].round().astype(int)
    # Just make sure we don't look at post cog assessment:
    df = df.query('session_id != 0 & session_id != 9')

    path_to_store = f'outputs/{study}/questionnaires/{questionnaire_name}/figures/stacked_bars'
    Path(path_to_store).mkdir(parents=True, exist_ok=True)

    zpdes, baseline = filter_condition(df, 'zpdes'), filter_condition(df, 'baseline')
    nb_sessions = len(xtickslabels)
    # Plot the participant level:
    dict_groups = {"zpdes": zpdes, "baseline": baseline}
    used_labels = set()  # Keep track of labels that have already been added to the plot
    add_legend = True
    for condition in conditions:
        plt.figure(figsize=(8, 6), dpi=300)
        for group_key, group_value in dict_groups.items():
            color = COLOR_ZPDES if group_key == "zpdes" else COLOR_BASELINE
            proportion = group_value.groupby(['session_id', condition]).size().unstack(fill_value=0)
            proportion = proportion / proportion.sum(axis=1).max()
            base_shift = 0.15
            shift = -base_shift if group_key == "zpdes" else base_shift
            bottom = np.zeros(nb_sessions)
            for i, val in enumerate(possible_values):
                if val in proportion.columns:
                    label = legend_labels[i] if val not in used_labels else None  # Only label the first occurrence
                    plt.bar([i + shift for i in range(len(xtickslabels))], proportion[val], bottom=bottom,
                            label=label, color=colors_barplots[i], width=bpwidth, alpha=0.8)
                    bottom += proportion[val]
                    used_labels.add(val)  # Mark this label as used
            # Juste for clarity add a border on the full bars:
            linestyle = '--' if group_key == "zpdes" else '-'
            plt.bar([i + shift for i in range(4)], [1 for i in range(len(xtickslabels))], color='none',
                    edgecolor=color,
                    width=bpwidth, linestyle=linestyle,
                    linewidth=1.5, label=group_key)
        if len(ytickslabels) > 0:
            plt.yticks(yticks, ytickslabels)
        plt.xticks([i for i in range(len(xtickslabels))], xtickslabels)
        if plot_title:
            plt.title(f"{condition}")
        if add_legend:
            # Create a list of handles for the legend entries
            handles = [mpatches.Patch(facecolor='none', label='zpdes', edgecolor=COLOR_ZPDES, linestyle='--'),
                       mpatches.Patch(facecolor='none', label='baseline', edgecolor=COLOR_BASELINE)
                       ] + [mpatches.Patch(color=colors_barplots[i], label=legend_labels[i]) for i in
                            range(len(possible_values))]
            # Create the legend with the ordered handles and labels
            legend = plt.legend(loc='center left', handles=handles, bbox_to_anchor=(1, 0.5))
            add_legend = False
            plt.gca().add_artist(legend)
            plt.draw()
            legend.figure.savefig(f"{path_to_store}/legend_scatter.svg", dpi=300,
                                  bbox_inches=legend.get_window_extent().transformed(
                                      plt.gcf().dpi_scale_trans.inverted()))
            legend.remove()
        plt.legend('', frameon=False)
        plt.subplots_adjust(right=0.75)
        plt.savefig(f"{path_to_store}/{condition}.svg", bbox_inches='tight')
        plt.close()


def generate_diagnostic_plots(trace, title, path_to_store, trace_plot=True, autocorr=True, rank=True):
    """
    Generate and save diagnostic plots for a Bayesian model trace.
    Parameters:
    - trace: MCMC trace from a PyMC model.
    - model_name: Name of the model, used for saving the figure.
    """
    path_to_store += "diagnostic_plots/"
    Path(path_to_store).mkdir(parents=True, exist_ok=True)
    if trace_plot:
        # Trace plots
        az.plot_trace(trace, var_names=['Intercept', 'group_slope', 'session_id_slope', 'group_session_interaction'])
        plt.tight_layout()
        plt.savefig(f"{path_to_store}/{title}_trace.svg", format='svg', dpi=300)
        plt.close()
    if autocorr:
        # Autocorrelation plots
        az.plot_autocorr(trace, var_names=['Intercept', 'group_slope', 'session_id_slope', 'group_session_interaction'])
        plt.savefig(f"{path_to_store}/{title}_autocorr.svg", format='svg', dpi=300)
        plt.close()
    if rank:
        # Rank plots
        az.plot_rank(trace)
        plt.tight_layout()
        plt.savefig(f"{path_to_store}/{title}_rank.svg", format='svg', dpi=300)
        plt.close()


def plot_participants_lines(dict_groups, condition, xtickslabels, zpdes_p, baseline_p, ytickslabels, yticks, plot_title,
                            add_legend, path_to_store):
    Path(f"{path_to_store}svg").mkdir(parents=True, exist_ok=True)
    for group_key, group_value in dict_groups.items():
        color = COLOR_ZPDES if group_key == "zpdes" else COLOR_BASELINE
        for p in group_value['id_participant'].unique():
            participant_values = group_value[group_value["id_participant"] == p][condition]
            base_shift = 0.15
            shift = base_shift if group_key == "zpdes" else -base_shift
            rd_deviation = np.random.random() / 10
            plt.scatter([i + shift + rd_deviation for i in range(len(xtickslabels))], participant_values,
                        s=1.5,
                        color=color,
                        alpha=1)
        intercept, slope = zpdes_p if group_key == "zpdes" else baseline_p
        plt.plot([i for i in range(len(xtickslabels))],
                 [intercept + slope * i for i in range(len(xtickslabels))], c=color, label=group_key,
                 marker='s')
    if len(ytickslabels) > 0:
        plt.yticks(yticks, ytickslabels)
    plt.xticks([i for i in range(len(xtickslabels))], xtickslabels)
    if plot_title:
        plt.title(f"{condition}")
    if add_legend:
        legend = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        add_legend = False
        plt.gca().add_artist(legend)
        plt.draw()
        legend.figure.savefig(f"{path_to_store}svg/legend_scatter.svg", dpi=300,
                              bbox_inches=legend.get_window_extent().transformed(
                                  plt.gcf().dpi_scale_trans.inverted()))
        legend.remove()
    plt.legend('', frameon=False)
    plt.subplots_adjust(right=0.75)
    plt.savefig(f"{path_to_store}svg/{condition}-scatter.svg", bbox_inches='tight')
    plt.close()
