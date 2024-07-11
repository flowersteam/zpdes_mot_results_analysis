import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
import os
import arviz as az
import numpy as np
import math
import json
import scipy.stats as stats
from matplotlib.patheffects import withStroke
import pymc as pm

from analysis_scripts.cognitive_battery.hierarchical_bayesian_models.visualize_utils import retrieve_config, \
    set_ax_deltas
from analysis_scripts.cognitive_battery.hierarchical_bayesian_models.utils import get_SDDR, get_hdi_bounds, \
    get_p_of_effect

matplotlib.rc('xtick', labelsize=10)
matplotlib.rc('ytick', labelsize=10)

# Load the colors we chose to make all figures visually similar
with open('analysis_scripts/cognitive_battery/hierarchical_bayesian_models/config/visual_features_config.json',
          'r') as file:
    data_colors = json.load(file)

# Create global variables for the colors
for key, value in data_colors.items():
    globals()[key] = value


def plot_traces_and_deltas(studies, config_fig, all_conditions, no_cdt_studies=[]):
    '''
    This function entangled 2 figures in order to avoid looping several times on the same objects.
    :param studies:
    :param config_fig:
    :param all_conditions:
    :param no_cdt_studies:
    :return:
    '''
    for study in studies:
        path_study = f"outputs/{study}/cognitive_battery"
        print(f"\n \n \n \n ===================={study}======================")
        for metric_type in all_conditions.keys():
            # Init few important variables:
            rope_start, rope_end, xmin, xmax, figsize, dpi, y_offset, step_offset = retrieve_config(config_fig,
                                                                                                    metric_type)
            labels, yticks = [], []

            # Init the delta summary figure
            # This figure is updated for all task measured on the metric type
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
            set_ax_deltas(ax, xmin, xmax)

            # Plot vertical lines for hdi:
            # plot_vline(rope_start, rope_end, ax)
            # Instead of plotting a rope, we plot the null effect point on 0:
            plot_vline_0(ax)

            for task in all_conditions[metric_type].keys():
                path = f"{path_study}/{task}"
                # creating a new directory for outputs in case it doesn't exist:
                path_to_store = f"{path}/visual_outputs"
                Path(path_to_store).mkdir(parents=True, exist_ok=True)
                # Get the trace and update the delta summary figure
                y_offset = plot_trace_and_deltas_of_task(all_conditions, metric_type, task, config_fig, study,
                                                         no_cdt_studies, path_to_store, ax, y_offset, step_offset,
                                                         labels,
                                                         rope_start, rope_end, yticks, fig, xmin, xmax, path)
            fig.savefig(f"{path_study}/results_{metric_type}.svg", bbox_inches='tight')
            plt.close()


def plot_trace_and_deltas_of_task(all_conditions, metric_type, task, config_fig, study,
                                  no_cdt_studies, path_to_store, ax, y_offset, step_offset, labels, rope_start,
                                  rope_end, yticks, fig, xmin, xmax, path):
    '''
    For each difficulty condition of a task, load the trace of the estimated parameters of all specified models
    :param all_conditions:
    :param metric_type:
    :param task:
    :param config_fig:
    :param study:
    :param no_cdt_studies:
    :param path_to_store:
    :param ax:
    :param y_offset:
    :param step_offset:
    :param labels:
    :param rope_start:
    :param rope_end:
    :param yticks:
    :param fig:
    :param xmin:
    :param xmax:
    :param path_study:
    :return:
    '''
    for task_condition in all_conditions[metric_type][task]['conditions']:
        for model in all_conditions[metric_type][task]['models']:
            sddr_mu, sddr_sigma = config_fig[metric_type]["var_hyp_range_sddr"][model]
            # var_names correspond to the parameters names we want to display
            var_names, is_a_control_study = retrieve_var_names(config_fig, metric_type, model, study,
                                                               no_cdt_studies)
            if os.path.exists(f"{path}/{task_condition}-{model}_inference_data.json"):
                # Open trace
                trace = az.from_json(f"{path}/{task_condition}-{model}_inference_data.json")
                # First plot the trace:
                # plot_trace(trace, var_names, path_to_store=f"{path_to_store}/{task_condition}-{model}",
                #            model_type=model, task=task, condition=task_condition)
                # Second plot the deltas:
                ax, y_offset, labels, yticks = plot_deltas(trace, ax_deltas=ax,
                                                           task_condition=task_condition,
                                                           task=task, y_offset=y_offset,
                                                           step_offset=step_offset,
                                                           labels=labels, rope_start=rope_start,
                                                           rope_end=rope_end, yticks=yticks,
                                                           model=model,
                                                           no_condition=is_a_control_study, study=study,
                                                           sddr_mu=sddr_mu, sddr_sigma=sddr_sigma,
                                                           xmax=xmax, add_pre=True, config_fig=config_fig,
                                                           metric_type=metric_type)
            # Add margin between task condition:
            y_offset += step_offset
        # Add lines to separate between tasks
        y_offset = add_visual_features_to_delta(fig, ax, y_offset, step_offset, xmin, xmax, labels, yticks)
    return y_offset


def retrieve_var_names(config_fig, metric_type, model, study, no_cdt_studies):
    '''
    It is possible that for some study, no group condition is used (typically for control group). As such, models are
    fitted without 'delta btw groups' parameter distribution (when it is the case, this function detects and deletes the
    'delta' parameters)
    :param config_fig:
    :param metric_type:
    :param model:
    :param study:
    :param no_cdt_studies: e.g 'v0_prolific'
    :return:
    '''
    var_names = config_fig[metric_type]["var_specifics"][model]
    # Make sure that var_names are correct if no conditions:
    is_a_control_study = study in no_cdt_studies
    if is_a_control_study:
        var_names = [var for var in var_names if "delta" not in var] + ["delta"]
    return var_names, is_a_control_study


def add_visual_features_to_delta(fig, ax, y_offset, step_offset, xmin, xmax, labels, yticks):
    # Add lines to separate between tasks:
    ax.hlines(y=y_offset - step_offset / 2, xmin=xmin, xmax=xmax, color="black")
    y_offset += step_offset
    labels.append("")
    yticks.append(y_offset)
    labels, yticks = labels[:-1], yticks[:-1]
    ax.set_yticks(yticks)
    ax.set_yticklabels(labels)
    # Adjust to add grids or ticks:
    for index_lab, label in enumerate(labels):
        if label == "":
            ax.yaxis.get_majorticklines()[index_lab * 2].set_visible(False)
        else:
            ax.get_ygridlines()[index_lab].set_visible(False)
    return y_offset


def plot_vline(rope_start, rope_end, ax):
    # Adding vertical lines
    ax.axvline(x=rope_start, color='gray', linestyle='--')
    ax.axvline(x=rope_end, color='gray', linestyle='--')


def plot_vline_0(ax):
    ax.axvline(x=0, color='red', linestyle='--')


def plot_trace(trace, var_names, path_to_store, model_type="hierarbinom", task="mot", condition="all"):
    az.plot_trace(trace, var_names=var_names, compact=True, legend=True)
    plt.suptitle(f"{task} - {condition}")
    plt.tight_layout()
    plt.savefig(f"{path_to_store}_traces_{model_type}.png")
    plt.close()


def plot_deltas(trace, ax_deltas=None, task_condition="5-nb-targets", task="moteval", y_offset=0,
                step_offset=0.2, labels=[], rope_start=-0.05, rope_end=0.05, yticks=[], model="hierarbinom",
                no_condition=False, study="v3_prolific", sddr_mu=0, sddr_sigma=0.05, xmax=0.3, add_pre=True,
                config_fig=None, metric_type=None):
    # plt.figure(figsize=figsize)
    # plot_distribution_of_deltas(deltas, data_pre)
    vars = ["delta_zpdes", "delta_baseline"]
    condition = ["zpdes", "baseline"]
    color_bars = [COLOR_ZPDES_v3_prolific, COLOR_BASELINE_v3_prolific]
    color_txt = ["white", "white"]
    # Instead of displaying the probability to be in ROPE:
    # probas = [get_p_in_ROPE(trace, rope_start, rope_end, var) for var in vars]
    # We display the probability to see an effect:
    probas = [get_p_of_effect(trace, var) for var in vars]
    hdis = [get_hdi_bounds(trace, var) for var in vars]
    sddrs = [get_SDDR(trace, var, parameter_priors={'mu': sddr_mu, 'sigma': sddr_sigma}) for var in vars]
    if model != "hierar_binom_n_var":
        labels += ["", f"{task}_{task_condition} \n {model}", ""]
    else:
        labels += ["", f"{task}_{task_condition}", ""]
    if add_pre:
        p, sddr, hdi = get_pre_bar(config_fig, metric_type, model, trace, sddr_mu, sddr_sigma)
        probas.append(p)
        sddrs.append(sddr)
        hdis.append(hdi)
        color_bars.append("white")
        color_txt.append("white")
        condition.append("diff_pre")
        labels.append("")
        yticks += [y_offset, y_offset + step_offset / 2, y_offset + step_offset, y_offset + (2 * step_offset)]
    else:
        yticks += [y_offset, y_offset + step_offset / 2, y_offset + step_offset]
    for delta, p, sddr, c_b, c_t, c_l in zip(hdis, probas, sddrs, color_bars, color_txt, condition):
        ax_deltas = plot_barh(delta, p, sddr, ax_deltas, height=0.5, color_bar=c_b, color_txt=c_t,
                              label=f"{task}_{task_condition}",
                              y_offset=y_offset, xmax=xmax)
        y_offset += step_offset
    return ax_deltas, y_offset, labels, yticks


def get_pre_bar(config_fig, metric_type, model, trace, sddr_mu, sddr_sigma):
    var = config_fig[metric_type]["var_delta_btw_study"][model]
    trace_var = trace.posterior[var]
    trace_diff = trace_var[:, :, 1, 0] - trace_var[:, :, 0, 0]
    vals = trace_diff.values.flatten()
    # returns the hdi
    hdi = az.hdi(vals, hdi_prob=0.94)
    proba_positive_effect = np.mean((trace_diff > 0))
    proba = np.max([proba_positive_effect, 1 - proba_positive_effect])
    flat_samples = trace_diff.values[:, 1000:].flatten()
    kde = stats.gaussian_kde(flat_samples)
    posterior_density_at_zero = kde(0)  # Evaluate the density at zero
    prior_density_at_zero = stats.norm.pdf(0, loc=sddr_mu, scale=sddr_sigma)
    sddr = posterior_density_at_zero / prior_density_at_zero
    return proba, sddr[0], hdi


def plot_barh(hdi, proba, sddr, ax, color_bar, height=0.3, label="delta_zpdes", color_txt='black', y_offset=0,
              xmax=0.3):
    # Plot the HDIs
    ax.barh(y_offset, width=(hdi[1] - hdi[0]), left=hdi[0], height=height, align='center', alpha=1, label=label,
            color=color_bar, edgecolor='black')
    # Displaying the probabilities in the middle of the bars
    proba_label = "" if math.isclose(proba, 0, abs_tol=0.01) else f"{proba:.1%}"
    center = hdi[0] + (hdi[1] - hdi[0]) / 2
    ax.text(center, y_offset, proba_label, ha='center', va='center', color=color_txt, fontsize=10, fontweight='bold',
            path_effects=[withStroke(linewidth=1, foreground='black')])
    ax.text(0.85 * xmax, y_offset, f"{1 / sddr:.2f}", ha='center', va='center', color='black', fontsize=9,
            fontweight='bold',
            bbox=dict(facecolor='white', edgecolor='none', pad=1))
    return ax


def render_model_graph(model, path_to_store, name):
    path_to_store = '' + '/'.join(filter(bool, path_to_store.split('/')[:-1])) + '/models/'
    Path(path_to_store).mkdir(parents=True, exist_ok=True)
    gv = pm.model_to_graphviz(model)
    gv.format = 'png'
    gv.render(filename=f"{path_to_store}/{name}-graphviz")
