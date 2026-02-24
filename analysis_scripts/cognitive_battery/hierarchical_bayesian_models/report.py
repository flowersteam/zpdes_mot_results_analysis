from .utils import get_SDDR
import arviz as az
from latex import build_pdf
import numpy as np
from datetime import datetime
import os

TEXBIN = "/Library/TeX/texbin"
os.environ["PATH"] = f"{TEXBIN}:" + os.environ.get("PATH", "")


def get_latex_tables(studies, config_fig, all_conditions):
    """
    This function returns a file with latex tables corresponding to various results
    """
    with open(
        "analysis_scripts/cognitive_battery/hierarchical_bayesian_models/config/main_preamble_reports.tex",
        "r",
    ) as file:
        str = file.read()
    times = ["post", "pre"]
    counter = 0
    for time in times:
        for metric_type in all_conditions.keys():
            for task in all_conditions[metric_type].keys():
                for model in all_conditions[metric_type][task]["models"]:
                    counter += 1
                    if time == "post":
                        str += generate_latex_post_table(
                            studies,
                            config_fig,
                            all_conditions,
                            time=time,
                            model=model,
                            task=task,
                        )
                    str += generate_latex_table(
                        studies,
                        config_fig,
                        all_conditions,
                        time=time,
                        model=model,
                        task=task,
                    )
                    # After pre / post; add comparisons post - pre :
                    str += "\n \n"
                    if counter % 5 == 0:
                        str += "\clearpage \n"
                    print(f"{task}-{model}")
    str += "\n \end{document} "
    name = datetime.now().strftime("%Y%m%d-%H%M%S")
    text_file = open(f"outputs/{name}-tables_new.tex", "w")
    _ = text_file.write(str)
    text_file.close()
    with open(f"outputs/{name}-tables_new.tex") as f:
        pdf = build_pdf(f)
    # Save the resulting PDF to file
    pdf.save_to(f"outputs/{name}-tables_new.pdf")


def generate_latex_table(
    studies,
    config_fig,
    all_conditions,
    time="pre",
    model="hierar_binom_n_var",
    task="moteval",
):
    latex_str = (
        "\\begin{table}[h]\n\\centering\n\\begin{adjustbox}{width=1.5\\textwidth,center=\\textwidth}\n"
        "\\begin{tabular}{|c|c|c|M{1.8cm}|c|M{2cm}|c|c|c|}\n\\hline\n"
    )
    latex_str += (
        "\\textbf{Study} & \\textbf{Condition} & \\textbf{Group} & "
        " \\textbf{Mean Posterior} & \\textbf{HDI Range} & "
        "\\textbf{Difference Mean Posterior} & \\textbf{BF10} &"
        " \\textbf{Difference HDI Range} & \\textbf{Btw group P(effect)}"
        "\\\\ \\hline\n "
    )
    groups = ["zpdes", "baseline"]
    for study in studies:
        first_study_entry = True
        for metric_type in all_conditions.keys():
            if (
                task in all_conditions[metric_type]
                and model in config_fig[metric_type]["var_delta_btw_study"]
            ):
                rope_start, rope_end = (
                    config_fig[metric_type]["rope_start"],
                    config_fig[metric_type]["rope_end"],
                )
                for task_condition in all_conditions[metric_type][task]["conditions"]:
                    nb_row_per_study = len(
                        all_conditions[metric_type][task]["conditions"]
                    ) * len(groups)
                    path = f"outputs/{study}/cognitive_battery/{task}"
                    trace = az.from_json(
                        f"{path}/{task_condition}-{model}_inference_data.json"
                    )
                    var = config_fig[metric_type]["var_delta_btw_study"][model]
                    trace_var = trace.posterior[var]
                    trace_summary = az.summary(trace_var)
                    time_index = 0 if time == "pre" else 1
                    trace_diff = (
                        trace_var[:, :, 1, time_index] - trace_var[:, :, 0, time_index]
                    )
                    trace_summary_diff = az.summary(trace_diff)
                    info_diff = trace_summary_diff.loc[f"{var}"]
                    mu, sigma = config_fig[metric_type]["var_hyp_range_sddr"][model]
                    info_diff["sddr"] = get_SDDR(
                        trace_diff, var, {"mu": mu, "sigma": sigma}
                    )
                    # proba_diff = sum(
                    #     [(sum(chain.values > rope_end) + sum(chain.values < rope_start)) / (len(chain.values))
                    #      for chain in trace_diff]) / len(trace_diff)
                    proba_diff = np.mean(
                        [
                            np.max([sum(chain.values > 0), sum(chain.values < 0)])
                            / (len(chain.values))
                            for chain in trace_diff
                        ]
                    )
                    first_task_condition_entry = True
                    for grp in groups:
                        if first_study_entry:
                            # Calculate the number of rows for this study entry
                            latex_str += f"\\multirow{{{nb_row_per_study}}}{{*}}{{{rename_str_for_latex(study)}}} & "
                            first_study_entry = False
                        else:
                            latex_str += " & "
                        if model == "switching_cost":
                            tmp_grp = 1 if grp == "zpdes" else 0
                            tmp_var = 1 if time == "post" else 0
                            info = trace_summary.loc[
                                f"switching_cost[{tmp_grp}, {tmp_var}]"
                            ]
                        else:
                            info = trace_summary.loc[f"{var}[{grp}, {time}]"]
                        if first_task_condition_entry:
                            latex_str += (
                                f" \\multirow{{{2}}}{{*}}{{{rename_str_for_latex(task_condition)}}} & "
                                f"{grp.capitalize()} & {info['mean']:.2f} & "
                            )
                            latex_str += (
                                f"[{info['hdi_3%']:.2f}, {info['hdi_97%']:.2f}] & "
                            )
                            # latex_str += f"{info['ess_tail']:.1f} & {info['r_hat']:.1f} & "
                            latex_str += (
                                f" \\multirow{{{2}}}{{*}}{{{info_diff['mean']:.2f}}} & "
                            )
                            latex_str += f" \\multirow{{{2}}}{{*}}{{{1 / info_diff['sddr']:.2f}}} & "
                            latex_str += f" \\multirow{{{2}}}{{*}}{{[{info_diff['hdi_3%']:.2f},{info_diff['hdi_97%']:.2f}]}} & "
                            if proba_diff > 0.95:
                                latex_str += f" \\multirow{{{2}}}{{*}}{{ \\textbf{{{proba_diff:.2f}}}}} \\\\ "
                            else:
                                latex_str += (
                                    f" \\multirow{{{2}}}{{*}}{{{proba_diff:.2f}}} \\\\ "
                                )
                            first_task_condition_entry = False
                        else:
                            latex_str += (
                                f" & {grp.capitalize()} & {info['mean']:.2f} & "
                            )
                            latex_str += (
                                f"[{info['hdi_3%']:.2f}, {info['hdi_97%']:.2f}] & "
                            )
                            # latex_str += f"{info['ess_tail']:.1f} & {info['r_hat']:.1f} & & &\\\\ "
                            latex_str += f" & & &\\\\ "
                        if (
                            grp == groups[-1]
                            and task_condition
                            == all_conditions[metric_type][task]["conditions"][-1]
                        ):
                            latex_str += "\\hline\n"
                        else:
                            if grp == groups[-1]:
                                latex_str += "\\cline{2-9}\n"
                            else:
                                latex_str += "\\cline{3-5}\n"
    latex_str += (
        "\\end{tabular}\n\\end{adjustbox}\n\\caption{Performance of "
        + task
        + " at "
        + time
        + "-test for "
        + rename_str_for_latex(model)
        + " model.}\n"
        "\\label{tab:my_label}\n\\end{table}"
    )
    return latex_str


def generate_latex_post_table(
    studies,
    config_fig,
    all_conditions,
    time="pre",
    model="hierar_binom_n_var",
    task="moteval",
):
    latex_str = (
        "\\begin{table}[h]\n\\centering\n\\begin{adjustbox}{width=1.5\\textwidth,center=\\textwidth}\n"
        "\\begin{tabular}{|c|c|c|M{3cm}|c|c|c|M{3.1cm}|c|c|c|}\n\\hline\n"
    )
    latex_str += (
        "\\textbf{Study} & \\textbf{Condition} & \\textbf{Group} & "
        "\\textbf{Within group mean} & \\textbf{HDI} & "
        "\\textbf{Within group P(effect)}"
        "& \\textbf{Within group BF10} &"
        "\\textbf{Mean posterior of between-group difference} & \\textbf{Between group HDI} &"
        " \\textbf{Between group P(effect)} & \\textbf{Between group BF10}"
        "\\\\ \\hline\n "
    )
    groups = ["zpdes", "baseline"]
    for study in studies:
        first_study_entry = True
        for metric_type in all_conditions.keys():
            if (
                task in all_conditions[metric_type]
                and model in config_fig[metric_type]["var_delta_btw_study"]
            ):
                rope_start, rope_end = (
                    config_fig[metric_type]["rope_start"],
                    config_fig[metric_type]["rope_end"],
                )
                for task_condition in all_conditions[metric_type][task]["conditions"]:
                    nb_row_per_study = len(
                        all_conditions[metric_type][task]["conditions"]
                    ) * len(groups)
                    # First get all metrics:
                    (
                        trace_summary,
                        evolution_trace_summary_diff,
                        var,
                        evolution_trace_diff,
                        BF,
                        diff_diff_summary,
                        diff_diff_proba,
                        diff_diff_BF,
                    ) = compute_metrics(
                        study,
                        task,
                        task_condition,
                        model,
                        config_fig,
                        metric_type,
                        time,
                    )
                    first_task_condition_entry = True
                    for grp in groups:
                        (
                            evolution_trace_diff,
                            evolution_info_diff,
                            evolution_proba_diff,
                        ) = get_evolution_diff(
                            model,
                            grp,
                            time,
                            evolution_trace_summary_diff,
                            var,
                            evolution_trace_diff,
                            config_fig,
                            metric_type,
                        )
                        if first_study_entry:
                            # Calculate the number of rows for this study entry
                            latex_str += f"\\multirow{{{nb_row_per_study}}}{{*}}{{{rename_str_for_latex(study)}}} & "
                            first_study_entry = False
                        else:
                            latex_str += " & "
                        if first_task_condition_entry:
                            # condition + group
                            latex_str += (
                                f" \\multirow{{{2}}}{{*}}{{{rename_str_for_latex(task_condition)}}} & "
                                f"{grp.capitalize()} &"
                            )
                            # Post - Pre for each group:
                            latex_str += f"{evolution_info_diff['mean']} & "
                            latex_str += f"[{evolution_info_diff['hdi_3%']}, {evolution_info_diff['hdi_97%']}] &"
                            if evolution_proba_diff > 0.5:
                                latex_str += (
                                    f" \\textbf{{{evolution_proba_diff:.2f}}} & "
                                )
                            else:
                                latex_str += f" {evolution_proba_diff:.2f} & "
                            latex_str += f"{BF[grp]:.2f} & "
                            # Diff evolution btw groups:
                            latex_str += f" \\multirow{{{2}}}{{*}}{{{diff_diff_summary['mean'].values[0]:.2f}}} & "
                            latex_str += f" \\multirow{{{2}}}{{*}}{{[{diff_diff_summary['hdi_3%'].values[0]:.2f}, {diff_diff_summary['hdi_97%'].values[0]:.2f}]}} & "
                            if diff_diff_proba > 0.5:
                                latex_str += (
                                    " \\multirow{2}{*}{\\textbf{"
                                    + f"{diff_diff_proba:.2f}"
                                    + "}} & "
                                )
                            else:
                                latex_str += (
                                    " \\multirow{2}{*}{"
                                    + f"{diff_diff_proba:.2f}"
                                    + "}  & "
                                )
                            latex_str += (
                                f" \\multirow{{{2}}}{{*}}{{{diff_diff_BF:.2f}}} \\\\"
                            )
                            first_task_condition_entry = False
                        else:
                            latex_str += f" & {grp.capitalize()} & {evolution_info_diff['mean']:.2f} & "
                            latex_str += f" [{evolution_info_diff['hdi_3%']:.2f}, {evolution_info_diff['hdi_97%']:.2f}] &"
                            if evolution_proba_diff > 0.5:
                                latex_str += (
                                    f" \\textbf{{{evolution_proba_diff:.2f}}} & "
                                )
                            else:
                                latex_str += f" {evolution_proba_diff:.2f} & "
                            latex_str += f"{BF[grp]:.2f} & "
                            latex_str += f" & & &\\\\ "
                        if (
                            grp == groups[-1]
                            and task_condition
                            == all_conditions[metric_type][task]["conditions"][-1]
                        ):
                            latex_str += "\\hline\n"
                        else:
                            if grp == groups[-1]:
                                latex_str += "\\cline{2-8}\n"
                            else:
                                latex_str += "\\cline{3-7}\n"
    latex_str += (
        "\\end{tabular}\n\\end{adjustbox}\n\\caption{Evolution of "
        + task
        + " performance for "
        + rename_str_for_latex(model)
        + " model.}\n"
        "\\label{tab:my_label}\n\\end{table}"
    )
    return latex_str


def compute_metrics(study, task, task_condition, model, config_fig, metric_type, time):
    path = f"outputs/{study}/cognitive_battery/{task}"
    trace = az.from_json(f"{path}/{task_condition}-{model}_inference_data.json")
    var = config_fig[metric_type]["var_delta_btw_study"][model]
    trace_var = trace.posterior[var]
    trace_summary = az.summary(trace_var)
    time_index = 0 if time == "pre" else 1
    # 1) Get all traces:
    # 1.2) POST - PRE within each group:
    evolution_trace_diff = trace_var[:, :, :, 1] - trace_var[:, :, :, 0]
    # Check if this sig:
    mu, sigma = config_fig[metric_type]["var_hyp_range_sddr"][model]
    params_sddr = {"mu": mu, "sigma": sigma}
    sddr_z, sddr_b = (
        get_SDDR(evolution_trace_diff[:, :, 1], var, params_sddr),
        get_SDDR(evolution_trace_diff[:, :, 0], var, params_sddr),
    )
    BF = {"zpdes": 1 / sddr_z, "baseline": 1 / sddr_b}
    # 1.3) Diff of diff (i.e POST-PRE diff between groups)
    # diff_diff_trace = (trace_var[:, :, 1, 1] - trace_var[:, :, 1, 0]) - (
    #         trace_var[:, :, 0, 1] - trace_var[:, :, 0, 0])
    # Evolution ZPDES - Evolution Baseline
    diff_diff_trace = evolution_trace_diff[:, :, 1] - evolution_trace_diff[:, :, 0]
    # 2) Get all summary statistics:
    # 2.2) POST - PRE within each group:
    evolution_trace_summary_diff = az.summary(evolution_trace_diff)
    # 2.3) Diff of diff (i.e POST-PRE diff between groups)
    diff_diff_summary = az.summary(diff_diff_trace)
    # diff_diff_proba = sum(
    #     [(sum(chain.values > 0)) / (len(chain.values)) for chain in diff_diff_trace]) / len(
    #     diff_diff_trace)
    diff_diff_proba = np.mean(
        [
            np.max([sum(chain.values > 0), sum(chain.values < 0)]) / (len(chain.values))
            for chain in diff_diff_trace
        ]
    )
    diff_diff_BF = 1 / get_SDDR(diff_diff_trace, var, params_sddr)
    return (
        trace_summary,
        evolution_trace_summary_diff,
        var,
        evolution_trace_diff,
        BF,
        diff_diff_summary,
        diff_diff_proba,
        diff_diff_BF,
    )


def get_evolution_diff(
    model,
    grp,
    time,
    evolution_trace_summary_diff,
    var,
    evolution_trace_diff,
    config_fig,
    metric_type,
):
    grp_index = 1 if grp == "zpdes" else 0
    if model == "switching_cost":
        tmp_grp = 1 if grp == "zpdes" else 0
        tmp_var = 1 if time == "post" else 0
        evolution_info_diff = evolution_trace_summary_diff.loc[f"{var}[{tmp_grp}]"]
    else:
        evolution_info_diff = evolution_trace_summary_diff.loc[f"{var}[{grp}]"]
    evolution_proba_diff = np.mean(
        [
            np.max([sum(chain.values > 0), sum(chain.values < 0)]) / (len(chain.values))
            for chain in evolution_trace_diff[:, :, grp_index]
        ]
    )
    return evolution_trace_diff, evolution_info_diff, evolution_proba_diff


def rename_str_for_latex(input_str):
    return input_str.replace("_", "\\_")
