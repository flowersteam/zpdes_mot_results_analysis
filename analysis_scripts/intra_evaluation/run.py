import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from pathlib import Path

from analysis_scripts.intra_evaluation.utils import parse_trajectory
from analysis_scripts.intra_evaluation.plot import (
    plot_scatter_per_activity_on_ax,
    plot_regression_line_per_activity_on_ax,
    plot_mean_intra,
)
from analysis_scripts.intra_evaluation.fit_models import (
    get_bayesian_mixed_lm_model,
    run_anonva_per_activity,
    get_average_mixed_lm_model,
)


def get_group_trajectory(df, size, get_log=False, fit_models=True):
    nb_activity_per_evaluation = 12
    points = [[None for i in range(size)] for j in range(nb_activity_per_evaluation)]
    means, slopes = [], []
    for activity_index in range(nb_activity_per_evaluation):
        # First get all points
        for participant_idx, participant in enumerate(df["trajectory"]):
            participant_traj = participant.reshape(4, 12)
            points[activity_index][participant_idx] = participant_traj[
                :, activity_index
            ].tolist()
        # Then get means:
        tmp_data = np.array(points[activity_index])
        means.append(tmp_data.mean(axis=0))
        # Finally get slopes:
        n_participants, n_measurements = tmp_data.shape
        df_long = pd.DataFrame(
            {
                "participant_id": np.repeat(np.arange(n_participants), n_measurements),
                "time": np.tile(np.arange(n_measurements), n_participants),
                "measurement": tmp_data.flatten(),
            }
        )
        if fit_models:
            # Define and fit the mixed-effects linear model with random slopes for time
            mdf_no_random_slopes = smf.mixedlm(
                "measurement ~ time", df_long, groups=df_long["participant_id"]
            ).fit()
            # Fit bayesian models
            if get_log:
                mdf = smf.mixedlm(
                    "measurement ~ time",
                    df_long,
                    groups=df_long["participant_id"],
                    re_formula="~time",
                ).fit()
                # Fit the model without random slopes
                print(f"Activity {activity_index}")
                print(mdf_no_random_slopes.summary())
                print(mdf.summary())
            slopes.append(
                [
                    mdf_no_random_slopes.params["Intercept"],
                    mdf_no_random_slopes.params["time"],
                ]
            )
    return points, means, slopes


def study_success_in_each_cell(
    df,
    ylims=(0, 5),
    yticks=[0, 1, 2, 3, 4],
    row_vertical_val=2.5,
    title="All participants",
    path_to_store="outputs/",
):
    """
    This method save the scatter plot for each activity of the intra-training and color dots with group belonging
    + compute a regression slope for each group and plot the corresponding line
    + Test for each activity whether significant interactions
    """
    save_string = ""
    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(15, 10))
    fig.subplots_adjust(wspace=0, hspace=0)
    conditions = {
        "zpdes": {"points": [], "means": [], "slopes": []},
        "baseline": {"points": [], "means": [], "slopes": []},
    }
    # Here is really the part used for plot:
    for condition, condition_dict in conditions.items():
        # First restrict to only a specific condition:
        tmp_data = df[df["condition"] == condition]
        tmp_size = len(tmp_data)
        # Then get an array of size 48 where each element is of size 4 x len(group)
        condition_dict["points"], condition_dict["means"], condition_dict["slopes"] = (
            get_group_trajectory(tmp_data, tmp_size)
        )
        # After getting the data just plot it on the ax:
        for idx, ax in enumerate(axes.flatten()):
            plot_scatter_per_activity_on_ax(
                ax, idx, condition_dict["points"], tmp_size, condition
            )
            plot_regression_line_per_activity_on_ax(
                ax, idx, condition_dict["slopes"], condition
            )
    for idx, ax in enumerate(axes.flatten()):
        # Use this loop to set the axis in correct display mode:
        ax.set_xticks([0, 1, 2, 3], ["S1", "S4", "S5", "S9"])
        ax.set_ylim(ylims)
        ax.set_yticks(yticks)
        if idx % 4 != 0:
            ax.set_yticks([])
        if idx in [1, 2, 3, 5, 6, 7]:
            ax.set_xticks([])
        cols = [
            "speed=easy, dist=easy",
            "speed=hard, dist=easy",
            "speed=easy, dist=hard",
            "speed=hard, dist=hard",
        ]
        if idx in [0, 1, 2, 3]:
            ax.xaxis.set_label_position("top")
            ax.xaxis.tick_top()
            ax.set_xticks([1.5])  # Assuming 5 is the center for the x-axis
            ax.set_xticklabels([cols[idx]], va="bottom")
        rows = ["nb_targets=easy", "nb_targets=medium", "nb_targets=hard"]
        if idx in [3, 7, 11]:
            ax.yaxis.set_label_position("right")
            ax.yaxis.tick_right()
            ax.set_yticks([row_vertical_val])  # Assuming 5 is the center for the x-axis
            ax.set_yticklabels([rows[idx % 3]], va="center")

        # Fit an ANOVA to look for sig difference
        # If sig diff, add on ax:
        p_val_group, p_val_time, p_val_interaction = run_anonva_per_activity(
            conditions, idx
        )
        dict, tmp_str = get_bayesian_mixed_lm_model(conditions, idx)
        save_string += f"Activity index: {idx} \n" + tmp_str + "\n"
        # 'group_slope', 'session_id_slope', 'group_session_interaction'
        # ax.text(0.5, 0.95,
        #         f'Grp HDI=[{dict["group_slope"]["3%"]:.2f}, {dict["group_slope"]["97%"]:.2f}]- BF={dict["group_slope"]["BF"][0]:.2f}, \n'
        #         f'Time HDI=[{dict["session_id_slope"]["3%"]:.2f}, {dict["session_id_slope"]["97%"]:.2f}] - BF={dict["session_id_slope"]["BF"][0]:.2f} \n'
        #         f'Inter HDI=[{dict["group_session_interaction"]["3%"]:.2f}, {dict["group_session_interaction"]["97%"]:.2f}] - BF={dict["group_session_interaction"]["BF"][0]:.2f}',
        #         ha='center', va='top',
        #         transform=ax.transAxes)
    fig.suptitle(
        title
        + f" (ZPDES={len(df[df['condition'] == 'zpdes'])}, Baseline={len(df[df['condition'] == 'baseline'])})"
    )
    plt.savefig(f"{path_to_store}/{title}-in-cell.png")
    with open(f"{path_to_store}/{title}_data.txt", "w") as file:
        file.write(save_string)


def get_average_mixed_model(df, path_to_store, seed):
    _, save_string = get_average_mixed_lm_model(df, seed)
    with open(f"{path_to_store}/average_data.txt", "w") as file:
        file.write(save_string)


def get_intra_in_cluster(df_data, ylims, yticks, row_vertical_val, path_to_store):
    # Calculate the third and sixth deciles
    third_decile = df_data["s_0"].quantile(0.3)
    sixth_decile = df_data["s_0"].quantile(0.6)
    # Split the DataFrame into two based on the median score
    low_performer = df_data[df_data["s_0"] <= third_decile]
    medium_performer = df_data[
        (df_data["s_0"] > third_decile) & (df_data["s_0"] <= sixth_decile)
    ]
    high_performer = df_data[df_data["s_0"] > sixth_decile]
    study_success_in_each_cell(
        low_performer,
        title="Low performers",
        ylims=ylims,
        yticks=yticks,
        row_vertical_val=row_vertical_val,
        path_to_store=path_to_store,
    )
    study_success_in_each_cell(
        medium_performer,
        title="Medium performers",
        ylims=ylims,
        yticks=yticks,
        row_vertical_val=row_vertical_val,
        path_to_store=path_to_store,
    )
    study_success_in_each_cell(
        high_performer,
        title="High performers",
        ylims=ylims,
        yticks=yticks,
        row_vertical_val=row_vertical_val,
        path_to_store=path_to_store,
    )


def run_intra_evals(studies, metric_type, seed=42):
    for study in studies:
        df_data = pd.read_csv(
            f"data/{study}/intra_evaluation/{study}_{metric_type}_intra.csv"
        )
        df_data["trajectory"] = df_data.apply(parse_trajectory, axis=1)
        if metric_type == "F1":
            df_data["trajectory"] = df_data.apply(
                lambda x: np.array(x["trajectory"]) / 4, axis=1
            )
            yticks = [0, 0.2, 0.4, 0.6, 0.8, 1]
            row_vertical_val = 0.6
            ylims = (0, 1.2)
        else:
            yticks = [0, 1, 2, 3, 4]
            row_vertical_val = 2.5
            ylims = (0, 5)
        path_to_store = f"outputs/{study}/intra_evaluation/"
        Path(path_to_store).mkdir(parents=True, exist_ok=True)
        # Plot descriptive plots:
        plot_mean_intra(
            df_data, path=path_to_store, metric_type=metric_type, study=study
        )
        get_average_mixed_model(df_data, path_to_store, seed=seed)
        study_success_in_each_cell(
            df_data,
            title="All participants",
            ylims=ylims,
            yticks=yticks,
            row_vertical_val=row_vertical_val,
            path_to_store=path_to_store,
        )
        # # Additionnal evaluation for subgroups
        get_intra_in_cluster(
            df_data, ylims, yticks, row_vertical_val, path_to_store=path_to_store
        )
