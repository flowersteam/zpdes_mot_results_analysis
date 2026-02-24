# -*- coding: utf-8 -*-

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from .extract_sorted_memory import Results_memory
from analysis_scripts.cognitive_battery.preprocessing.utils import (
    convert_to_global_task,
    detect_outliers_and_clean,
)

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def compute_result_exact_answers(row: pd.Series) -> int:
    """Count exact matches between response list and target list."""
    resp_val = row.get("results_responses", np.nan)
    tgt_val = row.get("results_targetvalue", np.nan)

    if pd.isna(resp_val) or pd.isna(tgt_val):
        return 0

    response = str(resp_val).split(",")
    target = str(tgt_val).split(",")
    return sum(x == y for x, y in zip(response, target))


def delete_uncomplete_participants(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Remove participant_ids that appear fewer than 2 times."""
    vc = dataframe["participant_id"].value_counts(dropna=False)
    to_delete = vc[vc < 2].index

    out = dataframe.copy()
    out = out[~out["participant_id"].isin(to_delete)]
    return out


def extract_id(dataframe: pd.DataFrame, num_count: int) -> list:
    """
    Return participant_ids that appear exactly `num_count` times.
    Robust to pandas column naming changes (no intermediate DataFrame).
    """
    vc = dataframe["participant_id"].value_counts(dropna=False)
    return vc[vc == num_count].index.tolist()


def extract_mu_ci_from_summary_accuracy(dataframe, ind_cond):
    outs = np.zeros((len(ind_cond), 3))  # mu, ci_min, ci_max
    for t, ind in enumerate(ind_cond):
        outs[t, 0] = dataframe[ind].mu_theta
        outs[t, 1] = dataframe[ind].ci_min
        outs[t, 2] = dataframe[ind].ci_max
    return outs


def extract_mu_ci_from_summary_rt(dataframe, ind_cond):
    outs = np.zeros((len(ind_cond), 3))  # mu, ci_min, ci_max
    for t, ind in enumerate(ind_cond):
        outs[t, 0] = dataframe[ind].mu_rt
        outs[t, 1] = dataframe[ind].ci_min
        outs[t, 2] = dataframe[ind].ci_max
    return outs


# ---------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------


def treat_data(
    dataframe: pd.DataFrame,
    dataframe_2: pd.DataFrame,
    conditions_names: list,
):
    """
    Compute per-participant metrics and write them back into `dataframe`.
    Then remove the "useless part" (session 2 rows) without using a risky merge.

    Key fixes:
    - Normalize participant_id dtype (string) to avoid get_group KeyError.
    - Iterate PRE/POST based on the row's task_status, not enumerate order.
    - Avoid outer merge without explicit keys (can explode / create _x/_y columns).
    - Replace extract_id() internals with stable value_counts logic.
    """

    df = dataframe.copy()

    # Normalize ID dtype for stable grouping / matching
    df["participant_id"] = df["participant_id"].astype(str)
    dataframe_2 = dataframe_2.copy()
    dataframe_2["participant_id"] = dataframe_2["participant_id"].astype(str)

    # Participants that are present exactely 4 times:
    indices_id = extract_id(df, num_count=4)

    sum_observers = []

    # Pre-build groupby for efficiency + stability
    gb = df.groupby("participant_id", sort=False)

    # Iterate over participants:
    for ob in indices_id:
        ob = str(ob)

        # If for any reason group missing, skip safely
        if ob not in gb.groups:
            continue

        tmp_df = gb.get_group(ob)

        # Participant-level summary (using full tmp_df)
        tmp_results = Results_memory(tmp_df)
        sum_observers.append(
            np.concatenate(
                (
                    [ob],
                    tmp_results.out_mat_hit_miss_sum,
                    tmp_results.out_mat_fa_cr_sum,
                    tmp_results.out_mat_rt_cond,
                    tmp_results.out_mat_rt_cond_std,
                )
            )
        )

        # Row-level writeback: determine status from the row itself
        # so we do not rely on row ordering.
        for row_index, row in df.loc[df["participant_id"] == ob].iterrows():
            status = row.get("task_status", None)
            if status not in ("PRE_TEST", "POST_TEST"):
                # If unexpected status values exist, skip (or handle differently if needed)
                continue

            sub = tmp_df[tmp_df["task_status"] == status]
            if sub.empty:
                # Nothing to compute for that status; skip to avoid downstream crashes
                continue

            tmp_row = Results_memory(sub)

            for conditions_name in conditions_names:
                for condition_index, condition in enumerate(conditions_name):
                    if "hit" in condition:
                        tmp_cond = "out_mat_hit_miss_sum"
                    elif "fa" in condition:
                        tmp_cond = "out_mat_fa_cr_sum"
                    else:
                        tmp_cond = "out_mat_rt_cond"

                    # Ensure column exists before assignment
                    if condition not in df.columns:
                        df[condition] = np.nan

                    df.loc[row_index, condition] = tmp_row.__dict__[tmp_cond][
                        condition_index
                    ]

    # Remove session 2 rows safely instead of outer merge without keys
    # Goal in original comment: "delete the useless part"
    if "session" in df.columns:
        df = df[df["session"] != 2].copy()
    else:
        # Fallback: if session not available, remove rows coming from dataframe_2 by participant_id + task_status + condition
        # (explicit keys to avoid cartesian explosions)
        key_cols = [
            c
            for c in ["participant_id", "task_status", "condition"]
            if c in df.columns and c in dataframe_2.columns
        ]
        if key_cols:
            drop_keys = dataframe_2[key_cols].drop_duplicates()
            df = df.merge(drop_keys.assign(_drop=1), on=key_cols, how="left")
            df = df[df["_drop"].isna()].drop(columns=["_drop"])
        # else: nothing safe to do; keep df as-is

    return df, sum_observers


def delete_single_participants(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only participant_ids that appear exactly twice.
    (Kept as in your original code, but robust to dtype issues.)
    """
    out = df.copy()
    out["participant_id"] = out["participant_id"].astype(str)
    count_series = out["participant_id"].value_counts()
    valid_ids = count_series[count_series == 2].index
    return out[out["participant_id"].isin(valid_ids)].copy()


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------


def format_data(path: str):
    # Read inputs
    csv_path_short_range = f"{path}/memorability_1.csv"
    dataframe_short_range = pd.read_csv(csv_path_short_range)
    dataframe_short_range["session"] = 1

    csv_path_long_range = f"{path}/memorability_2.csv"
    dataframe_long_range = pd.read_csv(csv_path_long_range)
    dataframe_long_range["session"] = 2

    # Normalize types early (important for groupby/get_group)
    for df in (dataframe_short_range, dataframe_long_range):
        if "participant_id" in df.columns:
            df["participant_id"] = df["participant_id"].astype(str)

    # Concatenate
    dataframe = pd.concat(
        [dataframe_short_range, dataframe_long_range], axis=0, ignore_index=True
    )

    # Re-create conditions
    tmp_conditions = [*[f"{elt}" for elt in range(2, 6)], "100"]
    conditions_names_hit_miss = [f"{elt}-hit-miss" for elt in tmp_conditions]
    conditions_names_fa_cr = [f"{elt}-fa-cr" for elt in tmp_conditions]
    conditions_names_rt = [f"{cdt}-rt" for cdt in tmp_conditions]
    tmp_conditions_names = [
        conditions_names_hit_miss,
        conditions_names_fa_cr,
        conditions_names_rt,
    ]

    # Compute and remove session 2 rows
    dataframe, sum_observers = treat_data(
        dataframe, dataframe_long_range, tmp_conditions_names
    )

    # Keep only IDs that appear exactly twice
    dataframe = delete_single_participants(dataframe)

    # Rename columns
    for col in list(dataframe.columns):
        if "hit-miss" in col:
            dataframe = dataframe.rename(
                columns={col: col.replace("hit-miss", "hit-correct")}
            )
        if "fa-cr" in col:
            dataframe = dataframe.rename(
                columns={col: col.replace("fa-cr", "fa-correct")}
            )

    # Derived columns
    real_conditions = [f"{cdt}-hit" for cdt in tmp_conditions] + [
        f"{cdt}-fa" for cdt in tmp_conditions
    ]
    for condition in real_conditions:
        dataframe[f"{condition}-nb"] = 16
        dataframe[f"{condition}-accuracy"] = (
            dataframe[f"{condition}-correct"] / dataframe[f"{condition}-nb"]
        )

    # Final columns selection
    base = ["participant_id", "task_status", "condition"]
    all_conditions = [f"{cdt}-accuracy" for cdt in real_conditions]
    all_conditions += [f"{cdt}-correct" for cdt in real_conditions]
    all_conditions += [f"{cdt}-nb" for cdt in real_conditions]
    all_conditions += [f"{cdt}-rt" for cdt in tmp_conditions]

    # Guard if some columns are missing (e.g., upstream participant filtered)
    keep_cols = [c for c in base + all_conditions if c in dataframe.columns]
    dataframe = dataframe[keep_cols].copy()

    nb_participants_init = dataframe["participant_id"].nunique()

    # Global task metrics
    dataframe["total-task-hit-correct"] = convert_to_global_task(
        dataframe, [f"{cdt}-hit-correct" for cdt in tmp_conditions]
    )
    dataframe["total-task-fa-correct"] = convert_to_global_task(
        dataframe, [f"{cdt}-fa-correct" for cdt in tmp_conditions]
    )

    dataframe["total-task-hit-nb"] = 20 * len(tmp_conditions)
    dataframe["total-task-hit-accuracy"] = (
        dataframe["total-task-hit-correct"] / dataframe["total-task-hit-nb"]
    )
    dataframe["total-task-fa-accuracy"] = (
        dataframe["total-task-fa-correct"] / dataframe["total-task-hit-nb"]
    )

    dataframe["total-task-short-hit-correct"] = convert_to_global_task(
        dataframe, [f"{cdt}-hit-correct" for cdt in tmp_conditions if cdt != "100"]
    )
    dataframe["total-task-short-fa-correct"] = convert_to_global_task(
        dataframe, [f"{cdt}-fa-correct" for cdt in tmp_conditions if cdt != "100"]
    )

    dataframe["total-task-short-hit-nb"] = 20 * (len(tmp_conditions) - 1)
    dataframe["total-task-short-hit-accuracy"] = (
        dataframe["total-task-short-hit-correct"] / dataframe["total-task-short-hit-nb"]
    )
    dataframe["total-task-short-fa-accuracy"] = (
        dataframe["total-task-short-fa-correct"] / dataframe["total-task-short-hit-nb"]
    )

    # d' and criterion (total)
    dataframe["total-task-fa-accuracy"] = np.clip(
        dataframe["total-task-fa-accuracy"], 1e-6, 1 - 1e-6
    )
    dataframe["total-task-hit-accuracy"] = np.clip(
        dataframe["total-task-hit-accuracy"], 1e-6, 1 - 1e-6
    )
    dataframe["total-task-dprime"] = norm.ppf(
        dataframe["total-task-hit-accuracy"]
    ) - norm.ppf(dataframe["total-task-fa-accuracy"])
    dataframe["total-task-criterion"] = -0.5 * (
        norm.ppf(dataframe["total-task-fa-accuracy"])
        + norm.ppf(dataframe["total-task-hit-accuracy"])
    )

    # d' and criterion (short)
    dataframe["total-task-short-fa-accuracy"] = np.clip(
        dataframe["total-task-short-fa-accuracy"], 1e-6, 1 - 1e-6
    )
    dataframe["total-task-short-hit-accuracy"] = np.clip(
        dataframe["total-task-short-hit-accuracy"], 1e-6, 1 - 1e-6
    )
    dataframe["total-task-short-dprime"] = norm.ppf(
        dataframe["total-task-short-hit-accuracy"]
    ) - norm.ppf(dataframe["total-task-short-fa-accuracy"])
    dataframe["total-task-short-criterion"] = -0.5 * (
        norm.ppf(dataframe["total-task-short-fa-accuracy"])
        + norm.ppf(dataframe["total-task-short-hit-accuracy"])
    )

    # d' and criterion (100)
    if (
        "100-fa-accuracy" in dataframe.columns
        and "100-hit-accuracy" in dataframe.columns
    ):
        dataframe["100-fa-accuracy"] = np.clip(
            dataframe["100-fa-accuracy"], 1e-6, 1 - 1e-6
        )
        dataframe["100-hit-accuracy"] = np.clip(
            dataframe["100-hit-accuracy"], 1e-6, 1 - 1e-6
        )
        dataframe["100-dprime"] = norm.ppf(dataframe["100-hit-accuracy"]) - norm.ppf(
            dataframe["100-fa-accuracy"]
        )
        dataframe["100-criterion"] = -0.5 * (
            norm.ppf(dataframe["100-fa-accuracy"])
            + norm.ppf(dataframe["100-hit-accuracy"])
        )

    # RT aggregates
    rt_cols = [col for col in dataframe.columns if col.endswith("-rt")]
    if rt_cols:
        dataframe["total-task-hit-rt"] = dataframe[rt_cols].mean(axis=1)

    rt_cols_short = [col for col in rt_cols if "100" not in col]
    if rt_cols_short:
        dataframe["total-task-short-hit-rt"] = dataframe[rt_cols_short].mean(axis=1)

    # Outlier cleaning (as in your code)
    dataframe = detect_outliers_and_clean(dataframe, "total-task-hit-accuracy")
    dataframe = detect_outliers_and_clean(dataframe, "total-task-short-criterion")

    print(
        f"Memorability, proportion removed: {dataframe['participant_id'].nunique()} / {nb_participants_init} "
    )
    dataframe.to_csv(f"{path}/memorability_lfa.csv", index=False)


def preprocess_and_save(study: str):
    task = "memorability"
    savedir = f"data/{study}/cognitive_battery"
    path = f"{savedir}/{task}"
    Path(savedir).mkdir(parents=True, exist_ok=True)
    format_data(path)
