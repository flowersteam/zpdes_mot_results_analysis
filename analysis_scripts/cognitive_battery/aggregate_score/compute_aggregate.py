import numpy as np
import pandas as pd
from pingouin import ttest
from pathlib import Path

from code.cognitive_battery.PCA.utils import get_df
from code.cognitive_battery.PCA.normalize import normalize_data


def run_broad_aggregate_difference(tasks, study, kept_columns, tasks_nb, expe_name, normalization_type="zscore"):
    df = get_df([study])
    # With all data:
    df = df[kept_columns + tasks + tasks_nb]
    df.sort_values(by=['participant_id', 'task_status'], inplace=True)
    path_to_store = f"../outputs/{study}/aggregate_score/{expe_name}/"
    Path(path_to_store).mkdir(parents=True, exist_ok=True)

    # First for RT values, just take the inverse:
    rt_cols = [col for col in df.columns if "rt" in col and "cost" not in col and "participant" not in col]
    df[rt_cols] = df[rt_cols].replace(0, np.nan).apply(lambda x: 1 / x)

    # Then normalize the data using z-scores
    df[tasks] = df[tasks].fillna(df[tasks].mean())
    df[tasks] = normalize_data(df, tasks, tasks, mode=normalization_type, shuffle=False)

    # Compute the diff within group and between groups
    diff_zpdes, diff_baseline, diff_pre, diff_post = compute_diff(df, tasks)

    # Save the differences
    save_aggregate(diff_zpdes, diff_baseline, diff_pre, diff_post, path_to_store, normalization_type, study)


def compute_diff(df, tasks):
    # Split into pre-post
    pre, post = df[df['task_status'] == "PRE_TEST"], df[df['task_status'] == "POST_TEST"]
    # Split into baseline-zpdes
    zpdes_pre, zpdes_post = pre[pre['condition'] == "zpdes"][tasks], post[post['condition'] == "zpdes"][tasks]
    baseline_pre, baseline_post = pre[pre['condition'] == "baseline"][tasks], post[post['condition'] == "baseline"][
        tasks]
    # Compute the mean z-score for each participant
    mean_pre_zpdes, mean_post_zpdes = zpdes_pre.mean(axis=1).astype(float), zpdes_post.mean(axis=1).astype(float)
    mean_pre_baseline, mean_post_baseline = baseline_pre.mean(axis=1).astype(float), baseline_post.mean(axis=1).astype(
        float)
    diff_zpdes = ttest(mean_pre_zpdes, mean_post_zpdes, paired=True)
    diff_baseline = ttest(mean_pre_baseline, mean_post_baseline, paired=True)
    diff_pre = ttest(pd.to_numeric(mean_pre_zpdes, errors='coerce'),
                     pd.to_numeric(mean_pre_baseline, errors='coerce'), paired=False)
    diff_post = ttest(pd.to_numeric(mean_post_zpdes, errors='coerce'),
                      pd.to_numeric(mean_post_baseline, errors='coerce'), paired=False)
    return diff_zpdes, diff_baseline, diff_pre, diff_post


def save_aggregate(diff_zpdes, diff_baseline, diff_pre, diff_post, path_to_store, normalization_type, study):
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
    save.to_csv(f"{path_to_store}/info_aggregate-{normalization_type}-{study}.csv")
