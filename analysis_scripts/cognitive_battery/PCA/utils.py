import copy

import pandas as pd


def get_df(studies, tasks, tasks_nb):
    tmp_all_df = []
    to_keep = ["participant_id", "condition", "task_status"]
    for study in studies:
        df = add_columns_for_study(study, tasks, to_keep, df=pd.DataFrame())
        df = add_columns_for_study(study, tasks_nb, to_keep, df=df)
        # Add the study column to the dataframe for the current study
        df["study"] = study
        tmp_all_df.append(df)
    # Append the study-specific dataframe to the final dataframe
    return pd.concat(tmp_all_df, ignore_index=True)


def add_columns_for_study(study, tasks, to_keep, df):
    for task in tasks:
        # 1- retrieve task
        task_name = task.split("_")[0]
        condition = "_".join(task.split("_")[1:])
        tmp = pd.read_csv(
            f"data/{study}/cognitive_battery/{task_name}/{task_name}_lfa.csv"
        )
        tmp = tmp[[*to_keep, condition]]
        tmp = tmp.rename(columns={condition: f"{task_name}_{condition}"})
        # Merge with the existing dataframe
        if df.empty:
            df = tmp
        else:
            # Merge using the to_keep columns to ensure consistency and alignment
            df = pd.merge(df, tmp, on=to_keep, how="outer")
    return df
