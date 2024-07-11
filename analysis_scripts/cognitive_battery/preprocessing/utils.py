import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MultipleLocator
import copy


def extract_mu_ci_from_summary_accuracy(dataframe, ind_cond):
    # 3 means the mu, ci_min, and ci_max
    out = np.zeros((len(ind_cond), 3))
    for t, ind in enumerate(ind_cond):
        out[t, 0] = dataframe[ind].mu_theta
        out[t, 1] = dataframe[ind].ci_min
        out[t, 2] = dataframe[ind].ci_max
    return out


def extract_mu_ci_from_summary_rt(dataframe, ind_cond):
    outs = np.zeros((len(ind_cond), 3))  # 3 means the mu, ci_min, and ci_max
    for index, _ in enumerate(ind_cond):
        outs[index, 0] = dataframe[index].mu_rt
        outs[index, 1] = dataframe[index].ci_min
        outs[index, 2] = dataframe[index].ci_max
    return outs


def transform_accuracy_to_nb_success(dataframe_list, outcomes_names):
    for df in dataframe_list:
        for col in outcomes_names:
            df[col] = df[col] * df['total_resp']


def get_pre_post_dataframe(dataframe, outcomes_names):
    # Divide in pre_test and post_test
    pretest = dataframe[dataframe['task_status'] == 'PRE_TEST'][outcomes_names]
    posttest = dataframe[dataframe['task_status'] == 'POST_TEST'][outcomes_names]
    return pretest, posttest


def get_overall_dataframe_accuracy(dataframe, outcomes_names):
    # summarize two days experiments
    indices_id = extract_id(dataframe, num_count=2)
    sum_observers = []
    for ob in indices_id:
        tmp_df = dataframe.groupby(["participant_id"]).get_group(ob)
        sum_observers.append([np.sum(tmp_df[index]) for index in outcomes_names])
    sum_observers = pd.DataFrame(sum_observers, columns=outcomes_names)
    return sum_observers


def get_overall_dataframe_rt(dataframe, outcomes_names):
    # summarize two days experiments
    indices_id = extract_id(dataframe, num_count=2)
    sum_observers = []
    for ob in indices_id:
        tmp_df = dataframe.groupby(["participant_id"]).get_group(ob)
        sum_observers.append([np.mean(tmp_df[index]) for index in outcomes_names])
    sum_observers = pd.DataFrame(sum_observers, columns=outcomes_names)
    return sum_observers


def transform_str_to_list(row, columns):
    for column in columns:
        if column in row:
            row[column] = row[column].split(",")
    return row


def delete_uncomplete_participants(dataframe: pd.DataFrame) -> pd.DataFrame:
    """

    """
    mask = pd.DataFrame(dataframe.participant_id.value_counts() < 2)
    participants_to_delete = mask[mask['count'] == True].index.tolist()
    for id in participants_to_delete:
        dataframe = dataframe[dataframe['participant_id'] != id]
    return dataframe


def extract_id(dataframe, num_count):
    """
    returns: List of all participants_id
    """
    mask = pd.DataFrame(dataframe.participant_id.value_counts() == num_count)
    indices_id = mask[mask['participant_id'] == True].index.tolist()
    return indices_id


def change_accuracy_for_correct_column(column_name: str):
    return column_name.replace('accuracy', 'correct')


def convert_to_global_task(task, conditions):
    return task[conditions].sum(axis=1)


def detect_outliers_and_clean(df, condition):
    def detect_outliers_iqr(series):
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 2 * IQR
        upper_bound = Q3 + 2 * IQR
        return series[(series < lower_bound) | (series > upper_bound)]

    # Detect outliers in pre-test
    pre = df[df['task_status'] == 'PRE_TEST']
    pre_outliers = detect_outliers_iqr(pre[condition])
    pre_outlier_ids = pre[pre[condition].isin(pre_outliers)].participant_id.unique()

    if len(pre_outlier_ids) > 0:
        print(f"{condition}: Outliers to remove because of pre-test: {pre_outlier_ids}")

    # Exclude pre-test outliers
    df_cleaned = df[~df['participant_id'].isin(pre_outlier_ids)]

    # Detect outliers in post-test
    post = df[df['task_status'] == 'POST_TEST']
    post_outliers = detect_outliers_iqr(post[condition])
    post_outliers_ids = post[post[condition].isin(post_outliers)].participant_id.unique()

    if len(post_outliers_ids) > 0:
        print(f"{condition}: Outliers to remove because of post-test: {post_outliers_ids}")

    # Exclude pre-test outliers
    df_cleaned = df[~df['participant_id'].isin(post_outliers_ids)]

    # Calculate change and detect outliers in change
    pre_cleaned = df_cleaned[df_cleaned['task_status'] == 'PRE_TEST']
    post_cleaned = df_cleaned[df_cleaned['task_status'] == 'POST_TEST']
    df_cleaned.loc[df_cleaned['task_status'] == 'POST_TEST', 'change'] = post_cleaned[condition].values - pre_cleaned[
        condition].values

    change_outliers = detect_outliers_iqr(df_cleaned[df_cleaned['task_status'] == 'POST_TEST']['change'])
    change_outlier_ids = df_cleaned[df_cleaned['change'].isin(change_outliers)].participant_id.unique()

    # Exclude change outliers
    df_cleaned = df_cleaned[~df_cleaned['participant_id'].isin(change_outlier_ids)]

    if len(change_outlier_ids) > 0:
        print(f"{condition}: Outliers to remove because of change: {change_outlier_ids}")

    # Drop the change column if not needed
    df_cleaned.drop(columns=['change'], inplace=True, errors='ignore')

    return df_cleaned