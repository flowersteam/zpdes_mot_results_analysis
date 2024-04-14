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
