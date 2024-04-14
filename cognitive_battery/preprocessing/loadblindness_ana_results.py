from .utils import *
from pathlib import Path


def compute_nearfarcond(row, ind_nearfar):
    """
        From the row of results, return the list of farcondition if elt is min/max in results_targetvalue
        The ind_nearfar 0 means near and 1 means far conditions.
    """
    results_responses = list(row["results_responses_pos"])
    results_targetvalue = list(row["results_target_distance"])
    min_tmp = min(results_targetvalue)
    targind_tmp = []
    targind_tmp = [0 if t == min_tmp else 1 for t in results_targetvalue]
    out = [results_responses[idx] for idx, elt in enumerate(targind_tmp) if elt == ind_nearfar]

    return np.array(out)


def parse_to_int(elt: str) -> int:
    """
        Parse string value into int, if string is null parse to 0
        If null; participant has not pressed the key when expected
    """
    if elt == '':
        return 0
    return int(elt)

def transform_string_to_row(row, column):
    return [int(elt) for elt in row[column].split(',') if elt]


def compute_sum_to_row(row, column):
    return np.sum(row[column])


def format_data(path):
    csv_path = f"{path}/loadblindness.csv"
    conditions_names = ['near', 'far', 'total-task']
    dataframe = pd.read_csv(csv_path, sep=",")
    # dataframe = delete_uncomplete_participants(dataframe)
    dataframe["results_responses_pos"] = dataframe.apply(
        lambda row: transform_string_to_row(row, "results_responses_pos"),
        axis=1)
    dataframe["results_target_distance"] = dataframe.apply(
        lambda row: transform_string_to_row(row, "results_target_distance"),
        axis=1)
    # For each condition:
    dataframe['far_response'] = dataframe.apply(lambda row: compute_nearfarcond(row, 1), axis=1)
    dataframe['near_response'] = dataframe.apply(lambda row: compute_nearfarcond(row, 0), axis=1)
    dataframe['far-correct'] = dataframe.apply(lambda row: compute_sum_to_row(row, "far_response"), axis=1)
    dataframe['near-correct'] = dataframe.apply(lambda row: compute_sum_to_row(row, "near_response"), axis=1)
    dataframe['near-nb'], dataframe['far-nb'] = dataframe['near_response'].apply(lambda row: len(row)), dataframe[
        'far_response'].apply(lambda row: len(row))
    dataframe['total_resp'] = dataframe.apply(lambda row: 20, axis=1)
    dataframe['near-accuracy'] = dataframe['near-correct'] / dataframe['near-nb']
    dataframe['far-accuracy'] = dataframe['far-correct'] / dataframe['far-nb']

    # Total task:
    dataframe['total-task-correct'] = dataframe['far-correct'] + dataframe['near-correct']
    dataframe['total-task-accuracy'] = (dataframe['near-accuracy'] + dataframe['far-accuracy']) / 2
    dataframe['total-task-nb'] = dataframe['near-nb'] + dataframe['far-nb']

    # nb_trials = len(dataframe['near_response'][0])
    # dataframe = dataframe[['participant_id', 'task_status', 'condition'] + conditions_names]
    base = ['participant_id', 'task_status', 'condition']
    condition_accuracy_names = [f"{elt}-accuracy" for elt in conditions_names]
    condition_correct_names = [f"{elt}-correct" for elt in conditions_names]
    condition_nb_names = [f"{elt}-nb" for elt in conditions_names]
    dataframe[base + condition_accuracy_names + condition_correct_names + condition_nb_names].to_csv(
        f'{path}/loadblindness_lfa.csv',
        index=False)


def preprocess_and_save(study):
    task = "loadblindness"
    savedir = f"../data/{study}/cognitive_battery"
    path = f"{savedir}/{task}"
    Path(savedir).mkdir(parents=True, exist_ok=True)
    format_data(path)