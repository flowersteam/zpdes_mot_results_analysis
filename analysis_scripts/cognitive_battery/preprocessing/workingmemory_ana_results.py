from analysis_scripts.cognitive_battery.preprocessing.utils import *
from pathlib import Path


def delete_uncomplete_participants(dataframe):
    """

    """
    mask = pd.DataFrame(dataframe.participant_id.value_counts() < 2)
    participants_to_delete = mask[mask['participant_id'] == True].index.tolist()
    for id in participants_to_delete:
        dataframe = dataframe[dataframe['participant_id'] != id]
    return dataframe


def transform_string_to_row(row, column):
    return [int(elt) for elt in row[column].split(',') if elt]


def compute_numbercond(row, ind_cond):
    """
        From the row of results, return the list of resuluts_correct according to results_num_stim condition 
    """
    results_responses = list(row["results_correct"])
    results_targetvalue = list(row["results_num_stim"])
    out = []
    out = [results_responses[ind] for ind, t in enumerate(results_targetvalue) if t == ind_cond]
    return np.array(out)


def compute_sum_to_row(row, column):
    return np.sum(row[column])


def extract_id(dataframe, num_count):
    mask = pd.DataFrame(dataframe.participant_id.value_counts() == num_count)
    indices_id = mask[mask['participant_id'] == True].index.tolist()
    return indices_id


def format_data(path):
    task = "workingmemory"
    # FIRST TREAT THE CSV AND PARSE IT TO DF
    csv_path = f"{path}/{task}.csv"
    dataframe = pd.read_csv(csv_path, sep=",")
    # Conditions:
    number_condition = [4, 5, 6, 7, 8]
    # Few pre-processing
    # dataframe = delete_uncomplete_participants(dataframe)
    dataframe["results_correct"] = dataframe.apply(lambda row: transform_string_to_row(row, "results_correct"),
                                                   axis=1)
    dataframe["results_num_stim"] = dataframe.apply(lambda row: transform_string_to_row(row, "results_num_stim"),
                                                    axis=1)
    # Other pre-processing (get accuracies and nb_correct
    for t in number_condition:
        dataframe[str(t)] = dataframe.apply(lambda row: compute_numbercond(row, t), axis=1)
        dataframe[f'{t}-correct'] = dataframe.apply(lambda row: compute_sum_to_row(row, str(t)), axis=1)
        dataframe[f'{t}-nb'] = dataframe.apply(lambda row: len(row[str(t)]), axis=1)
        dataframe[f'{t}-accuracy'] = dataframe[f'{t}-correct'] / dataframe[f'{t}-nb']
    dataframe['total-task-correct'] = convert_to_global_task(dataframe, [f'{col}-correct' for col in number_condition])
    dataframe['total-task-nb'] = 12 * len(number_condition)
    dataframe['total-task-accuracy'] = dataframe['total-task-correct'] / dataframe['total-task-nb']
    condition_accuracy_names = [f"{elt}-accuracy" for elt in number_condition] + ['total-task-accuracy']
    condition_correct_names = [f"{elt}-correct" for elt in number_condition] + ['total-task-correct']
    condition_nb_names = [f"{elt}-nb" for elt in number_condition] + ['total-task-nb']
    base = ['participant_id', 'task_status', 'condition']
    dataframe = dataframe[base + condition_accuracy_names + condition_correct_names + condition_nb_names]
    # If save_mode, store the dataframe into csv:
    dataframe.to_csv(f'{path}/workingmemory_lfa.csv', index=False)


def preprocess_and_save(study):
    task = "workingmemory"
    savedir = f"data/{study}/cognitive_battery"
    path = f"{savedir}/{task}"
    Path(savedir).mkdir(parents=True, exist_ok=True)
    format_data(path)
