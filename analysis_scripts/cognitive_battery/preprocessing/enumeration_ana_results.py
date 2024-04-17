from analysis_scripts.cognitive_battery.preprocessing.utils import *
from pathlib import Path


# Treat data:
def compute_result_exact_answers(row):
    response = row["results_responses"].split(',')
    target = row["results_targetvalue"].split(',')
    return sum(x == y for x, y in zip(response, target))


def compute_mean_per_row(row):
    return_value = np.array(row['results_rt'].split(','), dtype=np.int32)
    return np.mean(return_value)


def compute_std_per_row(row):
    return_value = np.array(row['results_rt'].split(','), dtype=np.int32)
    return np.std(return_value)


def compute_result_exact_answers_list(row):
    """
    Returns a binary sucess for each trial
    """
    response = row["results_responses"].split(',')
    target = row["results_targetvalue"].split(',')
    out = [1 if x == y else 0 for x, y in zip(response, target)]
    return np.array(out)


def compute_numbercond(row, ind_cond):
    """
        From the row of results, return the list of resuluts_correct according to results_num_stim condition 
    """
    results_responses = list(row["result_correct"])
    results_targetvalue = [int(t) for t in row["results_targetvalue"].split(',')]
    out = [results_responses[ind] for ind, t in enumerate(results_targetvalue) if t == ind_cond]
    return np.array(out)


def format_data(path):
    # FIRST TREAT THE CSV AND PARSE IT TO DF
    task = "enumeration"
    # csv_path = f"outputs/{study}/results_{study}/{task}/{task}.csv"
    df = pd.read_csv(f"{path}/{task}.csv", sep=",")
    conditions = ["5", "6", "7", "8", "9"]
    df['result_response_exact'] = df.apply(compute_result_exact_answers, axis=1)
    df['mean_rt_session'] = df.apply(compute_mean_per_row, axis=1)
    df['std_rt_session'] = df.apply(compute_std_per_row, axis=1)
    # pre_response_exact = dataframe[dataframe['task_status'] == "PRE_TEST"]['result_response_exact'].values
    # post_response_exact = dataframe[dataframe['task_status'] == "POST_TEST"]['result_response_exact'].values
    # condition extraction - add to dataframe a column result_correct where each cell is a list of 0 - 1
    # (binary success for each trial)
    df['result_correct'] = df.apply(compute_result_exact_answers_list, axis=1)
    base = ['participant_id', 'task_status', 'condition']
    condition_accuracy = [f"{i}-accuracy" for i in conditions]
    condition_correct = [f"{i}-correct" for i in conditions]
    condition_nb = [f"{i}-nb" for i in conditions]
    # Let's sort the 'result_correct' column by condition:
    # For each condition we create a list of 0-1 binary success
    # And we compute the number of success for each condition in the column {condition}-sum:
    for condition in conditions:
        df[f"{condition}-results"] = df.apply(lambda row: compute_numbercond(row, int(condition)), axis=1)
        df[f"{condition}-nb"] = df.apply(lambda row: len(row[f"{condition}-results"]), axis=1)
        df[f"{condition}-correct"] = df.apply(lambda row: np.sum(row[condition + "-results"]), axis=1)
        df[f"{condition}-accuracy"] = df[f"{condition}-correct"] / df[f"{condition}-nb"]
    condition_accuracy.append("total-task-accuracy")
    condition_correct.append("total-task-correct")
    condition_nb.append("total-task-nb")
    df['total-task-correct'] = convert_to_global_task(df, [f'{cdt}-correct' for cdt in conditions])
    df['total-task-nb'] = 20 * len(conditions)
    df['total-task-accuracy'] = df['total-task-correct'] / df['total-task-nb']
    df = df[base + condition_correct + condition_accuracy + condition_nb]
    df = delete_uncomplete_participants(df)
    df.to_csv(f"{path}/{task}_lfa.csv", index=False)


def preprocess_and_save(study):
    task = "enumeration"
    savedir = f"data/{study}/cognitive_battery"
    path = f"{savedir}/{task}"
    Path(savedir).mkdir(parents=True, exist_ok=True)
    format_data(path)