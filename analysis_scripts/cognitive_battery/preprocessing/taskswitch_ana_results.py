from analysis_scripts.cognitive_battery.preprocessing.utils import *
from pathlib import Path
from analysis_scripts.cognitive_battery.preprocessing.utils import detect_outliers_and_clean

# keyRes1 = F => 1 (ODD impair - LOW)
# keyRes2 = J => 2 (EVEN pair - HIGH)
# task1 = parity (0)
# task2 = relative (1)

def delete_uncomplete_participants(dataframe):
    """

    """
    mask = pd.DataFrame(dataframe.participant_id.value_counts() < 2)
    participants_to_delete = mask[mask['participant_id'] == True].index.tolist()
    for id in participants_to_delete:
        dataframe = dataframe[dataframe['participant_id'] != id]
    return dataframe

def delete_beggining_of_block(row):
    results = row["results_ind_switch"].split(",")
    results = [int(elt) for elt in results]
    new_row = copy.deepcopy(results)
    # for idx, elt in enumerate(results):
    # if idx % 33 == 0:
    # new_row[idx] = 0
    return new_row

def transform_string_to_row(row, column):
    return [int(elt) for elt in row[column].split(',') if elt]

def correct_sequence_of_answers(row):
    seq_answer_relative = []
    seq_answer_parity = []
    seq_relative_switch = []
    seq_parity_switch = []
    seq_relative_rt = []
    seq_parity_rt = []
    for response, task, target, switch, rt, ind in zip(row.results_responses, row.results_indtask,
                                                       row.results_trial_target,
                                                       row.results_ind_switch_clean, row.results_rt,
                                                       range(len(row.results_ind_switch_clean))):

        if ind != 0 and ind != 33 and ind != 66:  # to exclude the first trials
            # First check what activity is requested - if None => do not consider the trial
            if task == 1:
                seq_relative_switch.append(row.results_ind_switch_clean[ind - 1])
                seq_relative_rt.append(rt)
                if (response == 1 and target < 5) or (response == 2 and target > 5):
                    seq_answer_relative.append(1)
                else:
                    seq_answer_relative.append(0)
            elif task == 0:
                seq_parity_switch.append(row.results_ind_switch_clean[ind - 1])
                seq_parity_rt.append(rt)
                if (response == 1 and (target % 2) == 1) or (response == 2 and (target % 2) == 0):
                    seq_answer_parity.append(1)
                else:
                    seq_answer_parity.append(0)
    return seq_answer_relative, seq_answer_parity, seq_relative_switch, seq_parity_switch, seq_relative_rt, seq_parity_rt

def compute_correct_answer(row, answer_type):
    seq_answer_relative, seq_answer_parity, seq_relative_switch, seq_parity_switch, seq_relative_rt, seq_parity_rt = correct_sequence_of_answers(
        row)
    if answer_type == "correct_total":
        return seq_answer_relative.count(1) + seq_answer_parity.count(1)
    elif answer_type == "correct_relative":
        return seq_answer_relative.count(1)
    elif answer_type == "correct_parity":
        return seq_answer_parity.count(1)
    elif answer_type == "total_nb":
        return len(seq_answer_parity) + len(seq_answer_relative)
    elif answer_type == "parity_nb":
        return len(seq_answer_parity)
    elif answer_type == "relative_nb":
        return len(seq_answer_relative)
    elif answer_type == "check_switch":
        parity_errors_switch = sum(
            [1 for elt, sw in zip(seq_answer_parity, seq_parity_switch) if (sw == 1 and elt == 0)])
        relative_errors_switch = sum(
            [1 for elt, sw in zip(seq_answer_relative, seq_relative_switch) if (sw == 1 and elt == 0)])
        return parity_errors_switch + relative_errors_switch
    # summarize the relative and parity condition for accuracy
    elif answer_type == "check_switch_hit":
        parity_hit_switch = sum(
            [1 for elt, sw in zip(seq_answer_parity, seq_parity_switch) if (sw == 1 and elt == 1)])
        relative_hit_switch = sum(
            [1 for elt, sw in zip(seq_answer_relative, seq_relative_switch) if (sw == 1 and elt == 1)])
        return parity_hit_switch + relative_hit_switch
    elif answer_type == "check_unswitch_hit":
        parity_hit_unswitch = sum(
            [1 for elt, sw in zip(seq_answer_parity, seq_parity_switch) if (sw == 0 and elt == 1)])
        relative_hit_unswitch = sum(
            [1 for elt, sw in zip(seq_answer_relative, seq_relative_switch) if (sw == 0 and elt == 1)])
        return parity_hit_unswitch + relative_hit_unswitch
    # separate the relative and parity condition for accuracy
    elif answer_type == "parity_check_switch_hit":
        parity_hit_switch = sum(
            [1 for elt, sw in zip(seq_answer_parity, seq_parity_switch) if (sw == 1 and elt == 1)])
        return parity_hit_switch
    elif answer_type == "relative_check_switch_hit":
        relative_hit_switch = sum(
            [1 for elt, sw in zip(seq_answer_relative, seq_relative_switch) if (sw == 1 and elt == 1)])
        return relative_hit_switch
    elif answer_type == "parity_check_unswitch_hit":
        parity_hit_unswitch = sum(
            [1 for elt, sw in zip(seq_answer_parity, seq_parity_switch) if (sw == 0 and elt == 1)])
        return parity_hit_unswitch
    elif answer_type == "relative_check_unswitch_hit":
        relative_hit_unswitch = sum(
            [1 for elt, sw in zip(seq_answer_relative, seq_relative_switch) if (sw == 0 and elt == 1)])
        return relative_hit_unswitch
    # total number for each conditions
    elif answer_type == "parity_check_switch_total":
        parity_total_switch = sum(
            [1 for elt, sw in zip(seq_answer_parity, seq_parity_switch) if (sw == 1)])
        return parity_total_switch
    elif answer_type == "relative_check_switch_total":
        relative_total_switch = sum(
            [1 for elt, sw in zip(seq_answer_relative, seq_relative_switch) if (sw == 1)])
        return relative_total_switch
    elif answer_type == "parity_check_unswitch_total":
        parity_total_unswitch = sum(
            [1 for elt, sw in zip(seq_answer_parity, seq_parity_switch) if (sw == 0)])
        return parity_total_unswitch
    elif answer_type == "relative_check_unswitch_total":
        relative_hit_unswitch = sum(
            [1 for elt, sw in zip(seq_answer_relative, seq_relative_switch) if (sw == 0)])
        return relative_hit_unswitch
    # summarize the relative and parity condition for rt
    elif answer_type == "check_switch_rt":
        parity_rt_switch = sum(
            [rt for elt, rt, sw in zip(seq_answer_parity, seq_parity_rt, seq_parity_switch) if (sw == 1)])
        relative_rt_switch = sum(
            [rt for elt, rt, sw in zip(seq_answer_relative, seq_relative_rt, seq_relative_switch) if (sw == 1)])
        try:
            return (parity_rt_switch + relative_rt_switch) / (len(seq_parity_rt) + len(seq_relative_rt))
        except:
            print("ag")
    elif answer_type == "check_unswitch_rt":
        parity_rt_unswitch = sum(
            [rt for elt, rt, sw in zip(seq_answer_parity, seq_parity_rt, seq_parity_switch) if (sw == 0 and elt == 1)])
        relative_rt_unswitch = sum(
            [rt for elt, rt, sw in zip(seq_answer_relative, seq_relative_rt, seq_relative_switch) if (sw == 0)])
        return (parity_rt_unswitch + relative_rt_unswitch) / (len(seq_parity_rt) + len(seq_relative_rt))
    # separate the relative and parity condition for rt
    elif answer_type == "parity_check_switch_hit":
        parity_hit_switch = sum(
            [1 for elt, sw in zip(seq_answer_parity, seq_parity_switch) if (sw == 1 and elt == 1)])
        return parity_hit_switch
    elif answer_type == "relative_check_switch_hit":
        relative_hit_switch = sum(
            [1 for elt, sw in zip(seq_answer_relative, seq_relative_switch) if (sw == 1 and elt == 1)])
        return relative_hit_switch
    elif answer_type == "parity_check_unswitch_hit":
        parity_hit_unswitch = sum(
            [1 for elt, sw in zip(seq_answer_parity, seq_parity_switch) if (sw == 0 and elt == 1)])
        return parity_hit_unswitch
    elif answer_type == "relative_check_unswitch_hit":
        relative_hit_unswitch = sum(
            [1 for elt, sw in zip(seq_answer_relative, seq_relative_switch) if (sw == 0 and elt == 1 and elt == 1)])
        return relative_hit_unswitch
    elif answer_type == "parity_check_switch_rt":
        parity_rt_switch = sum(
            [rt for elt, rt, sw in zip(seq_answer_parity, seq_parity_rt, seq_parity_switch) if (sw == 1)])
        return (parity_rt_switch) / len(seq_parity_rt)
    elif answer_type == "relative_check_switch_rt":
        relative_rt_switch = sum(
            [rt for elt, rt, sw in zip(seq_answer_relative, seq_relative_rt, seq_relative_switch) if (sw == 1)])
        return (relative_rt_switch) / len(seq_relative_rt)
    elif answer_type == "parity_check_unswitch_rt":
        parity_rt_unswitch = sum(
            [rt for elt, rt, sw in zip(seq_answer_parity, seq_parity_rt, seq_parity_switch) if (sw == 0)])
        return (parity_rt_unswitch) / len(seq_parity_rt)
    elif answer_type == "relative_check_unswitch_rt":
        relative_rt_unswitch = sum(
            [rt for elt, rt, sw in zip(seq_answer_relative, seq_relative_rt, seq_relative_switch) if (sw == 0)])
        return (relative_rt_unswitch) / len(seq_relative_rt)

def compute_mean(row):
    return np.mean(row["results_rt"])

def extract_id(dataframe, num_count):
    mask = pd.DataFrame(dataframe.participant_id.value_counts() == num_count)
    indices_id = mask[mask['participant_id'] == True].index.tolist()
    return indices_id


def extract_mu_ci_from_summary_accuracy(dataframe, ind_cond):
    outs = np.zeros((len(ind_cond), 3))  # 3 means the mu, ci_min, and ci_max
    for t, ind in enumerate(ind_cond):
        outs[t, 0] = dataframe[ind].mu_theta
        outs[t, 1] = dataframe[ind].ci_min
        outs[t, 2] = dataframe[ind].ci_max
    return outs


def extract_mu_ci_from_summary_rt(dataframe, ind_cond):
    outs = np.zeros((len(ind_cond), 3))  # 3 means the mu, ci_min, and ci_max
    for t, ind in enumerate(ind_cond):
        outs[t, 0] = dataframe[ind].mu_rt
        outs[t, 1] = dataframe[ind].ci_min
        outs[t, 2] = dataframe[ind].ci_max
    return outs


def get_overall_dataframe_taskswitch(dataframe, outcomes_names):
    # summarize two days experiments
    indices_id = extract_id(dataframe, num_count=2)
    sum_observers, tmp_nb = [], []
    for ob in indices_id:
        tmp_df = dataframe.groupby(["participant_id"]).get_group(ob)
        sum_observers.append([np.sum(tmp_df[index]) for index in outcomes_names])
    sum_observers = pd.DataFrame(sum_observers, columns=outcomes_names)
    return sum_observers

def treat_data(dataframe):
    # dataframe = delete_uncomplete_participants(dataframe)
    # Remove participants that didn't provide at least 1 answer:
    dataframe["results_responses"] = dataframe.apply(lambda row: transform_string_to_row(row, "results_responses"),
                                                     axis=1)
    dataframe["results_trial_target"] = dataframe.apply(
        lambda row: transform_string_to_row(row, "results_trial_target"), axis=1)
    dataframe["results_indtask"] = dataframe.apply(
        lambda row: transform_string_to_row(row, "results_indtask"), axis=1)
    dataframe["results_rt"] = dataframe.apply(
        lambda row: transform_string_to_row(row, "results_rt"), axis=1)
    dataframe.drop(dataframe[dataframe['results_rt'].str.len() == 0].index, inplace=True)
    # print(dataframe.info())
    # results_ind_switch : remove first element of each row by null
    # 3 blocks - 99 responses (idx: 0 - 33 - 66 , beggining of each block should be set to null)
    # participant = dataframe[dataframe['task_status'] == "PRE_TEST"]
    # participant = participant[participant['participant_id'] == 15]
    dataframe["results_ind_switch_clean"] = dataframe.apply(delete_beggining_of_block, axis=1)
    # results_response: actual answer of the participant
    # ind_switch: is it a "reconfiguration answer" 1=lower-even / 2=higher-odd
    # results_trial_target: is the question
    dataframe["nb_correct_total_answer"] = dataframe.apply(lambda row: compute_correct_answer(row, "correct_total"),
                                                           axis=1)
    dataframe["nb_correct_relative_answer"] = dataframe.apply(
        lambda row: compute_correct_answer(row, "correct_relative"), axis=1)
    dataframe["nb_correct_parity_answer"] = dataframe.apply(lambda row: compute_correct_answer(row, "correct_parity"),
                                                            axis=1)
    dataframe["nb_total"] = dataframe.apply(lambda row: compute_correct_answer(row, "total_nb"), axis=1)
    dataframe["nb_parity"] = dataframe.apply(lambda row: compute_correct_answer(row, "parity_nb"), axis=1)
    dataframe["nb_relative"] = dataframe.apply(lambda row: compute_correct_answer(row, "relative_nb"), axis=1)
    dataframe["errors_in_switch"] = dataframe.apply(lambda row: compute_correct_answer(row, "check_switch"), axis=1)
    dataframe["total_error"] = dataframe["nb_total"] - dataframe["nb_correct_total_answer"]
    dataframe["accuracy"] = dataframe["nb_correct_total_answer"] / dataframe["nb_total"]
    dataframe["mean_RT"] = dataframe.apply(compute_mean, axis=1)
    # Additional condition extration:
    dataframe["correct_in_switch"] = dataframe.apply(lambda row: compute_correct_answer(row, "check_switch_hit"),
                                                     axis=1)
    dataframe["correct_in_unswitch"] = dataframe.apply(lambda row: compute_correct_answer(row, "check_unswitch_hit"),
                                                       axis=1)
    dataframe["switch-rt"] = dataframe.apply(lambda row: compute_correct_answer(row, "check_switch_rt"), axis=1)
    dataframe["unswitch-rt"] = dataframe.apply(lambda row: compute_correct_answer(row, "check_unswitch_rt"), axis=1)
    # (relative or parity) AND (switch OR unswitch) = 4 cases
    # 2 outcomes taken into account : nb correct and rt
    dataframe["relative-switch-correct"] = dataframe.apply(
        lambda row: compute_correct_answer(row, "relative_check_switch_hit"), axis=1)
    dataframe["relative-unswitch-correct"] = dataframe.apply(
        lambda row: compute_correct_answer(row, "relative_check_unswitch_hit"), axis=1)
    dataframe["relative-switch-nb"] = dataframe.apply(
        lambda row: compute_correct_answer(row, "relative_check_switch_total"), axis=1)
    dataframe["relative-unswitch-nb"] = dataframe.apply(
        lambda row: compute_correct_answer(row, "relative_check_unswitch_total"), axis=1)
    dataframe["relative-switch-rt"] = dataframe.apply(
        lambda row: compute_correct_answer(row, "relative_check_switch_rt"), axis=1)
    dataframe["relative-unswitch-rt"] = dataframe.apply(
        lambda row: compute_correct_answer(row, "relative_check_unswitch_rt"), axis=1)
    dataframe["parity-switch-correct"] = dataframe.apply(
        lambda row: compute_correct_answer(row, "parity_check_switch_hit"), axis=1)
    dataframe["parity-unswitch-correct"] = dataframe.apply(
        lambda row: compute_correct_answer(row, "parity_check_unswitch_hit"), axis=1)
    dataframe["parity-switch-nb"] = dataframe.apply(
        lambda row: compute_correct_answer(row, "parity_check_switch_total"), axis=1)
    dataframe["parity-unswitch-nb"] = dataframe.apply(
        lambda row: compute_correct_answer(row, "parity_check_unswitch_total"), axis=1)
    dataframe["parity-switch-rt"] = dataframe.apply(
        lambda row: compute_correct_answer(row, "parity_check_switch_rt"), axis=1)
    dataframe["parity-unswitch-rt"] = dataframe.apply(
        lambda row: compute_correct_answer(row, "parity_check_unswitch_rt"), axis=1)
    dataframe["parity-switching-cost-rt"] = dataframe["parity-switch-rt"] - dataframe["parity-unswitch-rt"]
    dataframe["relative-switching-cost-rt"] = dataframe["relative-switch-rt"] - dataframe["relative-unswitch-rt"]
    # I added these 337-340 to avoid error, as non-used params -nbs are searched for the stan simulation in the util.py.
    # I hesitated to edit the util.py due to the dependency to other files.
    dataframe["parity-switching-cost-nb"] = dataframe.apply(
        lambda row: compute_correct_answer(row, "parity_check_unswitch_total"), axis=1)
    dataframe["relative-switching-cost-nb"] = dataframe.apply(
        lambda row: compute_correct_answer(row, "parity_check_unswitch_total"), axis=1)
    dataframe = add_conditions(dataframe)
    dataframe['switch_indicator'] = dataframe.apply(
        lambda row: list(map(lambda elt: int(elt), row['results_ind_switch'].split(","))), axis=1)
    return dataframe

def add_conditions(df):
    # # TASK-SWITCH # #
    # df = pd.read_csv(os.path.join(path, "taskswitch_lfa.csv"))
    # Condition to check ==> accuracy in parity task or in relative task:
    df['parity-correct'] = df['parity-switch-correct'] + df['parity-unswitch-correct']
    df['parity-nb'] = df['parity-switch-nb'] + df['parity-unswitch-nb']
    # df['parity-accuracy'] = df['parity-correct'] / df['parity-nb']
    df['relative-correct'] = df['relative-switch-correct'] + df['relative-unswitch-correct']
    df['relative-nb'] = df['relative-switch-nb'] + df['relative-unswitch-nb']
    # df['relative-accuracy'] = df['relative-correct'] / df['relative-nb']
    # Other condition to check ==> accuracy in switch VS unswitch condition
    df['switch-correct'] = df['parity-switch-correct'] + df['relative-switch-correct']
    df['switch-nb'] = df['parity-switch-nb'] + df['relative-switch-nb']
    # df['switch-accuracy'] = df['switch-correct'] / df['switch-nb']
    df['unswitch-correct'] = df['parity-unswitch-correct'] + df['relative-unswitch-correct']
    df['unswitch-nb'] = df['parity-unswitch-nb'] + df['relative-unswitch-nb']
    # df['unswitch-accuracy'] = df['unswitch-correct'] / df['unswitch-nb']
    # Switch and unswitch correct contains whether the participant answered relative/parity task:
    df['total-task-correct'] = convert_to_global_task(df, ['switch-correct', 'unswitch-correct'])
    df['total-task-nb'] = df['switch-nb'] + df['unswitch-nb']
    # df['total-task-accuracy'] = df['total-task-correct'] / df['total-task-nb']
    return df

def format_data(path):
    # DATAFRAME CREATION
    csv_path = f"{path}/taskswitch.csv"
    dataframe = pd.read_csv(csv_path, sep=",")
    dataframe = treat_data(dataframe)
    base = ['participant_id', 'task_status', 'condition', 'switch_indicator', 'results_rt']
    conditions = ['parity-switch', 'parity-unswitch', 'relative-switch', 'relative-unswitch', 'switch', 'unswitch',
                  'relative', 'parity', 'total-task']
    condition_names_correct = [f"{cdt}-correct" for cdt in conditions]
    column_nb_to_keep = ['nb_total', *[f"{cdt}-nb" for cdt in conditions]]
    condition_names_rt = ['parity-switching-cost-rt', 'relative-switching-cost-rt']
    dataframe = dataframe[base + condition_names_correct + condition_names_rt + column_nb_to_keep]
    for cdt in conditions:
        dataframe[f"{cdt}-accuracy"] = dataframe[f"{cdt}-correct"] / dataframe[f"{cdt}-nb"]
    nb_participants_init = len(dataframe['participant_id'].unique())
    # for condition in condition_names_rt:
    #     dataframe = detect_outliers_and_clean(dataframe, condition)
    # for condition in [f"{cdt}-accuracy" for cdt in conditions]:
    #     dataframe = detect_outliers_and_clean(dataframe, condition)
    dataframe = detect_outliers_and_clean(dataframe, 'total-task-accuracy')
    print(f"Taskswitch, proportion removed: {len(dataframe['participant_id'].unique())} / {nb_participants_init} ")
    dataframe.to_csv(f"{path}/taskswitch_lfa.csv", index=False)
    return dataframe

def preprocess_and_save(study):
    task = "taskswitch"
    savedir = f"data/{study}/cognitive_battery"
    path = f"{savedir}/{task}"
    Path(savedir).mkdir(parents=True, exist_ok=True)
    format_data(path)