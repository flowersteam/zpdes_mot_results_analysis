from analysis_scripts.cognitive_battery.preprocessing.utils import *
from pathlib import Path
from analysis_scripts.cognitive_battery.preprocessing.utils import detect_outliers_and_clean

def is_one(result):
    """From a scalar accuracy (between 0 and 1), returns 1 if result is 1 and 0 otherwise"""
    if result == 1:
        return 1
    else:
        return 0

def compute_mean_per_condition(row):
    """
    3 conditions for MOT: speed=1,4 or 8
    Compute mean accuracy and mean RT for each condition
    """
    if "results_num_target" in row:
        return treat_with_2_conditions(row)
    else:
        return treat_with_1_condition(row)


def treat_with_2_conditions(row):
    dict_mean_accuracy_per_condition = {}
    dict_mean_rt_per_condition = {}
    for idx, (condition_key_speed, condition_key_nb_target) in enumerate(
            zip(row['results_speed_stim'], row['results_num_target'])):
        cdt_name = f"{condition_key_speed}-speed-{condition_key_nb_target}-nb-target"
        if cdt_name not in dict_mean_accuracy_per_condition:
            dict_mean_accuracy_per_condition[cdt_name] = []
            dict_mean_rt_per_condition[cdt_name] = []
        dict_mean_accuracy_per_condition[cdt_name].append(float(row['results_correct'][idx]))
        dict_mean_rt_per_condition[cdt_name].append(float(row['results_rt'][idx]))
    for idx, condition_key in enumerate(row['results_speed_stim']):
        if f"{condition_key}-speed" not in dict_mean_accuracy_per_condition:
            dict_mean_accuracy_per_condition[f"{condition_key}-speed"] = []
            dict_mean_rt_per_condition[f"{condition_key}-speed"] = []
        dict_mean_accuracy_per_condition[f"{condition_key}-speed"].append(float(row['results_correct'][idx]))
        dict_mean_rt_per_condition[f"{condition_key}-speed"].append(float(row['results_rt'][idx]))
    if 'results_num_target' in row:
        for idx, condition_key in enumerate(row['results_num_target']):
            if f"{condition_key}-nb-targets" not in dict_mean_accuracy_per_condition:
                dict_mean_accuracy_per_condition[f"{condition_key}-nb-targets"] = []
                dict_mean_rt_per_condition[f"{condition_key}-nb-targets"] = []
            dict_mean_accuracy_per_condition[f"{condition_key}-nb-targets"].append(float(row['results_correct'][idx]))
            dict_mean_rt_per_condition[f"{condition_key}-nb-targets"].append(float(row['results_rt'][idx]))
    for key in dict_mean_accuracy_per_condition.keys():
        # Before getting the mean accuracy, we need to parse each trial to a binary success vector (i.e 0=failure, 1=success)
        row[f"{key}-rt"] = np.mean(dict_mean_rt_per_condition[key])
        row[f"{key}-accuracy"] = np.mean(list(map(lambda x: is_one(x), dict_mean_accuracy_per_condition[key])))
        row[f"{key}-accuracy_continuous"] = np.mean(list(map(lambda x: x, dict_mean_accuracy_per_condition[key])))
        row[f"{key}-correct"] = np.sum(list(map(lambda x: is_one(x), dict_mean_accuracy_per_condition[key])))
        row[f"{key}-nb"] = len(dict_mean_accuracy_per_condition[key])
    return row


def treat_with_1_condition(row):
    dict_mean_accuracy_per_condition = {}
    dict_mean_rt_per_condition = {}
    for idx, condition_key in enumerate(row['results_speed_stim']):
        if condition_key not in dict_mean_accuracy_per_condition:
            dict_mean_accuracy_per_condition[condition_key] = []
            dict_mean_rt_per_condition[condition_key] = []
        dict_mean_accuracy_per_condition[condition_key].append(float(row['results_correct'][idx]))
        dict_mean_rt_per_condition[condition_key].append(float(row['results_rt'][idx]))
    for key in dict_mean_accuracy_per_condition.keys():
        row[f"{key}-speed-rt"] = np.mean(dict_mean_rt_per_condition[key])
        row[f"{key}-speed-accuracy"] = np.mean(dict_mean_accuracy_per_condition[key])
        row[f"{key}-speed-correct"] = np.sum(list(map(lambda x: is_one(x), dict_mean_accuracy_per_condition[key])))
        row[f"{key}-speed-nb"] = len(dict_mean_accuracy_per_condition[key])
        row[f"{key}-speed-accuracy_continuous"] = np.mean(list(map(lambda x: x, dict_mean_accuracy_per_condition[key])))
    return row


def count_number_of_trials(row):
    return len(row['results_correct'])


def compute_result_sum_hr(row):
    return 18 - row['result_nb_omission']


def format_data(path, save_lfa=False):
    # FIRST TREAT THE CSV AND PARSE IT TO DF
    csv_path = f"{path}/moteval.csv"
    df = pd.read_csv(csv_path, sep=",")
    df = df.apply(lambda row: transform_str_to_list(row, [
        'results_responses', 'results_rt', 'results_speed_stim', 'results_correct', 'results_num_target']), axis=1)
    df = df.apply(compute_mean_per_condition, axis=1)
    df.to_csv(f'{path}/moteval_treat.csv')
    nb_trials = len(df['results_correct'][0])
    # Declare all conditions:
    conditions_speed = [1, 4, 8]
    outcomes_names_acc = [f"{cdt}-speed-accuracy" for cdt in conditions_speed]
    outcomes_names_acc_continuous = [f"{cdt}-speed-accuracy_continuous" for cdt in conditions_speed]
    outcomes_names_rt = [f"{cdt}-speed-rt" for cdt in conditions_speed]
    outcomes_names_correct = [f"{cdt}-speed-correct" for cdt in conditions_speed]
    outcomes_names_nb = [f"{cdt}-speed-nb" for cdt in conditions_speed]
    conditions_nb_targets = [3, 5]
    if "results_num_target" in df.columns:
        outcomes_names_acc = outcomes_names_acc + [f"{cdt}-nb-targets-accuracy" for cdt in conditions_nb_targets]
        outcomes_names_acc_continuous = outcomes_names_acc_continuous + [f"{cdt}-nb-targets-accuracy_continuous" for cdt
                                                                         in conditions_nb_targets]
        outcomes_names_rt = outcomes_names_rt + [f"{cdt}-nb-targets-rt" for cdt in conditions_nb_targets]
        outcomes_names_correct = outcomes_names_correct + [f"{cdt}-nb-targets-correct" for cdt in conditions_nb_targets]
        outcomes_names_nb = outcomes_names_nb + [f"{cdt}-nb-targets-nb" for cdt in conditions_nb_targets]
        outcomes_names_acc = outcomes_names_acc + [f"{cdt_s}-speed-{cdt_t}-nb-target-accuracy" for cdt_t in
                                                   conditions_nb_targets for cdt_s in conditions_speed]
        outcomes_names_rt = outcomes_names_rt + [f"{cdt_s}-speed-{cdt_t}-nb-target-rt" for cdt_t in
                                                 conditions_nb_targets for cdt_s in conditions_speed]
        outcomes_names_correct = outcomes_names_correct + [f"{cdt_s}-speed-{cdt_t}-nb-target-correct" for cdt_t in
                                                           conditions_nb_targets for cdt_s in conditions_speed]
        outcomes_names_nb = outcomes_names_nb + [f"{cdt_s}-speed-{cdt_t}-nb-target-nb" for cdt_t in
                                                 conditions_nb_targets for cdt_s in conditions_speed]
    base = ['participant_id', 'task_status', 'condition']
    df = df[
        base + outcomes_names_acc + outcomes_names_rt + outcomes_names_correct + outcomes_names_nb + outcomes_names_acc_continuous]
    # Only taking nb-targets (otherwise overlapping):
    df['total-task-correct'] = convert_to_global_task(df, [col for col in
                                                           [f"{cdt}-speed-correct" for cdt in conditions_speed]])
    df['total-task-accuracy'] = df['total-task-correct'] / nb_trials
    df['total-task-nb'] = nb_trials
    outcomes_names_acc.append('total-task-accuracy')
    nb_participants_init = len(df['participant_id'].unique())
    # for condition in outcomes_names_acc:
    #     df = detect_outliers_and_clean(df, condition)
    df = detect_outliers_and_clean(df, 'total-task-accuracy')
    print(f"Moteval, proportion removed: {len(df['participant_id'].unique())} / {nb_participants_init} ")
    df.to_csv(f'{path}/moteval_lfa.csv', index=False)
    # For other formats (analysis in JASP):
    cdts_JASP = [f"{cdt_s}-speed-{cdt_t}-nb-target-accuracy" for cdt_t in
                 conditions_nb_targets for cdt_s in conditions_speed]
    save_for_jasp(df, cdts_JASP, path, "moteval")
    cdts_targets = [f"{cdt_t}-nb-targets-accuracy" for cdt_t in conditions_nb_targets]
    cdts_targets_continuous = [f"{cdt_t}-nb-targets-accuracy_continuous" for cdt_t in conditions_nb_targets]
    save_for_jasp(df, cdts_targets + cdts_targets_continuous, path, "moteval_targets")
    cdts_speed = [f"{cdt}-speed-accuracy" for cdt in conditions_speed]
    save_for_jasp(df, cdts_speed, path, "moteval_speed")
    return df


def save_for_jasp(df, cdts, path, study_name):
    new_df = pd.DataFrame(columns=["participant_id", "condition", "difficulty", "DV_pre", "DV_post"])
    p_ids = df["participant_id"].unique()
    for participant in p_ids:
        p_row_pre = df.query(f'participant_id == {participant} & task_status=="PRE_TEST"')
        p_row_post = df.query(f'participant_id == {participant} & task_status=="POST_TEST"')
        condition = p_row_pre["condition"].values[0]
        for cdt in cdts:
            new_df.loc[len(new_df)] = {"participant_id": participant,
                                       "condition": condition,
                                       "difficulty": cdt, "DV_pre": p_row_pre[cdt].values[0],
                                       "DV_post": p_row_post[cdt].values[0]}
    # # Concatenate the intra_training MOT evaluation:
    # binary_intra = pd.read_csv(foutputs/v3_utl/results_v3_utl/v3_utl_binary_intra.csv')
    # F1_intra = pd.read_csv(foutputs/v3_utl/results_v3_utl/v3_utl_F1_intra.csv')
    # new_df = new_df.merge(F1_intra, on=["participant_id", "condition"])
    # new_df = new_df.rename(columns={f"s_{i}": f"F1_s_{i}" for i in range(4)})
    # new_df = new_df.merge(binary_intra, on=["participant_id", "condition"])
    # new_df = new_df.rename(columns={f"s_{i}": f"Bi_s_{i}" for i in range(4)})
    # new_df.to_csv(f'{path}/{study_name}_jasp.csv', index=False)

def preprocess_and_save(study):
    task = "moteval"
    savedir = f"data/{study}/cognitive_battery"
    path = f"{savedir}/{task}"
    Path(savedir).mkdir(parents=True, exist_ok=True)
    format_data(path)
