from .extract_sorted_memory import Results_memory
from analysis_scripts.cognitive_battery.preprocessing.utils import *
from pathlib import Path


# Treat data:
def compute_result_exact_answers(row):
    response = row["results_responses"].split(',')
    target = row["results_targetvalue"].split(',')

    return sum(x == y for x, y in zip(response, target))


def delete_uncomplete_participants(dataframe):
    """

    """
    mask = pd.DataFrame(dataframe.participant_id.value_counts() < 2)
    participants_to_delete = mask[mask['participant_id'] == True].index.tolist()
    for id in participants_to_delete:
        dataframe = dataframe[dataframe['participant_id'] != id]
    return dataframe


def extract_id(dataframe, num_count):
    mask = pd.DataFrame(dataframe.participant_id.value_counts() == num_count)
    indices_id = mask[mask['count'] == True].index.tolist()
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


def treat_data(dataframe, dataframe_2, conditions_names):
    indices_id = extract_id(dataframe, num_count=4)
    test_status = ["PRE_TEST", "POST_TEST"]
    sum_observers = []
    for ob in indices_id:
        tmp_df = dataframe.groupby(["participant_id"]).get_group(ob)
        tmp_results = Results_memory(tmp_df)
        sum_observers.append(np.concatenate(([ob], tmp_results.out_mat_hit_miss_sum, tmp_results.out_mat_fa_cr_sum,
                                             tmp_results.out_mat_rt_cond, tmp_results.out_mat_rt_cond_std)))
        # Task status index varies from 0 to 3 => pre/post for memo 1 and 2
        # We use Results_memory for one participant_id for pre and post (and we do it for both memo 1 and 2)
        # We don't need to update the second part of memorability dataframe BUT this part is needed in Results_memory()
        # initialisation as it's the long range condition
        for task_status_index, (row_index, row) in enumerate(dataframe[dataframe['participant_id'] == ob].iterrows()):
            tmp_row = Results_memory(tmp_df[tmp_df['task_status'] == test_status[task_status_index % 2]])
            for conditions_name in conditions_names:
                for condition_index, condition in enumerate(conditions_name):
                    if 'hit' in condition:
                        tmp_cond = 'out_mat_hit_miss_sum'
                    elif 'fa' in condition:
                        tmp_cond = 'out_mat_fa_cr_sum'
                    else:
                        tmp_cond = 'out_mat_rt_cond'
                    dataframe.loc[row_index, condition] = tmp_row.__dict__[tmp_cond][condition_index]
    # This is only to delete the useless part
    dataframe = pd.merge(dataframe, dataframe_2, how='outer', indicator=True)
    dataframe = dataframe[dataframe['_merge'] == 'left_only']
    return dataframe, sum_observers


def format_data(path):
    # Get memorability 1
    csv_path_short_range = f"{path}/memorability_1.csv"
    dataframe_short_range = pd.read_csv(csv_path_short_range)
    # dataframe_short_range = delete_uncomplete_participants(dataframe_short_range)
    dataframe_short_range['session'] = 1
    # Get memorability 2
    csv_path_long_range = f"{path}/memorability_2.csv"
    dataframe_long_range = pd.read_csv(csv_path_long_range)
    # dataframe_long_range = delete_uncomplete_participants(dataframe_long_range)
    dataframe_long_range['session'] = 2
    # Concatenate
    dataframe = pd.concat([dataframe_short_range, dataframe_long_range], axis=0)
    # For memorability task, conditions is not used because of mswym code
    # Let's re-create the proper conditions:
    tmp_conditions = [*[f"{elt}" for elt in range(2, 6)], "100"]
    conditions_names_hit_miss = [f"{elt}-hit-miss" for elt in tmp_conditions]
    conditions_names_fa_cr = [f"{elt}-fa-cr" for elt in tmp_conditions]
    conditions_names_rt = [f"{cdt}-rt" for cdt in tmp_conditions]
    tmp_conditions_names = [conditions_names_hit_miss, conditions_names_fa_cr, conditions_names_rt]
    # Treat data to get dataframe
    dataframe, sum_observers = treat_data(dataframe, dataframe_long_range, tmp_conditions_names)
    # Rename columns
    for col in dataframe.columns:
        if 'hit-miss' in col:
            dataframe = dataframe.rename(columns={col: col.replace('hit-miss', 'hit-correct')})
        if 'fa' in col:
            dataframe = dataframe.rename(columns={col: col.replace('fa-cr', 'fa-correct')})
    real_conditions = [f"{cdt}-hit" for cdt in tmp_conditions] + [f"{cdt}-fa" for cdt in tmp_conditions]
    for condition in real_conditions:
        dataframe[f'{condition}-nb'] = 16
        dataframe[f'{condition}-accuracy'] = dataframe[f'{condition}-correct'] / dataframe[f'{condition}-nb']
    # Finaly keep final columns with proper conditions:
    base = ['participant_id', 'task_status', 'condition']
    all_conditions = [f"{cdt}-accuracy" for cdt in real_conditions]
    all_conditions += [f"{cdt}-correct" for cdt in real_conditions]
    all_conditions += [f"{cdt}-nb" for cdt in real_conditions]
    all_conditions += [f"{cdt}-rt" for cdt in tmp_conditions]
    dataframe = dataframe[base + all_conditions]
    dataframe['total-task-hit-correct'] = convert_to_global_task(dataframe, [f'{cdt}-hit-correct' for cdt in tmp_conditions])
    dataframe['total-task-fa-correct'] = convert_to_global_task(dataframe, [f'{cdt}-fa-correct' for cdt in tmp_conditions])
    dataframe['total-task-nb'] = 20 * len(tmp_conditions)
    dataframe['total-task-hit-nb'] = 20 * len(tmp_conditions)
    dataframe['total-task-hit-accuracy'] = dataframe['total-task-hit-correct'] / dataframe['total-task-nb']
    dataframe['total-task-fa-accuracy'] = dataframe['total-task-fa-correct'] / dataframe['total-task-nb']
    dataframe['total-task-hit-rt'] = dataframe[[col for col in dataframe.columns if '-rt' in col]].mean(axis=1)
    dataframe.to_csv(f'{path}/memorability_lfa.csv', index=False)


def preprocess_and_save(study):
    task = "memorability"
    savedir = f"data/{study}/cognitive_battery"
    path = f"{savedir}/{task}"
    Path(savedir).mkdir(parents=True, exist_ok=True)
    format_data(path)