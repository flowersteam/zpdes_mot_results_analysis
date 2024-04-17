from pathlib import Path
import pandas as pd
from analysis_scripts.questionnaires.utils import format_questionnaire


def retrieve_questionnaire(study, questionnaire_name):
    df = pd.read_csv(f'data/{study}/questionnaires/all_answers.csv')
    df.drop(columns=['Unnamed: 0'], inplace=True)
    df.columns = ['id', 'condition', 'component', 'instrument', 'handle', 'session_id', 'value']
    return df[df['instrument'] == questionnaire_name]


def clean_dataset(df):
    """
    Delete participants with missing sessions
    :param df:
    :return:
    """
    # Get the number of session, we take the median nb of sessions (avoid min and max if too much responses)
    nb_sessions = df.groupby('id_participant')['session_id'].count().median()
    correct_sessions_length = df.groupby('id_participant').filter(lambda x: len(x) == nb_sessions).iloc[0][
        "id_participant"]
    example_participant = df[df['id_participant'] == correct_sessions_length]['session_id']
    for p in df['id_participant'].unique():
        if len(df[df['id_participant'] == p]) > nb_sessions:
            print(f"Problem with particpant {p} - to many values")
        elif len(df[df['id_participant'] == p]) < nb_sessions:
            # Then retrieve which session was uncorrect
            participant_pb = df[df['id_participant'] == p]
            for id_ex in example_participant:
                if id_ex not in participant_pb['session_id'].values:
                    # Add row with the mean
                    row = pd.DataFrame([participant_pb.select_dtypes(include=['number']).drop(
                        columns=['id_participant', 'session_id']).mean()])
                    row['id_participant'] = p
                    row['session_id'] = id_ex
                    row['condition'] = participant_pb['condition'].iloc[0]
                    df = pd.concat([df, row], ignore_index=True)
    return df


def treat_nasa_tlx(study):
    # First retrieve questionnaire:
    df = retrieve_questionnaire(study, "mot-NASA-TLX")
    path_to_store = f'data/{study}/questionnaires/nasa_tlx'
    Path(path_to_store).mkdir(parents=True, exist_ok=True)
    df.to_csv(f'{path_to_store}/nasa_tlx_raw.csv')

    # Format questionnaire (general format for all questionnaires):
    conditions = ['Mental_demand', 'Physical_demand', 'Temporal_demand', 'Performance', 'Effort', 'Frustration',
                  'load_index']
    df = format_questionnaire(df)

    # Additionnal manipulation specifically for nasa:
    # Drop of the 9th session
    df = df[df.loc[:, 'session_id'] != 9]
    df = df.groupby('id_participant').filter(lambda x: len(x) == 8)
    df = df.rename(columns={'Mental Demand': 'Mental_demand', 'Physical demand': 'Physical_demand',
                            'Temporal demand': 'Temporal_demand'})
    df['load_index'] = df[conditions[:-1]].sum(axis=1)
    df = clean_dataset(df)
    df.to_csv(f'{path_to_store}/nasa_tlx.csv')


def treat_UES(study):
    """
    :param study:
    :return:
    """
    # First retrieve questionnaire:
    df = retrieve_questionnaire(study, "mot-UES")
    path_to_store = f'data/{study}/questionnaires/ues'
    Path(path_to_store).mkdir(parents=True, exist_ok=True)
    df.to_csv(f'{path_to_store}/ues_raw.csv')

    # General format questionnaire:
    df = format_questionnaire(df)

    # Additionnal questionnaire format:
    conditions = ['FA-S.1', 'FA-S.2', 'FA-S.3', 'PU-S.1', 'PU-S.2', 'PU-S.3', 'AE-S.1', 'AE-S.2', 'AE-S.3', 'RW-S.1',
                  'RW-S.2', 'RW-S.3']
    conditions_to_keep = ['FA', 'PU', 'AE', 'RW', 'engagement_score']
    reverse_condition = ['PU-S.1', 'PU-S.2', 'PU-S.3']
    df[reverse_condition] = 5 - df[reverse_condition]
    df['engagement_score'] = df[conditions].mean(axis=1)
    df['FA'] = df[['FA-S.1', 'FA-S.2', 'FA-S.3']].mean(axis=1)
    df['PU'] = df[['PU-S.1', 'PU-S.2', 'PU-S.3']].mean(axis=1)
    df['AE'] = df[['AE-S.1', 'AE-S.2', 'AE-S.3']].mean(axis=1)
    df['RW'] = df[['RW-S.1', 'RW-S.2', 'RW-S.3']].mean(axis=1)

    # Delete participants with missing sessions:
    df = clean_dataset(df)
    df = df.drop(columns=conditions)
    df.to_csv(f'{path_to_store}/ues.csv')


def treat_SIMS(study):
    df = retrieve_questionnaire(study, "mot-SIMS")
    path_to_store = f'data/{study}/questionnaires/sims'
    Path(path_to_store).mkdir(parents=True, exist_ok=True)
    df.to_csv(f'{path_to_store}/sims_raw.csv')

    # Format questionnaire:
    df = format_questionnaire(df)
    df = df.rename(
        columns={'Intrinsic motivation': 'Intrinsic_motivation', 'Identified regulation': 'Identified_regulation',
                 'External regulation': 'External_regulation'})
    # Create SDI
    df['SDI'] = (2 * df['Intrinsic_motivation'] + df['Identified_regulation']) - (
            2 * df['Amotivation'] + df['External_regulation'])
    # keep all sessions and groups:
    df = df.groupby('id_participant').filter(lambda x: len(x) == 4)
    df = clean_dataset(df)
    df.to_csv(f'{path_to_store}/sims.csv')


def treat_TENS(study):
    df = retrieve_questionnaire(study, "mot-TENS")
    path_to_store = f'data/{study}/questionnaires/tens'
    Path(path_to_store).mkdir(parents=True, exist_ok=True)
    df.to_csv(f'{path_to_store}/tens_raw.csv')

    # Format questionnaire:
    df = format_questionnaire(df)

    # keep all sessions and groups:
    df = df.groupby('id_participant').filter(lambda x: len(x) == 4)
    df = clean_dataset(df)
    df.to_csv(f'{path_to_store}/tens.csv')


def treat_LP(study):
    """
    # 0: Lors de la prochaine activité d'entraînement, quel nombre de cibles souhaiteriez-vous avoir ?
    # 1: Quel est le nombre de cibles que vous pouvez suivre sans difficulté ?
    # 2: Quel est le nombre de cibles que vous pouvez suivre avec difficulté ?
    # Format questionnaire:
    :param study:
    :return:
    """
    df = retrieve_questionnaire(study, "mot-LP")
    path_to_store = f'data/{study}/questionnaires/lp'
    Path(path_to_store).mkdir(parents=True, exist_ok=True)
    df.to_csv(f'{path_to_store}/lp_raw.csv')

    # format_questionnaire() split according to component column
    # In LP questionnaire we need to split according to handle
    df['component'] = df['handle']
    df = format_questionnaire(df)
    df = clean_dataset(df)
    conditions = ['Difficulty_expectation', 'Easy_Feasible_Zone', 'Hard_Feasible_Zone']
    df = df.rename(columns={f'mot-LP-{i}': conditions[i] for i in range(3)})
    df.to_csv(f'{path_to_store}/lp.csv')
