from pathlib import Path
from .utils import *
import pandas as pd
import numpy as np


def transform_to_list(row):
    # We divide by 99 to have max staircase = 1
    return 1 - np.mean(list(map(lambda x: int(x), row['difficulty_proposed'].split(',')))[-5:]) / 99


def get_threshold_last(row):
    frame_duration = 17.5
    return np.mean(
        list(map(lambda x: float(x) * frame_duration, row['measured_difficulties_frame_count'].split(',')))[-5:])


def get_final_step(row):
    return np.mean(list(map(lambda x: int(x), row['difficulty_proposed'].split(',')))[-5:])


def get_nb_trials(row):
    return len(row['difficulty_proposed'].split(','))


def format_data(path):
    # FIRST TREAT THE CSV AND PARSE IT TO DF
    task = "ufov"
    # csv_path = f"../outputs/{study}/results_{study}/{task}/{task}.csv"
    try:
        df = pd.read_csv(f"{path}/{task}.csv", sep=",")
    except FileNotFoundError:
        return pd.DataFrame({})
    base = ['participant_id', 'task_status', 'condition']
    df["final-accuracy_continuous"] = df.apply(lambda row: transform_to_list(row), axis=1)
    df["final_step"] = df.apply(lambda row: get_final_step(row), axis=1)
    df["nb_trials"] = df.apply(lambda row: get_nb_trials(row), axis=1)
    df["final-threshold"] = df.apply(lambda row: get_threshold_last(row), axis=1)
    df["final-rt"] = df.apply(lambda row: get_threshold_last(row), axis=1)
    condition_accuracy = ["final-accuracy_continuous", "final_step", "nb_trials", "final-threshold", "final-rt"]
    df = df[base + condition_accuracy]
    df = delete_uncomplete_participants(df)
    df.to_csv(f"{path}/{task}_lfa.csv", index=False)


def preprocess_and_save(study):
    task = "ufov"
    savedir = f"../data/{study}/cognitive_battery"
    path = f"{savedir}/{task}"
    Path(savedir).mkdir(parents=True, exist_ok=True)
    format_data(path)
