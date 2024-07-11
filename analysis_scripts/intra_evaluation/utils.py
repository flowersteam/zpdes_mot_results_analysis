import pandas as pd
import numpy as np

def transform_to_long(data, group_name):
    n_participants = len(data)
    n_measurements = len(data[0])
    df = pd.DataFrame({
        'group': group_name,
        'participant_id': np.repeat(np.arange(n_participants), n_measurements),
        'time': np.tile(np.arange(n_measurements), n_participants),
        'measurement': np.array(data).flatten()
    })
    return df

def parse_trajectory(row):
    values = row['trajectory'][1:-1].split()
    values = np.array(list(map(lambda x: float(x), values)))
    return values
