import pandas as pd


def get_df(studies):
    df = pd.DataFrame()
    for study in studies:
        tmp = pd.read_csv(f"data/{study}/cognitive_battery/all_tasks_results.csv")
        tmp['study'] = study
        df = pd.concat([df, tmp], ignore_index=True)
    return df
