from sklearn.preprocessing import StandardScaler
import pandas as pd
import copy
from scipy.stats import norm
from scipy.stats import rankdata
import numpy as np


def zeros_ones_transform(row, n=10):
    if row == 0:
        return 1 / (2 * n)
    elif row == 1:
        return 1 - (1 / (2 * n))
    return row


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer


def normalize_data(data, columns_for_pca, columns_nb_trials, mode="zscore", shuffle=True):
    def safe_log1p(x):
        return np.log1p(np.where(x <= 0, 1e-9, x))

    rt_pipeline = Pipeline(steps=[
        ('log_transform', FunctionTransformer(func=safe_log1p, inverse_func=np.expm1, validate=True)),
        ('scaler', StandardScaler())
    ])
    if mode == "zscore":
        # Create a StandardScaler instance
        scaler = StandardScaler()
        # Fit the scaler on the selected columns and transform the data
        df_normalized = scaler.fit_transform(data[columns_for_pca])
        # Create a new DataFrame with the scaled data
        df_normalized = pd.DataFrame(df_normalized, columns=columns_for_pca)
    elif mode == "probit":
        # First transform 0-1 values
        # Then probit transform
        df_normalized = copy.deepcopy(data)
        for index_col, col in enumerate(columns_for_pca):
            if "accuracy" in col:
                df_normalized[col] = [zeros_ones_transform(data_val, nb_val) for nb_val, (counter, data_val) in
                                      zip(df_normalized[columns_nb_trials[index_col]], enumerate(df_normalized[col]))]
                try:
                    df_normalized[col] = df_normalized[col].apply(norm.ppf)
                except:
                    print("af")
            elif "rt" in col:
                try:
                    df_normalized[col] = rt_pipeline.fit_transform(df_normalized[col].values.reshape(-1, 1)).flatten()
                except Exception as e:
                    print("ah")
            elif "d'" in col:
                scaler = StandardScaler()
                df_normalized[col] = scaler.fit_transform(df_normalized[col].values.reshape(-1, 1)).flatten()
        df_normalized = df_normalized[columns_for_pca]
    elif mode == "percentile_rank":
        df_normalized = pd.DataFrame(columns=columns_for_pca)
        for col in columns_for_pca:
            df_normalized[col] = rankdata(data[col], method='min') / len(data[col]) * 100
            df_normalized[columns_for_pca] = (df_normalized[columns_for_pca] - df_normalized[columns_for_pca].mean()) / \
                                             df_normalized[columns_for_pca].std()
    else:
        df_normalized = None
        print("Non existing normalization")
    if shuffle:
        df_normalized = df_normalized.sample(frac=1, random_state=0)
    return df_normalized
