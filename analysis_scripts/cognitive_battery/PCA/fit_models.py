import pandas as pd
import copy
from pathlib import Path
from .normalize import normalize_data
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA, FastICA
import numpy as np
import pickle

from analysis_scripts.cognitive_battery.PCA.utils import get_df
from analysis_scripts.cognitive_battery.PCA.visualize import (
    get_all_PCA_figures,
    plot_diff_on_PC,
)


def format_conditions(conditions):
    kept_columns = ["participant_id", "condition", "task_status", "study"]
    tasks = []
    for key, tasks_dict in conditions.items():
        condition_name = "".join(key.split("_")[1:])
        for tasks_conditions in tasks_dict.values():
            tasks += [[f"{t}-{condition_name}" for t in tasks_conditions]]
    # Create all tasks (accuracy + RT type):
    # tasks = [
    #     [f"{cdt}-accuracy" for cdt in task_condition]
    #     for task_condition in conditions["conditions_acc"].values()
    # ]
    # tasks += [
    #     [f"{cdt}-rt" for cdt in task_condition]
    #     for task_condition in conditions["conditions_rt"].values()
    # ]
    # Flatten tasks into 1-d
    tasks = [item for sublist in tasks for item in sublist]
    # Create nb especially for accuracy metrics normalization:
    tasks_nb = [
        [f"{cdt}-nb" for cdt in task_condition]
        for task_condition in conditions["conditions_accuracy"].values()
    ]
    # tasks_nb += [[f"{cdt}-nb" for cdt in task_condition] for task_condition in conditions_rt.values()]
    # Flatten tasks into 1-d
    tasks_nb = [item for sublist in tasks_nb for item in sublist]
    return tasks, tasks_nb, kept_columns


def run_dimensionality_reduction(
    data,
    columns_for_pca,
    columns_nb_trials,
    n_components=5,
    DR_type="PCA",
    normalization_mode="probit",
):
    filled_data = fill_na_with_mean(copy.deepcopy(data), columns_for_pca)
    df_normalized = normalize_data(
        filled_data, columns_for_pca, columns_nb_trials, mode=normalization_mode
    )
    return get_DR(DR_type, df_normalized, columns_for_pca, n_components)


def model_choice(DR_type, df, n_components):
    if DR_type == "PCA":
        model = PCA(n_components=n_components)
    elif DR_type == "ICA":
        model = FastICA(
            n_components=n_components, random_state=0, whiten="unit-variance"
        )
    else:
        model = None
        print("Model doesn't exit")
    return model


def validate_model(dataframe, test, model):
    # model is trained one using pca.fit or ica.fit
    data_extract = copy.deepcopy(dataframe.values[test, :])
    codes = model.transform(data_extract)
    # test prediction error:
    data_est = model.inverse_transform(codes)
    rmse = np.sqrt(np.mean((data_extract.flatten() - data_est.flatten()) ** 2))
    correlation = np.corrcoef(data_extract.flatten(), data_est.flatten())
    correlation = correlation[0, 1]
    return rmse, correlation


def get_DR(DR_type, df_normalized, columns_for_pca, n_components=5, n_splits=10):
    # We are going to run K-fold cross validation:
    kf = KFold(n_splits=n_splits)
    models, rmses, explained_vars, correlations, components = [], [], [], [], []
    for train, test in kf.split(df_normalized):
        sub_train_df = copy.deepcopy(df_normalized.values[train, :])
        # Fit the PCA model to the data
        DR_algo = model_choice(DR_type, df_normalized, n_components)
        DR_algo.fit(sub_train_df)
        rmse, correlation = validate_model(df_normalized, test, DR_algo)
        models.append(DR_algo)
        rmses.append(rmse)
        correlations.append(correlation)
        components.append(DR_algo.components_)
        if DR_type == "PCA":
            explained_vars.append(
                (np.cumsum(DR_algo.explained_variance_ratio_)[n_components - 1])
            )
    # Average of all models - not recommended but just to make sure we fit with previous work:
    mean_components = np.mean(components, 0)
    average_loadings = pd.DataFrame(
        np.reshape(mean_components, [n_components, df_normalized.shape[1]]),
        columns=df_normalized.columns,
    ).T
    # Transform the data to the principal components
    main_model = model_choice(DR_type, df_normalized, n_components)
    main_model.fit(df_normalized)
    main_model_loadings = pd.DataFrame(
        np.reshape(main_model.components_, [n_components, df_normalized.shape[1]]),
        columns=df_normalized.columns,
    ).T
    latent_df = main_model.transform(df_normalized)
    # Create a new DataFrame with the principal components
    latent_df = pd.DataFrame(
        data=latent_df, columns=[f"PC{i + 1}" for i in range(n_components)]
    )
    # Assuming you have df_normalized, the DataFrame with the centered and scaled data,
    # and a list of variable names, calculate the variable contributions (correlations with principal components)
    main_variable_contributions = np.corrcoef(df_normalized.T, latent_df.T)[
        : len(columns_for_pca), -n_components:
    ]
    return (
        latent_df,
        average_loadings,
        main_model_loadings,
        main_variable_contributions,
        main_model,
        rmses,
        correlations,
    )


def fill_na_with_mean(data, columns_for_pca):
    for study in data["study"].unique():
        study_mean = data.loc[data["study"] == study, columns_for_pca].mean()
        data.loc[data["study"] == study, columns_for_pca] = data.loc[
            data["study"] == study, columns_for_pca
        ].fillna(study_mean)
    return data


def save_results(
    df_pca,
    n_components,
    DR_type,
    columns_for_pca,
    variable_contributions,
    model,
    path,
    average_loadings,
    main_model_loadings,
    rmses,
    correlations,
):
    # Save loadings + errors
    Path(f"{path}/csv").mkdir(parents=True, exist_ok=True)
    average_loadings.to_csv(f"{path}/csv/average_loadings.csv")
    main_model_loadings.to_csv(f"{path}/csv/kept_main_loadings.csv")
    metrics = pd.DataFrame.from_dict(
        {
            "mean_mean": [np.mean(rmses)],
            "std_rmse": [np.std(rmses)],
            "standard_error_rmse": [np.std(rmses) / np.sqrt(len(rmses))],
            "mean_correl": [np.mean(correlations)],
            "error_correl": [np.std(correlations) / np.sqrt(len(correlations))],
        }
    )
    metrics.to_csv(f"{path}/csv/cross_validation_metrics.csv")
    Path(f"{path}/pickle").mkdir(parents=True, exist_ok=True)
    # First save the model PCA object to re-use it later:
    with open(f"{path}/pickle/model.pkl", "wb") as file:
        pickle.dump(model, file)
    Path(f"{path}/figures").mkdir(parents=True, exist_ok=True)
    get_all_PCA_figures(
        df_pca,
        path,
        DR_type,
        model,
        n_components,
        variable_contributions,
        columns_for_pca,
        average_loadings,
        main_model_loadings,
    )


def get_diff_from_PCA(
    df_analysis, main_model, tasks, tasks_nb, study, path, kept_columns
):
    # Use the found transformation
    df_analysis[tasks] = df_analysis[tasks].fillna(df_analysis[tasks].mean())
    cols = [col for col in df_analysis.columns if col not in kept_columns]
    df_analysis[tasks] = normalize_data(
        df_analysis[cols], tasks, tasks_nb, mode="probit", shuffle=False
    )
    df_analysis.drop(columns=tasks_nb, inplace=True)
    compute_and_plot_diff(df_analysis, main_model, tasks, study, path)


def compute_and_plot_diff(df_analysis, main_model, tasks, study, path, name="main"):
    pre, post = (
        df_analysis[df_analysis["task_status"] == "PRE_TEST"],
        df_analysis[df_analysis["task_status"] == "POST_TEST"],
    )
    pre_zpdes, pre_baseline = (
        pre[pre["condition"] == "zpdes"],
        pre[pre["condition"] == "baseline"],
    )
    post_zpdes, post_baseline = (
        post[post["condition"] == "zpdes"],
        post[post["condition"] == "baseline"],
    )
    # Now project the data:
    pre_zpdes[tasks], post_zpdes[tasks] = (
        main_model.transform(pre_zpdes[tasks]),
        main_model.transform(post_zpdes[tasks]),
    )
    pre_baseline[tasks], post_baseline[tasks] = (
        main_model.transform(pre_baseline[tasks]),
        main_model.transform(post_baseline[tasks]),
    )
    PC_names = {col_task: f"PC{i}" for i, col_task in enumerate(tasks)}
    pre_zpdes.rename(columns=PC_names, inplace=True)
    post_zpdes.rename(columns=PC_names, inplace=True)
    pre_baseline.rename(columns=PC_names, inplace=True)
    post_baseline.rename(columns=PC_names, inplace=True)
    Path(f"{path}/ttests_data/").mkdir(parents=True, exist_ok=True)
    for pc_name in PC_names.values():
        plot_diff_on_PC(
            pre_baseline,
            post_baseline,
            pre_zpdes,
            post_zpdes,
            pc_name,
            study,
            path,
            name=name,
        )


def run_PCA(
    config,
    tasks,
    tasks_nb,
    kept_columns,
    models=["PCA"],
    study_diff="v3_prolific",
    n_components=19,
    add_title="",
):
    # Loop over every interesting combinations:
    for studies, time_condition_to_keep, expe_name in config:
        # First retrieve the correct df:
        df = get_df(studies, tasks, tasks_nb)
        # With all data:
        df = df[kept_columns + tasks + tasks_nb]
        df.sort_values(by=["participant_id", "task_status"], inplace=True)
        # Filter time_conditions
        if time_condition_to_keep != "all":
            df = df[df["task_status"] == time_condition_to_keep]
        # Let's create a folder to store our results
        # Results are put in the first study in list, to make sure there is no override we add all studies to name
        path_to_store = f"outputs/{studies[0]}/dimensionality_reduction/{add_title}-{studies}-{expe_name}"
        for model_name in models:
            path = f"{path_to_store}/{model_name}/"
            Path(path).mkdir(parents=True, exist_ok=True)

            # First step is to fit the model:
            (
                latent_df,
                average_loadings,
                main_model_loadings,
                main_variable_contributions,
                main_model,
                rmses,
                correlations,
            ) = run_dimensionality_reduction(
                df,
                tasks,
                tasks_nb,
                n_components=n_components,
                DR_type=model_name,
                normalization_mode="probit",
            )

            # Second step is to save results of the PCA:
            save_results(
                latent_df,
                n_components,
                model_name,
                tasks,
                main_variable_contributions,
                main_model,
                path,
                average_loadings,
                main_model_loadings,
                rmses,
                correlations,
            )

            # Finally we look for enhancement in the latent space:
            study_diff_df = get_df([study_diff], tasks, tasks_nb)
            get_diff_from_PCA(
                study_diff_df,
                main_model,
                tasks,
                tasks_nb,
                study_diff,
                path,
                kept_columns,
            )
