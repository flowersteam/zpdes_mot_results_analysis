import json
import os
from pathlib import Path

import arviz
import pandas as pd
import pymc as pm

from analysis_scripts.cognitive_battery.hierarchical_bayesian_models.build_models import (
    build_model,
)
from analysis_scripts.cognitive_battery.hierarchical_bayesian_models.visualize import (
    render_model_graph,
)

N_CORES = int(os.getenv("PYMC_CORES", "1"))


def get_traces(
    studies: list,
    all_conditions: dict,
    nb_samples: int = 4000,
    render_model_image: bool = False,
    get_trace: bool = True,
    seed: int = 42,
):
    """
    Iterates over all given studies and task conditions to estimate the posterior by calling the 'build_and_get_trace' function.

    For each study and task condition, this function prepares the path for data storage, loads the corresponding
    data, and invokes the model building and trace generation process. If 'render_model_image' is set to True,
    it also renders a graphical representation of the model.

    :param list studies: A list of studies to be processed.
    :param dict all_conditions: A dictionary where keys are metric types and values are dictionaries
                                with task details and conditions.
    :param int nb_samples: The number of samples to draw in the trace. Default is 4000.
    :param bool render_model_image: A flag to indicate whether to render the model image. Default is False.
    :param bool get_trace: A flag to indicate whether to get traces, it can be set to false when only saving model graphs
    :return: None
    """
    for study in studies:
        print(f"\n ====================={study}===================== \n")
        main_path_data = f"data/{study}/cognitive_battery/"
        main_path_outputs = f"outputs/{study}/cognitive_battery/"
        for metric_type in all_conditions.keys():
            for task in all_conditions[metric_type].keys():
                path_to_store = f"{main_path_outputs}/{task}"
                path_data = f"{main_path_data}/{task}"
                Path(path_to_store).mkdir(parents=True, exist_ok=True)
                # Open the correct dataframe
                if os.path.exists(f"{path_data}/{task}_lfa.csv"):
                    df = pd.read_csv(f"{path_data}/{task}_lfa.csv")
                    print(f"=====================Start [{task}]=====================")
                    for task_condition in all_conditions[metric_type][task][
                        "conditions"
                    ]:
                        for model in all_conditions[metric_type][task]["models"]:
                            print(
                                f"=====================Condition / Model: "
                                f"{study}-{metric_type}-{task}-{task_condition}-{model}====================="
                            )
                            build_and_get_trace(
                                df=df,
                                task_condition=task_condition,
                                model_type=model,
                                nb_samples=nb_samples,
                                path_to_store=f"{path_to_store}/{task_condition}",
                                graphviz=render_model_image,
                                get_trace=get_trace,
                                seed=seed,
                            )


def build_and_get_trace(
    df: pd.DataFrame,
    task_condition: dict,
    model_type: str,
    nb_samples: int,
    path_to_store: str,
    graphviz: bool = False,
    get_trace: bool = True,
    seed: int = 42,
) -> arviz.InferenceData:
    """
    Preprocess the data, build a probabilistic model, and return the trace as InferenceData.

    :param pd.DataFrame df: The dataframe containing the dataset of observations.
    :param dict task_condition: A dictionary specifying the task conditions for the model.
    :param str model_type: The type of model to build.
    :param int nb_samples: The number of samples to draw from the posterior.
    :param str path_to_store: The file path to store the InferenceData JSON.
    :param bool graphviz: (optional) If true, generates and saves a Graphviz diagram of the model.
    :param bool get_trace: (optional) If False, don't get trace
    :return: An arviz.InferenceData object containing the model trace.
    :rtype: arviz.InferenceData
    """
    # Reformat data and get a trace as Inference data
    data_combined, coords, deltas, data_pre, data_post = reformat_data(
        df, task_condition
    )
    model = build_model(
        model_type, data_combined, task_condition=task_condition, coords=coords
    )
    if graphviz:
        render_model_graph(model, path_to_store, name=model_type)
    if get_trace:
        with model:
            trace = pm.sample(
                nb_samples,
                cores=N_CORES,
                idata_kwargs={"log_likelihood": True},
                random_seed=seed,
            )
        trace.to_json(f"{path_to_store}-{model_type}_inference_data.json")
        return trace


def reformat_data(
    df: pd.DataFrame, task_condition: dict
) -> (pd.DataFrame, dict, list, pd.DataFrame, pd.DataFrame):
    """
    Reformats the input DataFrame based on task conditions and computes indices for pre and post conditions.

    This function processes the input DataFrame by creating subsets for pre-test and post-test conditions,
    assigning condition indices, and combining the data into a single DataFrame. Additionally, it computes
    delta values for the task conditions if applicable.

    :param pd.DataFrame df: The input DataFrame containing the data to be reformatted.
    :param dict task_condition: A dictionary of task conditions used to reformat the data.
    :return: A tuple containing the combined DataFrame, coordinates for conditions, delta values list,
             and separate DataFrames for pre-test and post-test conditions.
    :rtype: (pd.DataFrame, dict, list, pd.DataFrame, pd.DataFrame)
    """
    # Create 'condition' and 'time' indices
    data_pre = df[df["task_status"] == "PRE_TEST"]
    data_pre["condition_idx"] = (data_pre["condition"] == "zpdes").astype(int)
    data_pre["time_idx"] = 0  # 0 for 'pre'
    data_post = df[df["task_status"] == "POST_TEST"]
    data_post["condition_idx"] = (data_post["condition"] == "zpdes").astype(int)
    data_post["time_idx"] = 1  # 1 for 'post'
    data_pre = data_pre.set_index("participant_id")
    data_post = data_post.set_index("participant_id")
    # To handle control condition, let's add a third possibility "control"
    # Merge data_pre and data_post into one dataframe
    data_combined = pd.concat([data_pre, data_post], axis=0)
    if sum(data_combined["condition"] == "no_condition") > 0:
        conditions = ["control"]
    else:
        conditions = ["baseline", "zpdes"]
    times = ["pre", "post"]
    coords = {"condition": conditions, "time": times}
    # For individuals (might be deleted):
    if (f"{task_condition}-nb" in df) & (f"{task_condition}-correct" in df):
        n = data_combined[f"{task_condition}-nb"].unique()[0]
        deltas = (
            data_post[f"{task_condition}-correct"]
            - data_pre[f"{task_condition}-correct"]
        ) / n
    else:
        deltas = []
    return data_combined, coords, deltas, data_pre, data_post
