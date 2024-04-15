import json
import time

from preprocessing.preprocessing import pre_process_all

from hierarchical_bayesian_models.fit_models import get_traces
from hierarchical_bayesian_models.visualize import plot_traces_and_deltas
from hierarchical_bayesian_models.report import get_latex_tables

from utils import Chrono

from PCA.fit_models import run_PCA, format_conditions

from aggregate_score.compute_aggregate import run_broad_aggregate_difference

# Define on what studies you want to conduct analysis
studies = ["v3_prolific"]

# Load a bunch of config files to choose what tasks/conditions/metrics you want to look at
# First load the conditions you are interested in:
with open('hierarchical_bayesian_models/config/main_config_model_fit.JSON', 'r') as f:
    all_conditions = json.load(f)

# Parameters for fitting mode (e.g nb samples)
with open('hierarchical_bayesian_models/config/config_models_fit.JSON', 'r') as f:
    config_fit_models = json.load(f)

# These are specific parameters to plot the results
with open('hierarchical_bayesian_models/config/config_figures.JSON', 'r') as f:
    config_fig = json.load(f)


def run_hierarchical_bayesian_models():
    chrono = Chrono()
    # First, data pre-processing
    for study in studies:
        pre_process_all(study=study)

    # Then get the traces for all studies and all conditions in defined in JSON file
    get_traces(studies, all_conditions, nb_samples=config_fit_models['nb_samples'],
               render_model_image=config_fit_models['render_model'])

    # Visualize the results
    plot_traces_and_deltas(studies, config_fig, all_conditions)

    # Write Latex Report in tables
    get_latex_tables(studies, config_fig, all_conditions)

    print(f"Time taken to run the script: {chrono.get_elapsed_time()} seconds")


def run_dimensionality_reduction_models(studies):
    chrono = Chrono()

    # This name is used mainly to store results
    global_expe_name = "probit-all-tasks-ordered"

    # Choose on what data are fitted the PCA models:
    config = []
    for study in studies:
        config.append(([study], "PRE_TEST", f"pre_test-{global_expe_name}"))
        config.append(([study], "all", f"all-{global_expe_name}"))
    # It is also possible to concatenate several studies to conduct the PCA on it with for example:
    # config.append((studies, "PRE_TEST", f"all_studies_pre_test-{global_expe_name}"))

    # Choose the difficulty condition per task to keep for dimensionality reduction:
    # These are specific parameters to plot the results
    with open('PCA/config/conditions.JSON', 'r') as f:
        conditions = json.load(f)
    tasks, tasks_nb, kept_columns = format_conditions(conditions)

    models = ["PCA", "ICA"]

    # Finally run all models specified in models (here a PCA and a FastICA)
    run_PCA(config, tasks, tasks_nb, kept_columns, study_diff="v3_prolific", models=models, n_components=24)

    # Print time taken
    print(f"Time taken to run the script: {chrono.get_elapsed_time()} seconds")


def run_aggregate_score(studies):
    chrono = Chrono()

    # As in PCA, we rely on the conditions.json to look for activities to consider in the aggregate score:
    with open('PCA/config/conditions.JSON', 'r') as f:
        conditions = json.load(f)
    tasks, tasks_nb, kept_columns = format_conditions(conditions)

    for study in studies:
        run_broad_aggregate_difference(tasks=tasks, study=study, kept_columns=kept_columns, tasks_nb=tasks_nb,
                                       expe_name=f"{study}-zscore")
        run_broad_aggregate_difference(tasks=tasks, study=study, kept_columns=kept_columns, tasks_nb=tasks_nb,
                                       normalization_type="percentile_rank", expe_name=f"{study}-percentile")
    print(f"Time taken to run the script: {chrono.get_elapsed_time()} seconds")


if __name__ == "__main__":
    studies = ["v3_prolific"]
    run_aggregate_score(studies)
