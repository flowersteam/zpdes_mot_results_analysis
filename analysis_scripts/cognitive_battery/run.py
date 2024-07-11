import json

from analysis_scripts.cognitive_battery.preprocessing.preprocessing import pre_process_all

from analysis_scripts.cognitive_battery.hierarchical_bayesian_models.fit_models import get_traces
from analysis_scripts.cognitive_battery.hierarchical_bayesian_models.visualize import plot_traces_and_deltas
from analysis_scripts.cognitive_battery.hierarchical_bayesian_models.report import get_latex_tables

from analysis_scripts.utils import Chrono

from analysis_scripts.cognitive_battery.PCA.fit_models import run_PCA, format_conditions

from analysis_scripts.cognitive_battery.aggregate_score.compute_aggregate import run_broad_aggregate_difference

# Define on what studies you want to conduct analysis
studies = ["v3_prolific"]

# Load a bunch of config files to choose what tasks/conditions/metrics you want to look at
# First load the conditions you are interested in:
with open('analysis_scripts/cognitive_battery/hierarchical_bayesian_models/config/main_config_model_fit.JSON',
          'r') as f:
    all_conditions = json.load(f)

# Parameters for fitting mode (e.g nb samples)
with open('analysis_scripts/cognitive_battery/hierarchical_bayesian_models/config/config_models_fit.JSON', 'r') as f:
    config_fit_models = json.load(f)

# These are specific parameters to plot the results
with open('analysis_scripts/cognitive_battery/hierarchical_bayesian_models/config/config_figures.JSON', 'r') as f:
    config_fig = json.load(f)


def run_hierarchical_bayesian_models(studies, pre_process=True, fit_models=True, get_plot=True, get_latex=True):
    chrono = Chrono()
    print("\n=====================Start [Cognitive battery] - bayesian modeling=====================")

    # First, data pre-processing
    if pre_process:
        for study in studies:
            pre_process_all(study=study)

    if fit_models:
        # Then get the traces for all studies and all conditions in defined in JSON file
        get_traces(studies, all_conditions, nb_samples=config_fit_models['nb_samples'],
                   render_model_image=config_fit_models['render_model'])
    if get_plot:
        # Visualize the results
        plot_traces_and_deltas(studies, config_fig, all_conditions)

    if get_latex:
        # Write Latex Report in tables
        get_latex_tables(studies, config_fig, all_conditions)

    print(f"Time taken to run the script: {chrono.get_elapsed_time()} seconds")


def get_model_diagrams(studies):
    chrono = Chrono()
    print("\n=====================Start [Cognitive battery] - Get diagrams=====================")

    # Then get the traces for all studies and all conditions in defined in JSON file
    get_traces(studies, all_conditions, nb_samples=config_fit_models['nb_samples'],
               render_model_image=True, get_trace=False)

    print(f"Time taken to run the script: {chrono.get_elapsed_time()} seconds")


def run_dimensionality_reduction_models(studies):
    chrono = Chrono()
    print("\n=====================Start [Cognitive Battery] -  dimensionality reduction analysis=====================")
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
    with open('analysis_scripts/cognitive_battery/PCA/config/conditions.JSON', 'r') as f:
        conditions = json.load(f)
    tasks, tasks_nb, kept_columns = format_conditions(conditions)

    models = ["PCA", "ICA"]

    # Finally run all models specified in models (here a PCA and a FastICA)
    run_PCA(config, tasks, tasks_nb, kept_columns, study_diff=studies[0], models=models, n_components=24)

    # Print time taken
    print(f"Time taken to run the script: {chrono.get_elapsed_time()} seconds")


def run_aggregate_score(studies):
    chrono = Chrono()
    print("\n=====================Start [Cognitive Battery] - Aggregate score calculations=====================")
    # As in PCA, we rely on the conditions.json to look for activities to consider in the aggregate score:
    with open('analysis_scripts/cognitive_battery/PCA/config/conditions.JSON', 'r') as f:
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
