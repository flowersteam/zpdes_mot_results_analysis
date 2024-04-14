import json
import time

from preprocessing.preprocessing import pre_process_all
from hierarchical_bayesian_models.fit_models import get_traces
from hierarchical_bayesian_models.visualize import plot_traces_and_deltas
from hierarchical_bayesian_models.report import get_latex_tables

# Define on what studies you want to conduct analysis
studies = ["v3_prolific"]

# Load a bunch of config files to choose what tasks/conditions/metrics you want to look at
# First load the conditions you are interested in:
with open('config/main_config_model_fit.JSON', 'r') as f:
    all_conditions = json.load(f)

# Parameters for fitting mode (e.g nb samples)
with open('config/config_models_fit.JSON', 'r') as f:
    config_fit_models = json.load(f)

# These are specific parameters to plot the results
with open('config/config_figures.JSON', 'r') as f:
    config_fig = json.load(f)


if __name__ == "__main__":
    start_time = time.time()

    studies = ["v3_prolific"]
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

    # Record the end time
    end_time = time.time()
    # Calculate the time taken
    time_taken = end_time - start_time
    print(f"Time taken to run the script: {time_taken} seconds")