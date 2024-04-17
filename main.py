import warnings

warnings.filterwarnings('ignore')

from analysis_scripts.questionnaires.run import run_questionnaires_analysis
from analysis_scripts.cognitive_battery.run import run_hierarchical_bayesian_models, run_aggregate_score, \
    run_dimensionality_reduction_models

from analysis_scripts.utils import Chrono


def run_all(studies):
    chrono = Chrono()
    run_questionnaires_analysis(studies)
    run_aggregate_score(studies)
    run_dimensionality_reduction_models(studies)
    run_hierarchical_bayesian_models(studies)
    print(f"Time taken to run the script: {chrono.get_elapsed_time()} seconds")


if __name__ == '__main__':
    studies = ['v3_prolific']
    run_all(studies)
