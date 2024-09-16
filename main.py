import warnings
import numpy as np

np.random.seed(0)
warnings.filterwarnings('ignore')

from analysis_scripts.questionnaires.run import run_questionnaires_analysis
from analysis_scripts.cognitive_battery.run import run_hierarchical_bayesian_models, run_aggregate_score, \
    run_dimensionality_reduction_models
from analysis_scripts.lgcm.run import run_lgcm_analysis
from analysis_scripts.intra_evaluation.run import run_intra_evals
from analysis_scripts.utils import Chrono


def run_all(studies):
    chrono = Chrono()
    # run_intra_evals(studies, metric_type="F1")
    run_hierarchical_bayesian_models(studies, pre_process=False, fit_models=False, get_plot=True, get_latex=False)
    # run_aggregate_score(studies)
    # run_dimensionality_reduction_models(studies)
    # run_questionnaires_analysis(studies)
    print(f"Time taken to run the script: {chrono.get_elapsed_time()} seconds")


if __name__ == '__main__':
    studies = ['v3_prolific', 'v3_utl']
    # run_lgcm_analysis(study="v3_utl")
    run_all(studies)
