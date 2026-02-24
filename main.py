import warnings
import random
import numpy as np
import os

# =========================
# Global reproducibility
# =========================

SEED = 42

# 1. Fix Python RNG
random.seed(SEED)

# 2. Fix NumPy legacy RNG
np.random.seed(SEED)

# 3. Fix BLAS / MKL nondeterminism
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

warnings.filterwarnings("ignore")

from analysis_scripts.questionnaires.run import run_questionnaires_analysis
from analysis_scripts.cognitive_battery.run import (
    run_hierarchical_bayesian_models,
    run_aggregate_score,
    run_dimensionality_reduction_models,
)
from analysis_scripts.lgcm.run import run_lgcm_analysis
from analysis_scripts.intra_evaluation.run import run_intra_evals
from analysis_scripts.utils import Chrono


def run_all(studies):
    chrono = Chrono()
    print("Start analysis")
    # run_intra_evals(studies, metric_type="F1", seed=SEED)
    # run_hierarchical_bayesian_models(
    #     studies,
    #     pre_process=False,
    #     fit_models=True,
    #     get_plot=True,
    #     get_latex=False,
    #     seed=SEED,
    # )
    # run_aggregate_score(studies)
    run_dimensionality_reduction_models(studies)
    run_questionnaires_analysis(studies)
    print(f"Time taken to run the script: {chrono.get_elapsed_time()} seconds")


if __name__ == "__main__":
    studies = ["v3_utl"]
    # run_lgcm_analysis(study="v3_utl")
    run_all(studies)
