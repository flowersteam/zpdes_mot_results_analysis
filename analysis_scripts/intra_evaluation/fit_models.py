import os
import pandas as pd
import pymc as pm
import arviz as az
import numpy as np
from statsmodels.formula.api import ols
import statsmodels.api as sm
import scipy.stats as stats

from analysis_scripts.intra_evaluation.utils import transform_to_long

N_CORES = int(os.getenv("PYMC_CORES", "1"))


def run_anonva_per_activity(data, idx):
    zpdes, baseline = data["zpdes"]["points"][idx], data["baseline"]["points"][idx]
    # Transform both datasets
    df_zpdes = transform_to_long(zpdes, "zpdes")
    df_baseline = transform_to_long(baseline, "baseline")
    df_combined = pd.concat([df_zpdes, df_baseline]).reset_index(drop=True)
    # Fit the model
    model = ols(
        "measurement ~ C(group) + C(time) + C(group):C(time)", data=df_combined
    ).fit()
    # Perform ANOVA
    anova_results = sm.stats.anova_lm(model, typ=2)
    p_val_group = anova_results.loc["C(group)", "PR(>F)"]
    p_val_time = anova_results.loc["C(time)", "PR(>F)"]
    p_val_interaction = anova_results.loc["C(group):C(time)", "PR(>F)"]
    return p_val_group, p_val_time, p_val_interaction


def get_bayesian_mixed_lm_model(data, idx, range_max=1, seed=42):
    # Converting categorical variables to codes
    df_zpdes, df_baseline = (
        pd.DataFrame(data["zpdes"]["points"][idx]),
        pd.DataFrame(data["baseline"]["points"][idx]),
    )
    df_zpdes["condition"], df_baseline["condition"] = "zpdes", "baseline"
    df = pd.concat([df_zpdes, df_baseline], ignore_index=True)
    df["participant_id"] = range(1, len(df) + 1)
    data = pd.melt(
        df,
        id_vars=["participant_id", "condition"],
        value_vars=[i for i in range(4)],
        var_name="timepoint",
        value_name="measurement",
    )
    data["group_code"] = data["condition"].astype("category").cat.codes
    data["id_code"] = data["participant_id"].astype("category").cat.codes
    n_groups = len(data["participant_id"].unique())
    return fit_and_describe_bayesian_model(data, range_max, n_groups, seed=seed)


def get_average_mixed_lm_model(data, seed=42):
    data = data.rename(columns={f"s_{i}": i for i in range(4)})
    data = pd.melt(
        data,
        id_vars=["participant_id", "condition"],
        value_vars=[i for i in range(4)],
        var_name="timepoint",
        value_name="measurement",
    )
    data["group_code"] = data["condition"].astype("category").cat.codes
    data["id_code"] = data["participant_id"].astype("category").cat.codes
    n_groups = len(data["participant_id"].unique())
    range_max = 1
    return fit_and_describe_bayesian_model(data, range_max, n_groups, seed=seed)


def fit_and_describe_bayesian_model(data, range_max, n_groups, seed):
    with pm.Model() as model:
        id_code = pm.Data("id_code", data["id_code"])
        # Option 1: int (si timepoint doit être 0..3)
        data["timepoint"] = pd.to_numeric(data["timepoint"], errors="raise").astype(
            "int64"
        )
        timepoint = pm.Data("timepoint", data["timepoint"].to_numpy())
        # timepoint = pm.Data("timepoint", data["timepoint"])
        # Priors
        intercept = pm.Normal("Intercept", mu=range_max, sigma=range_max / 10)
        group_slope = pm.Normal("group_slope", mu=0, sigma=0.05)
        session_id_slope = pm.Normal("session_id_slope", mu=0, sigma=0.05)
        group_session_interaction = pm.Normal(
            "group_session_interaction", mu=0, sigma=0.05
        )
        sigma = pm.HalfNormal("sigma", sigma=0.1)
        participant_intercept = pm.Normal(
            "participant_Intercept", mu=0, sigma=range_max / 10, shape=n_groups
        )
        # Expected value
        mu = (
            intercept
            + group_slope * data["group_code"]
            + session_id_slope * timepoint
            + group_session_interaction * data["group_code"] * timepoint
            + participant_intercept[id_code]
        )
        # Likelihood
        likelihood = pm.Normal("y", mu=mu, sigma=sigma, observed=data["measurement"])
        # Inference
        trace = pm.sample(5000, tune=1000, target_accept=0.95, seed=seed, cores=N_CORES)
    # Summary
    summary_df = az.summary(
        trace,
        var_names=[
            var for var in trace.posterior.data_vars if var != "participant_Intercept"
        ],
    )
    # Probability calculations
    rope = 0.001
    # A dictionary to hold the SDDR values
    sddr_values = {}
    # Define the parameters based on their priors in the model
    parameter_priors = {
        "group_slope": {"mu": 0, "sigma": 0.05},
        "session_id_slope": {"mu": 0, "sigma": 0.05},
        "group_session_interaction": {"mu": 0, "sigma": 0.05},
    }
    summary_str = "Bayesian model summary:\n"
    dict = {}
    for varname in [
        "Intercept",
        "group_slope",
        "session_id_slope",
        "group_session_interaction",
    ]:
        dict[varname] = {}
        flat_samples = trace.posterior[varname].values[:, 1000:].flatten()
        p_positive = (flat_samples > 0).mean()
        p_threshold = (abs(flat_samples) > rope).mean()
        summary_df.loc[varname, "P(>0)"] = p_positive
        summary_df.loc[varname, f"P(||>{rope})"] = p_threshold
        direction = "positive" if np.median(flat_samples) > 0 else "negative"
        prob_in_direction = p_positive if direction == "positive" else 1 - p_positive
        dict[varname]["p_effect"] = prob_in_direction
        dict[varname]["3%"] = summary_df.loc[varname, "hdi_3%"]
        dict[varname]["97%"] = summary_df.loc[varname, "hdi_97%"]
        summary_df.loc[varname, f"P(dir>{rope})"] = prob_in_direction
        # SDDR:
        if varname in parameter_priors:
            kde = stats.gaussian_kde(flat_samples)
            posterior_density_at_zero = kde(0)  # Evaluate the density at zero
            prior_density_at_zero = stats.norm.pdf(
                0,
                loc=parameter_priors[varname]["mu"],
                scale=parameter_priors[varname]["sigma"],
            )
            sddr = posterior_density_at_zero / prior_density_at_zero
            summary_df.loc[varname, "SDDR"] = sddr
            dict[varname]["BF"] = sddr
    summary_str += summary_df.to_string()
    return dict, summary_str
