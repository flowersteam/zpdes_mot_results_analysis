import pymc as pm
from ast import literal_eval
import pandas as pd


def build_model(type, data, task_condition=None, coords=None):
    models = {"hierarbinom": build_hierar_binom,
              "hierar_binom_n_var": build_hierar_binom_n_var,
              "hierar_binom_gain": build_hierar_binom_gain,
              "hierar_binom_n_gain": build_hierar_binom_n_gain,
              "RT_normal": build_RT_normal,
              "RT_normal_hyper": build_RT_normal_hyper,
              "ufov_threshold": build_ufov_threshold,
              "ufov_threshold_hyper": build_ufov_threshold_hyper,
              "switching_cost_hyper": build_switching_cost_RT_hyper,
              "switching_cost": build_switching_cost_RT,
              "precision_normal_hyper": build_precision_moteval_hyper,
              "precision_normal": build_precision_normal,
              "hierar_binom_covar_pre": build_hierar_binom_covar_pre,
              "sdt": build_sdt_normal
              }
    return models[type](data, task_condition, coords)


def build_precision_moteval_hyper(data_combined, task_condition, coords):
    with pm.Model(coords=coords) as precision_model:
        # Hyperpriors for population mean and standard deviation
        mu_hyper = pm.Normal('mu_hyper', mu=0.5, sigma=0.1)
        sigma_hyper = pm.HalfNormal('sigma_hyper', 0.1)
        # Data
        g = pm.MutableData("g", data_combined.condition_idx, dims="obs_id")
        t = pm.MutableData("t", data_combined.time_idx, dims="obs_id")
        # Participant-level means
        mu = pm.Normal('mu', mu=mu_hyper, sigma=sigma_hyper, dims=("condition", "time"))
        # Deterministic parameters for the differences
        # If 2 groups:
        # pdb.set_trace()
        if len(coords['condition']) > 1:
            delta_zpdes = pm.Deterministic("delta_zpdes", mu[1, 1] - mu[1, 0])  # post - pre for zpdes
            delta_baseline = pm.Deterministic("delta_baseline", mu[0, 1] - mu[0, 0])  # post - pre for baseline
            delta_groups = pm.Deterministic("delta_groups", delta_zpdes - delta_baseline)
        # Else, i.e control:
        else:
            delta = pm.Deterministic("delta", mu[0, 1] - mu[0, 0])
        # Common standard deviation for all participants
        sigma = pm.HalfNormal('sigma', 0.1, dims=("condition", "time"))
        # likelihood
        pm.Normal("y", mu=mu[g, t], sigma=sigma[g, t],
                  observed=data_combined[f'{task_condition}-accuracy_continuous'], dims="obs_id")
    return precision_model


def build_precision_normal(data_combined, task_condition, coords):
    with pm.Model(coords=coords) as precision_model:
        # Hyperpriors for population mean and standard deviation
        # mu_hyper = pm.Normal('mu_hyper', mu=0.5, sigma=0.1)
        # sigma_hyper = pm.HalfNormal('sigma_hyper', 0.1)
        # Data
        g = pm.MutableData("g", data_combined.condition_idx, dims="obs_id")
        t = pm.MutableData("t", data_combined.time_idx, dims="obs_id")
        # Participant-level means
        mu = pm.Normal('mu', mu=0.5, sigma=0.1, dims=("condition", "time"))
        # Deterministic parameters for the differences
        # If 2 groups:
        # pdb.set_trace()
        if len(coords['condition']) > 1:
            delta_zpdes = pm.Deterministic("delta_zpdes", mu[1, 1] - mu[1, 0])  # post - pre for zpdes
            delta_baseline = pm.Deterministic("delta_baseline", mu[0, 1] - mu[0, 0])  # post - pre for baseline
            delta_groups = pm.Deterministic("delta_groups", delta_zpdes - delta_baseline)
        # Else, i.e control:
        else:
            delta = pm.Deterministic("delta", mu[0, 1] - mu[0, 0])
        # Common standard deviation for all participants
        sigma = pm.HalfNormal('sigma', 0.1, dims=("condition", "time"))
        # likelihood
        pm.Normal("y", mu=mu[g, t], sigma=sigma[g, t],
                  observed=data_combined[f'{task_condition}-accuracy_continuous'], dims="obs_id")
    return precision_model


def build_hierar_binom(data_combined, task_condition, coords):
    n = data_combined[f"{task_condition}-nb"].unique()[0]
    with pm.Model(coords=coords) as hierarchical_model:
        # Global-level priors (one for each condition and time)
        beta = pm.Beta("beta", alpha=1, beta=1, dims=("condition", "time"))
        # Deterministic parameters for the differences
        if len(coords['condition']) > 1:
            delta_zpdes = pm.Deterministic("delta_zpdes", beta[1, 1] - beta[1, 0])  # post - pre for zpdes
            delta_baseline = pm.Deterministic("delta_baseline", beta[0, 1] - beta[0, 0])  # post - pre for baseline
            delta_groups = pm.Deterministic("delta_groups", delta_zpdes - delta_baseline)
        # Else, i.e control:
        else:
            delta = pm.Deterministic("delta", beta[0, 1] - beta[0, 0])
        # Data
        g = pm.MutableData("g", data_combined.condition_idx, dims="obs_id")
        t = pm.MutableData("t", data_combined.time_idx, dims="obs_id")
        # Introduce a variability term for the local-level
        sigma = pm.HalfNormal('sigma', 0.1, dims=("condition", "time"))
        # Model local-level p's using the global-level p's as mean and sigma as SD
        p_observation = pm.Beta('p_observation', mu=beta[g, t], sigma=sigma[g, t], dims="obs_id")
        # Likelihood
        pm.Binomial("y", n=n, p=p_observation, observed=data_combined[f'{task_condition}-correct'].values,
                    dims="obs_id")
    return hierarchical_model


def build_hierar_binom_gain(data_combined, task_condition, coords):
    n = data_combined[f"{task_condition}-nb"].unique()[0]
    with pm.Model(coords=coords) as hierarchical_model:
        # Global-level priors (one for each condition and time)
        beta = pm.Beta("beta", alpha=1, beta=1, dims=("condition", "time"))
        # Deterministic parameters for the differences
        if len(coords['condition']) > 1:
            delta_zpdes = pm.Deterministic("delta_zpdes",
                                           (beta[1, 1] - beta[1, 0]) / (1 - beta[1, 0]))  # post - pre for zpdes
            delta_baseline = pm.Deterministic("delta_baseline",
                                              (beta[0, 1] - beta[0, 0]) / (1 - beta[0, 0]))  # post - pre for baseline
            delta_groups = pm.Deterministic("delta_groups", delta_zpdes - delta_baseline)
        # Else, i.e control:
        else:
            delta = pm.Deterministic("delta", beta[0, 1] - beta[0, 0])
        # Data
        g = pm.MutableData("g", data_combined.condition_idx, dims="obs_id")
        t = pm.MutableData("t", data_combined.time_idx, dims="obs_id")
        # Introduce a variability term for the local-level
        sigma = pm.HalfNormal('sigma', 0.1, dims=("condition", "time"))
        # Model local-level p's using the global-level p's as mean and sigma as SD
        p_observation = pm.Beta('p_observation', mu=beta[g, t], sigma=sigma[g, t], dims="obs_id")
        # Likelihood
        pm.Binomial("y", n=n, p=p_observation, observed=data_combined[f'{task_condition}-correct'].values,
                    dims="obs_id")
    return hierarchical_model


def build_hierar_binom_n_var(data_combined, task_condition, coords):
    with pm.Model(coords=coords) as hierarchical_model:
        # Global-level priors (one for each condition and time)
        beta = pm.Beta("beta", alpha=1, beta=1, dims=("condition", "time"))
        # Deterministic parameters for the differences
        if len(coords['condition']) > 1:
            delta_zpdes = pm.Deterministic("delta_zpdes", beta[1, 1] - beta[1, 0])  # post - pre for zpdes
            delta_baseline = pm.Deterministic("delta_baseline", beta[0, 1] - beta[0, 0])  # post - pre for baseline
            delta_groups = pm.Deterministic("delta_groups", delta_zpdes - delta_baseline)
        else:
            delta = pm.Deterministic("delta", beta[0, 1] - beta[0, 0])  # post - pre for ctrl
        # Data
        g = pm.MutableData("g", data_combined.condition_idx, dims="obs_id")
        t = pm.MutableData("t", data_combined.time_idx, dims="obs_id")
        n = pm.MutableData("n", data_combined[f'{task_condition}-nb'], dims="obs_id")
        # Introduce a variability term for the local-level
        sigma = pm.HalfNormal('sigma', 0.1, dims=("condition", "time"))
        # Model local-level p's using the global-level p's as mean and sigma as SD
        p_observation = pm.Beta('p_observation', mu=beta[g, t], sigma=sigma[g, t], dims="obs_id")
        # Likelihood
        pm.Binomial("y", n=n, p=p_observation, observed=data_combined[f'{task_condition}-correct'].values,
                    dims="obs_id")
    return hierarchical_model


def build_hierar_binom_n_gain(data_combined, task_condition, coords):
    with pm.Model(coords=coords) as hierarchical_model:
        # Global-level priors (one for each condition and time)
        beta = pm.Beta("beta", alpha=1, beta=1, dims=("condition", "time"))
        # Deterministic parameters for the differences
        if len(coords['condition']) > 1:
            delta_zpdes = pm.Deterministic("delta_zpdes",
                                           (beta[1, 1] - beta[1, 0]) / (1 - beta[1, 0]))  # post - pre for zpdes
            delta_baseline = pm.Deterministic("delta_baseline",
                                              (beta[0, 1] - beta[0, 0]) / (1 - beta[0, 0]))
            delta_groups = pm.Deterministic("delta_groups", delta_zpdes - delta_baseline)
        else:
            delta = pm.Deterministic("delta", beta[0, 1] - beta[0, 0])  # post - pre for ctrl
        # Data
        g = pm.MutableData("g", data_combined.condition_idx, dims="obs_id")
        t = pm.MutableData("t", data_combined.time_idx, dims="obs_id")
        n = pm.MutableData("n", data_combined[f'{task_condition}-nb'], dims="obs_id")
        # Introduce a variability term for the local-level
        sigma = pm.HalfNormal('sigma', 0.1, dims=("condition", "time"))
        # Model local-level p's using the global-level p's as mean and sigma as SD
        p_observation = pm.Beta('p_observation', mu=beta[g, t], sigma=sigma[g, t], dims="obs_id")
        # Likelihood
        pm.Binomial("y", n=n, p=p_observation, observed=data_combined[f'{task_condition}-correct'].values,
                    dims="obs_id")
    return hierarchical_model


def build_hierar_binom_covar_pre(data_combined, task_condition, coords):
    with pm.Model(coords=coords) as hierarchical_model:
        # Global-level priors (one for each condition and time)
        beta = pm.Beta("beta", alpha=1, beta=1, dims=("condition", "time"))

        # New: Coefficient for the pre-test covariate
        pretest_coeff = pm.Normal("pretest_coeff", mu=0, sigma=1, dims="condition")

        # Deterministic parameters for the differences
        if len(coords['condition']) > 1:
            delta_zpdes = pm.Deterministic("delta_zpdes", beta[1, 1] - beta[1, 0])  # post - pre for zpdes
            delta_baseline = pm.Deterministic("delta_baseline", beta[0, 1] - beta[0, 0])  # post - pre for baseline
            delta_groups = pm.Deterministic("delta_groups", delta_zpdes - delta_baseline)
        else:
            delta = pm.Deterministic("delta", beta[0, 1] - beta[0, 0])  # post - pre for ctrl

        # Data
        g = pm.MutableData("g", data_combined.condition_idx, dims="obs_id")
        t = pm.MutableData("t", data_combined.time_idx, dims="obs_id")
        n = pm.MutableData("n", data_combined[f'{task_condition}-nb'], dims="obs_id")

        # New: pre-test accuracy as covariate
        acc_pre = data_combined.groupby('participant_id')[f'{task_condition}-accuracy'].transform('first')
        pretest_accuracy = pm.MutableData("pretest_accuracy", acc_pre, dims="obs_id")

        # Introduce a variability term for the local-level
        sigma = pm.HalfNormal('sigma', 0.1, dims=("condition", "time"))

        # Model local-level p's using the global-level p's as mean and sigma as SD, and add pre-test accuracy as a covariate
        p_mean = beta[g, t] + pretest_coeff[g] * pretest_accuracy
        p_mean = pm.Deterministic("p_mean", pm.math.invlogit(p_mean))  # Ensure it's between 0 and 1
        p_observation = pm.Beta('p_observation', mu=p_mean, sigma=sigma[g, t], dims="obs_id")

        # Likelihood
        pm.Binomial("y", n=n, p=p_observation, observed=data_combined[f'{task_condition}-correct'].values,
                    dims="obs_id")
    return hierarchical_model


def build_RT_normal_hyper(data_combined, task_condition, coords):
    with pm.Model(coords=coords) as RT_model:
        # Data
        g = pm.MutableData("g", data_combined.condition_idx, dims="obs_id")
        t = pm.MutableData("t", data_combined.time_idx, dims="obs_id")
        # Hyperpriors for population mean and standard deviation
        mu_hyper = pm.Uniform('mu_hyper', lower=0, upper=3000)
        sigma_hyper = pm.HalfNormal('sigma_hyper', 100)
        # Participant-level means
        mu = pm.Normal('mu', mu=mu_hyper, sigma=sigma_hyper, dims=("condition", "time"))
        # Common standard deviation for all participants
        sigma = pm.HalfNormal('sigma', 0.1, dims=("condition", "time"))
        # Deterministic parameters for the differences
        if len(coords['condition']) > 1:
            delta_zpdes = pm.Deterministic("delta_zpdes", mu[1, 1] - mu[1, 0])  # post - pre for zpdes
            delta_baseline = pm.Deterministic("delta_baseline", mu[0, 1] - mu[0, 0])  # post - pre for baseline
            delta_groups = pm.Deterministic("delta_groups", delta_zpdes - delta_baseline)
        else:
            delta = pm.Deterministic("delta", mu[0, 1] - mu[0, 0])  # post - pre for zpdes
        # likelihood
        pm.Normal("y", mu=mu[g, t], sigma=sigma[g, t],
                  observed=data_combined[f'{task_condition}-rt'], dims="obs_id")
    return RT_model


def build_RT_normal(data_combined, task_condition, coords):
    with pm.Model(coords=coords) as RT_model:
        # Data
        g = pm.MutableData("g", data_combined.condition_idx, dims="obs_id")
        t = pm.MutableData("t", data_combined.time_idx, dims="obs_id")
        # Hyperpriors for population mean and standard deviation
        # mu_hyper = pm.Uniform('mu_hyper', lower=0, upper=3000)
        # sigma_hyper = pm.HalfNormal('sigma_hyper', 100)
        # Participant-level means
        mu = pm.Normal('mu', mu=500, sigma=50, dims=("condition", "time"))
        # Common standard deviation for all participants
        sigma = pm.HalfNormal('sigma', 10, dims=("condition", "time"))
        # Deterministic parameters for the differences
        if len(coords['condition']) > 1:
            delta_zpdes = pm.Deterministic("delta_zpdes", mu[1, 1] - mu[1, 0])  # post - pre for zpdes
            delta_baseline = pm.Deterministic("delta_baseline", mu[0, 1] - mu[0, 0])  # post - pre for baseline
            delta_groups = pm.Deterministic("delta_groups", delta_zpdes - delta_baseline)
        else:
            delta = pm.Deterministic("delta", mu[0, 1] - mu[0, 0])  # post - pre for zpdes
        # likelihood
        pm.Normal("y", mu=mu[g, t], sigma=sigma[g, t],
                  observed=data_combined[f'{task_condition}-rt'], dims="obs_id")
        # pm.LogNormal("y", mu=mu[g, t], sigma=sigma[g, t],
        #              observed=data_combined[f'{task_condition}-rt'], dims="obs_id")
    return RT_model


def build_ufov_threshold_hyper(data_combined, task_condition, coords):
    mean_prior = 60
    std_prior = 100
    with pm.Model(coords=coords) as UFOV_model:
        # Data
        g = pm.MutableData("g", data_combined.condition_idx, dims="obs_id")
        t = pm.MutableData("t", data_combined.time_idx, dims="obs_id")
        # Hyperpriors for Gamma distribution parameters
        alpha_hyper = pm.Normal('alpha_hyper', mu=3.6, sigma=0.1)
        beta_hyper = pm.Normal('beta_hyper', mu=0.06, sigma=0.01)
        # Participant-level means from Gamma distribution
        mu = pm.Gamma('mu', alpha=alpha_hyper, beta=beta_hyper, dims=("condition", "time"))
        # Common standard deviation for all participants from HalfNormal
        sigma = pm.HalfNormal('sigma', std_prior, dims=("condition", "time"))
        if len(coords['condition']) > 1:
            # Deterministic parameters for the differences
            delta_zpdes = pm.Deterministic("delta_zpdes", mu[1, 1] - mu[1, 0])  # post - pre for zpdes
            delta_baseline = pm.Deterministic("delta_baseline", mu[0, 1] - mu[0, 0])  # post - pre for baseline
            delta_groups = pm.Deterministic("delta_groups", delta_zpdes - delta_baseline)
        else:
            delta = pm.Deterministic("delta", mu[0, 1] - mu[0, 0])  # post - pre for zpdes
        # likelihood with HalfNormal
        pm.Normal("y", mu=mu[g, t], sigma=sigma[g, t], observed=data_combined[f'{task_condition}-threshold'],
                  dims="obs_id")
    return UFOV_model


def build_ufov_threshold(data_combined, task_condition, coords):
    mean_prior = 60
    std_prior = 18
    with pm.Model(coords=coords) as UFOV_model:
        # Data
        g = pm.MutableData("g", data_combined.condition_idx, dims="obs_id")
        t = pm.MutableData("t", data_combined.time_idx, dims="obs_id")
        # Hyperpriors for Gamma distribution parameters
        # alpha_hyper = pm.Normal('alpha_hyper', mu=3.6, sigma=0.1)
        # beta_hyper = pm.Normal('beta_hyper', mu=0.06, sigma=0.01)
        # Participant-level means from Gamma distribution
        mu = pm.Normal('mu', mu=mean_prior, sigma=std_prior, dims=("condition", "time"))
        # Common standard deviation for all participants from HalfNormal
        sigma = pm.HalfNormal('sigma', std_prior, dims=("condition", "time"))
        if len(coords['condition']) > 1:
            # Deterministic parameters for the differences
            delta_zpdes = pm.Deterministic("delta_zpdes", mu[1, 1] - mu[1, 0])  # post - pre for zpdes
            delta_baseline = pm.Deterministic("delta_baseline", mu[0, 1] - mu[0, 0])  # post - pre for baseline
            delta_groups = pm.Deterministic("delta_groups", delta_zpdes - delta_baseline)
        else:
            delta = pm.Deterministic("delta", mu[0, 1] - mu[0, 0])  # post - pre for zpdes
        # likelihood with HalfNormal
        pm.Normal("y", mu=mu[g, t], sigma=sigma[g, t], observed=data_combined[f'{task_condition}-threshold'],
                  dims="obs_id")
    return UFOV_model


def build_switching_cost_RT(data_combined, task_condition, coords):
    data_combined['switch_indicator'] = data_combined.apply(lambda row: literal_eval(row['switch_indicator']), axis=1)
    data_combined['results_rt'] = data_combined.apply(lambda row: literal_eval(row['results_rt']), axis=1)
    # This models needs additionnal transformation of dataset:
    coords["switch_indicator"] = ["switch", "unswitch"]
    rows = []
    for idx, row in data_combined.iterrows():
        for rt, switch in zip(row['results_rt'], row['switch_indicator']):
            rows.append({
                'participant_id': idx,
                'task_status': row['task_status'],
                'condition': row['condition'],
                'results_rt': rt,
                'switch_indicator': switch,
                'condition_idx': row['condition_idx'],
                'time_idx': row['time_idx']
            })
    df_transformed = pd.DataFrame(rows)
    # Hierarchical model
    with pm.Model(coords=coords) as switching_cost_RT:
        # Data
        g = pm.MutableData("g", df_transformed.condition_idx, dims="obs_id")
        t = pm.MutableData("t", df_transformed.time_idx, dims="obs_id")
        s = pm.MutableData("s", df_transformed.switch_indicator, dims="obs_id")
        # Hyperpriors for group means
        # mu_hyper_mean = pm.Normal('mu_hyper_mean', mu=0, sigma=10)
        # mu_hyper_sd = pm.HalfNormal('mu_hyper_sd', sigma=10)
        # Hyperpriors for group standard deviations
        # sigma_hyper_sd = pm.HalfNormal('sigma_hyper_sd', sigma=10)
        # Priors for group means given the hyperpriors
        mu = pm.Normal('mu', mu=0, sigma=10, dims=("condition", "time", "switch_indicator"))
        # Priors for group standard deviations given the hyperpriors
        sigma = pm.HalfNormal('sigma', sigma=10, dims=("condition", "time", "switch_indicator"))
        # Likelihood for each observation
        rt_obs = pm.Normal('rt_obs', mu=mu[g, t, s],
                           sigma=sigma[g, t, s], observed=df_transformed["results_rt"])
        if len(coords['condition']) > 1:
            # Deterministic node for switching cost
            switching_cost = pm.Deterministic('switching_cost', mu[:, :, 1] - mu[:, :, 0])
            # Deterministic parameters for the differences
            delta_zpdes = pm.Deterministic("delta_zpdes",
                                           switching_cost[1, 1] - switching_cost[1, 0])  # post - pre for zpdes
            delta_baseline = pm.Deterministic("delta_baseline",
                                              switching_cost[0, 1] - switching_cost[0, 0])  # post - pre for baseline
            delta_groups = pm.Deterministic("delta_groups", delta_zpdes - delta_baseline)
        else:
            # Deterministic node for switching cost
            switching_cost = pm.Deterministic('switching_cost', mu[:, :, 1] - mu[:, :, 0])
            delta = pm.Deterministic("delta", switching_cost[0, 1] - switching_cost[0, 0])
    return switching_cost_RT


def build_switching_cost_RT_hyper(data_combined, task_condition, coords):
    data_combined['switch_indicator'] = data_combined.apply(lambda row: literal_eval(row['switch_indicator']), axis=1)
    data_combined['results_rt'] = data_combined.apply(lambda row: literal_eval(row['results_rt']), axis=1)
    # This models needs additionnal transformation of dataset:
    coords["switch_indicator"] = ["switch", "unswitch"]
    rows = []
    for idx, row in data_combined.iterrows():
        for rt, switch in zip(row['results_rt'], row['switch_indicator']):
            rows.append({
                'participant_id': idx,
                'task_status': row['task_status'],
                'condition': row['condition'],
                'results_rt': rt,
                'switch_indicator': switch,
                'condition_idx': row['condition_idx'],
                'time_idx': row['time_idx']
            })
    df_transformed = pd.DataFrame(rows)
    # Hierarchical model
    with pm.Model(coords=coords) as switching_cost_RT:
        # Data
        g = pm.MutableData("g", df_transformed.condition_idx, dims="obs_id")
        t = pm.MutableData("t", df_transformed.time_idx, dims="obs_id")
        s = pm.MutableData("s", df_transformed.switch_indicator, dims="obs_id")
        # Hyperpriors for group means
        mu_hyper_mean = pm.Normal('mu_hyper_mean', mu=0, sigma=10)
        mu_hyper_sd = pm.HalfNormal('mu_hyper_sd', sigma=10)
        # Hyperpriors for group standard deviations
        sigma_hyper_sd = pm.HalfNormal('sigma_hyper_sd', sigma=10)
        # Priors for group means given the hyperpriors
        mu = pm.Normal('mu', mu=mu_hyper_mean, sigma=mu_hyper_sd, dims=("condition", "time", "switch_indicator"))
        # Priors for group standard deviations given the hyperpriors
        sigma = pm.HalfNormal('sigma', sigma=sigma_hyper_sd, dims=("condition", "time", "switch_indicator"))
        # Likelihood for each observation
        rt_obs = pm.Normal('rt_obs', mu=mu[g, t, s],
                           sigma=sigma[g, t, s], observed=df_transformed["results_rt"])
        if len(coords['condition']) > 1:
            # Deterministic node for switching cost
            switching_cost = pm.Deterministic('switching_cost', mu[:, :, 1] - mu[:, :, 0])
            # Deterministic parameters for the differences
            delta_zpdes = pm.Deterministic("delta_zpdes",
                                           switching_cost[1, 1] - switching_cost[1, 0])  # post - pre for zpdes
            delta_baseline = pm.Deterministic("delta_baseline",
                                              switching_cost[0, 1] - switching_cost[0, 0])  # post - pre for baseline
            delta_groups = pm.Deterministic("delta_groups", delta_zpdes - delta_baseline)
        else:
            # Deterministic node for switching cost
            switching_cost = pm.Deterministic('switching_cost', mu[:, :, 1] - mu[:, :, 0])
            delta = pm.Deterministic("delta", switching_cost[0, 1] - switching_cost[0, 0])
    return switching_cost_RT


def build_sdt_normal(data_combined, task_condition, coords):
    with pm.Model(coords=coords) as RT_model:
        # Data
        g = pm.MutableData("g", data_combined.condition_idx, dims="obs_id")
        t = pm.MutableData("t", data_combined.time_idx, dims="obs_id")
        # Participant-level dprime
        mu_d_prime = pm.Normal('mu', mu=0, sigma=10, dims=("condition", "time"))
        sigma = pm.HalfNormal('sigma', 10, dims=("condition", "time"))
        # Deterministic parameters for the differences
        if len(coords['condition']) > 1:
            delta_zpdes = pm.Deterministic("delta_zpdes", mu_d_prime[1, 1] - mu_d_prime[1, 0])  # post - pre for zpdes
            delta_baseline = pm.Deterministic("delta_baseline", mu_d_prime[0, 1] - mu_d_prime[0, 0])  # post - pre for baseline
            delta_groups = pm.Deterministic("delta_groups", delta_zpdes - delta_baseline)
        else:
            delta = pm.Deterministic("delta", mu_d_prime[0, 1] - mu_d_prime[0, 0])  # post - pre for zpdes
        # likelihood
        pm.Normal("y", mu=mu_d_prime[g, t], sigma=sigma[g, t],
                  observed=data_combined[f'{task_condition}-dprime'], dims="obs_id")
        # pm.LogNormal("y", mu=mu[g, t], sigma=sigma[g, t],
        #              observed=data_combined[f'{task_condition}-rt'], dims="obs_id")
    return RT_model
