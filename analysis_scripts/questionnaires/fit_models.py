import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import pandas as pd
import statsmodels.formula.api as smf
import arviz as az
import scipy.stats as stats
from pathlib import Path

from analysis_scripts.questionnaires.utils import filter_condition, create_file_to_store, write_reg_results_to_file
from analysis_scripts.questionnaires.visualize import generate_diagnostic_plots, plot_participants_lines


def get_regression_line(df, condition, range_max, path_to_store):
    '''Returns the regression parameters for each group + run an ANOVA groupxtime to assess significant diff'''
    zpdes, baseline = filter_condition(df, 'zpdes'), filter_condition(df, 'baseline')
    mdf_no_random_slopes_zpdes = smf.mixedlm(f"{condition} ~ session_id", zpdes, groups=zpdes["id_participant"]).fit()
    mdf_no_random_slopes_baseline = smf.mixedlm(f"{condition} ~ session_id", baseline,
                                                groups=baseline["id_participant"]).fit()
    zpdes_p = mdf_no_random_slopes_zpdes.params['Intercept'], mdf_no_random_slopes_zpdes.params['session_id']
    baseline_p = mdf_no_random_slopes_baseline.params['Intercept'], mdf_no_random_slopes_baseline.params['session_id']
    zpdes['group'] = 'zpdes'
    baseline['group'] = 'baseline'
    # Combine the dataframes
    combined_data = pd.concat([zpdes, baseline], ignore_index=True)
    # Define the mixed model with group, session_id, and their interaction
    model_formula = f"{condition} ~ group*session_id"
    mdf_combined = smf.mixedlm(model_formula, combined_data, groups=combined_data["id_participant"]).fit()
    shapiro_test = stats.shapiro(mdf_combined.resid)
    info = str(mdf_combined.summary()) + f"\n Shapiro test: {shapiro_test.statistic}, p={shapiro_test.pvalue} \n"
    info_bayesian = f"\n Bayesian analysis {condition}: \n" + get_bayesian_mixed_lm_model(combined_data, condition,
                                                                                          path_to_store,
                                                                                          range_max=range_max)
    return zpdes_p, baseline_p, info, info_bayesian


def build_model(data, range_max, condition):
    # Converting categorical variables to codes
    data['group_code'] = data['group'].astype('category').cat.codes
    data['id_code'] = data['id_participant'].astype('category').cat.codes
    n_groups = len(data['id_participant'].unique())
    with pm.Model() as model:
        # Priors
        intercept = pm.Normal('Intercept', mu=range_max, sigma=range_max / 10)
        group_slope = pm.Normal('group_slope', mu=0, sigma=1)
        session_id_slope = pm.Normal('session_id_slope', mu=0, sigma=1)
        group_session_interaction = pm.Normal('group_session_interaction', mu=0, sigma=1)
        sigma = pm.HalfNormal('sigma', sigma=10)
        participant_intercept = pm.Normal('participant_Intercept', mu=0, sigma=range_max / 10, shape=n_groups)
        # Expected value
        mu = (intercept +
              group_slope * data['group_code'] +
              session_id_slope * data['session_id'] +
              group_session_interaction * data['group_code'] * data['session_id'] +
              participant_intercept[data['id_code']])
        # Likelihood
        likelihood = pm.Normal('y', mu=mu, sigma=sigma, observed=data[condition])
    return model


def get_bayesian_mixed_lm_model(data, condition, path_to_store, range_max=20, ):
    # Build model
    model = build_model(data, range_max, condition)
    # Then get MCMC estimations
    with model:
        # Inference
        trace = pm.sample(10000, tune=1000, target_accept=0.95)
    generate_diagnostic_plots(trace, condition, path_to_store)
    # Summary
    summary_df = az.summary(trace,
                            var_names=[var for var in trace.posterior.data_vars if var != 'participant_Intercept'])
    # Probability calculations
    rope = 0.001
    # A dictionary to hold the SDDR values
    sddr_values = {}
    # Define the parameters based on their priors in the model
    parameter_priors = {
        'group_slope': {'mu': 0, 'sigma': 1},
        'session_id_slope': {'mu': 0, 'sigma': 1},
        'group_session_interaction': {'mu': 0, 'sigma': 1},
    }
    summary_str = "Bayesian model summary:\n"
    for varname in ['Intercept', 'group_slope', 'session_id_slope', 'group_session_interaction']:
        flat_samples = trace.posterior[varname].values[:, 1000:].flatten()
        p_positive = (flat_samples > 0).mean()
        p_threshold = (abs(flat_samples) > rope).mean()
        summary_df.loc[varname, 'P(>0)'] = p_positive
        summary_df.loc[varname, f'P(||>{rope})'] = p_threshold
        direction = 'positive' if np.median(flat_samples) > 0 else 'negative'
        prob_in_direction = p_positive if direction == "positive" else 1 - p_positive
        summary_df.loc[varname, f'P(dir>{rope})'] = prob_in_direction
        # SDDR:
        if varname in parameter_priors:
            kde = stats.gaussian_kde(flat_samples)
            posterior_density_at_zero = kde(0)  # Evaluate the density at zero
            prior_density_at_zero = stats.norm.pdf(0, loc=parameter_priors[varname]['mu'],
                                                   scale=parameter_priors[varname]['sigma'])
            sddr = prior_density_at_zero / posterior_density_at_zero
            summary_df.loc[varname, 'SDDR'] = sddr
    summary_str += summary_df.to_string()
    return summary_str


def get_advanced_scatter_plots(questionnaire_name, study, conditions, xtickslabels, possible_values, yticks=[],
                               ytickslabels=[], plot_title=False):
    # First retrieve df:
    df = pd.read_csv(f'data/{study}/questionnaires/{questionnaire_name}/{questionnaire_name}.csv')
    # Just make sure we don't look at post cog assessment:
    df = df.query('session_id != 0 & session_id != 9')

    zpdes, baseline = filter_condition(df, 'zpdes'), filter_condition(df, 'baseline')
    path_to_store = f'outputs/{study}/questionnaires/{questionnaire_name}/figures/inferential_scatter_plots/'
    Path(path_to_store).mkdir(parents=True, exist_ok=True)

    # Plot the participant level:
    dict_groups = {"zpdes": zpdes, "baseline": baseline}
    file_name, file_name_bayesian = create_file_to_store(path_to_store, conditions)
    add_legend = True
    # Iterate over condition:
    for condition in conditions:
        print(f"===================={condition}====================")
        plt.figure(figsize=(8, 6), dpi=300)

        # First fit the linear regression model
        zpdes_p, baseline_p, info_stats, info_bayesian = get_regression_line(
            df[['id_participant', 'session_id', 'condition', condition]],
            condition, range_max=np.max(possible_values), path_to_store=path_to_store)

        # Keep the results in txt files
        write_reg_results_to_file(file_name, file_name_bayesian, info_stats, info_bayesian)
        plot_participants_lines(dict_groups, condition, xtickslabels, zpdes_p, baseline_p, ytickslabels, yticks,
                                plot_title, add_legend, path_to_store)
