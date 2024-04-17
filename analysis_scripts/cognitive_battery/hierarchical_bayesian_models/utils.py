import numpy as np
import scipy.stats as stats
import arviz as az


def get_SDDR(trace, var, parameter_priors={}):
    # First check if trace is already of type
    try :
        flat_samples = trace.posterior[var].values[:, 1000:].flatten()
    except AttributeError:
        flat_samples = trace.values[:, 1000:].flatten()
    kde = stats.gaussian_kde(flat_samples)
    posterior_density_at_zero = kde(0)  # Evaluate the density at zero
    prior_density_at_zero = stats.norm.pdf(0, loc=parameter_priors['mu'],
                                           scale=parameter_priors['sigma'])
    sddr = posterior_density_at_zero / prior_density_at_zero
    return sddr[0]


def get_p_in_ROPE(trace, rope_start, rope_end, var):
    return 1 - np.mean((trace.posterior[var] > rope_start) & (trace.posterior[var] < rope_end))


def get_p_of_effect(trace, var):
    proba_positive_effect = np.mean((trace.posterior[var] > 0))
    return np.max([proba_positive_effect, 1 - proba_positive_effect])


def get_hdi_bounds(trace, var, hdi_prob=0.94):
    # Compute the HDIs:
    vals = trace.posterior[var].values.flatten()
    # returns the hdi
    return az.hdi(vals, hdi_prob=0.94)
