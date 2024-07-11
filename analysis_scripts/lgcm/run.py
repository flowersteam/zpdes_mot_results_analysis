from semopy import Model, semplot, calc_stats
import pandas as pd
from pathlib import Path
import numpy as np

np.random.seed(42)


def fit_model(model, data, file_name='', study='v3_prolific'):
    model.fit(data)
    results = model.inspect()
    stats_convergence = calc_stats(model)
    path = f'outputs/{study}/lgcm'
    Path(path).mkdir(parents=True, exist_ok=True)
    semplot(model, f"{path}/{file_name}-diagram.png", plot_covs=True)
    results.to_csv(f'{path}/{file_name}-results.csv')
    stats_convergence.to_csv(f'{path}/{file_name}-convergence.csv')
    return results


def run_lgcm_analysis(study):
    model_desc = """
        # Latent variables
        Intercept =~ 1*s_0 + 1*s_1 + 1*s_2 + 1*s_3
        Slope =~ 0*s_0 + 1*s_1 + 2*s_2 + 3*s_3
    
        # Allow intercept and slope to correlate
        Intercept ~~ Slope
    
        # Regression paths
        s_0 ~ Intercept
        s_1 ~ Intercept + Slope
        s_2 ~ Intercept + Slope
        s_3 ~ Intercept + Slope
    """

    # Get data:
    data = pd.read_csv(f'data/{study}/lgcm/{study}_F1_intra.csv')
    zpdes_data = data[data['condition'] == 'zpdes']
    baseline_data = data[data['condition'] == 'baseline']
    model = Model(model_desc)
    results_zpdes = fit_model(model, zpdes_data, file_name='zpdes', study=study)
    results_baseline = fit_model(model, baseline_data, file_name='baseline', study=study)
    print(results_zpdes)
    print(results_baseline)
