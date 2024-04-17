import json

from analysis_scripts.questionnaires.preprocessing import treat_LP, treat_UES, treat_TENS, treat_SIMS, treat_nasa_tlx
from analysis_scripts.questionnaires.visualize import descriptive_visualization, get_proportion_plots
from analysis_scripts.questionnaires.fit_models import get_advanced_scatter_plots

from analysis_scripts.utils import Chrono


def preprocess_all(study):
    treat_LP(study)
    treat_TENS(study)
    treat_UES(study)
    treat_SIMS(study)
    treat_nasa_tlx(study)


def run_all_descriptive_visualizations(study):
    with open('analysis_scripts/questionnaires/config/config_questionnaire.JSON', 'r') as f:
        conditions = json.load(f)
    questionnaires = ['nasa_tlx', 'ues', 'tens', 'sims', 'lp']
    for questionnaire in questionnaires:
        descriptive_visualization(study, conditions[questionnaire], questionnaire)

    # Then run proportion plots

    # UES - specific colors
    colors_barplots = ['#b2182b', '#d6604d', '#f4a582', '#fddbc7', '#d1e5f0', '#92c5de', '#4393c3', '#2166ac']
    colors_barplots.reverse()
    legend_labels = ["Strongly disagree", "Disagree", "Neither agree nor disagree", "Agree", "Strongly agree"]
    possible_values = [i for i in range(0, len(legend_labels))]
    get_proportion_plots('ues', study, 'ues', conditions['ues'], xtickslabels=["s1", "s4", "s5", "s8"],
                         possible_values=possible_values, yticks=[i / 10 for i in range(11)],
                         ytickslabels=[i * 10 for i in range(11)],
                         colors_barplots=colors_barplots, legend_labels=legend_labels)

    # SIMS
    colors_barplots = ['#b2182b', '#d6604d', '#f4a582', '#fddbc7', '#d1e5f0', '#92c5de', '#4393c3', '#2166ac']
    colors_barplots.reverse()
    possible_values = [i for i in range(0, 7)]
    legend_labels = ["Not at all", "Very little", "A little", "Moderately", "Enough", "A lot", "Exactly"]
    get_proportion_plots('sims', study, 'sims', conditions['sims'], xtickslabels=["s1", "s4", "s5", "s8"],
                         possible_values=possible_values, yticks=[i / 10 for i in range(11)],
                         ytickslabels=[i * 10 for i in range(11)],
                         colors_barplots=colors_barplots, legend_labels=legend_labels)

    # LP
    possible_values = [i for i in range(2, 8)]
    colors_barplots = ['#2166ac', '#67a9cf', '#d1e5f0', '#fddbc7', '#ef8a62', '#b2182b']
    get_proportion_plots('lp', study, 'lp', conditions['lp'], xtickslabels=["s1", "s4", "s5", "s8"],
                         possible_values=possible_values, yticks=[i / 10 for i in range(11)],
                         ytickslabels=[i * 10 for i in range(11)],
                         colors_barplots=colors_barplots, legend_labels=possible_values)


def run_mixed_effect_models(study):
    """
    This function both fit the models and get some visualization of the results (i.e 'advanced scatter plots' and
    diagnostic plots such as trace).
    :param study:
    :return:
    """
    with open('analysis_scripts/questionnaires/config/config_questionnaire.JSON', 'r') as f:
        conditions = json.load(f)

    # NASA-tlx
    possible_values = [i for i in range(0, 21)]
    print(f"====================Nasa_tlx====================")
    get_advanced_scatter_plots(questionnaire_name='nasa_tlx', study=study,
                               conditions=[c for c in conditions['nasa_tlx'] if c != 'load_index'],
                               xtickslabels=[f"s{i}" for i in range(1, 9)],
                               possible_values=possible_values, yticks=possible_values, ytickslabels=possible_values)
    get_advanced_scatter_plots(questionnaire_name='nasa_tlx', study=study, conditions=['load_index'],
                               xtickslabels=[f"s{i}" for i in range(1, 9)],
                               possible_values=[i for i in range(0, 121, 10)], yticks=[i for i in range(0, 121, 10)],
                               ytickslabels=[i for i in range(0, 121, 10)])
    print(f"====================Ues====================")
    # UES
    possible_values = [i for i in range(0, 5)]
    get_advanced_scatter_plots(questionnaire_name='ues', study=study,
                               conditions=conditions['ues'],
                               xtickslabels=["s2", "s4", "s5", "s7"],
                               possible_values=possible_values, yticks=possible_values, ytickslabels=possible_values)
    print(f"====================Learning Progress====================")
    # # LP
    possible_values = [i for i in range(2, 8)]
    get_advanced_scatter_plots(questionnaire_name='lp', study=study, conditions=conditions['lp'],
                               xtickslabels=["s1", "s4", "s5", "s8"],
                               possible_values=possible_values, yticks=possible_values, ytickslabels=possible_values)
    print(f"====================SIMS====================")
    # # SIMS
    conditions_sims = conditions['sims'][:-1]
    possible_values = [i for i in range(0, 7)]
    get_advanced_scatter_plots(questionnaire_name='sims', study=study, conditions=conditions_sims,
                               xtickslabels=["s1", "s4", "s5", "s8"],
                               possible_values=possible_values, yticks=possible_values, ytickslabels=possible_values)
    possible_values_sdi = [i for i in range(-16, 17)]
    get_advanced_scatter_plots(questionnaire_name='sims', study=study, conditions=["SDI"],
                               xtickslabels=["s1", "s4", "s5", "s8"],
                               possible_values=possible_values_sdi, yticks=possible_values_sdi,
                               ytickslabels=possible_values_sdi)
    print(f"====================TENS====================")
    # TENS
    possible_values = [i for i in range(0, 6)]
    get_advanced_scatter_plots(questionnaire_name='tens', study=study, conditions=conditions['tens'],
                               xtickslabels=["s1", "s4", "s5", "s8"],
                               possible_values=possible_values, yticks=possible_values, ytickslabels=possible_values)


def run_questionnaires_analysis(studies):
    chrono = Chrono()
    print("=====================Start [Questionnaires] - full analysis=====================")
    for study in studies:
        # First preprocess the questionnaire data
        preprocess_all(study)
        # Then get some descriptive visu
        print(f"====================Get descriptive figures====================")
        run_all_descriptive_visualizations(study)
        # Finally fit models
        print(f"====================Start Mixed effects models====================")
        run_mixed_effect_models(study)
    print(f"Time taken to run the script: {chrono.get_elapsed_time()} seconds")


if __name__ == '__main__':
    studies = ['v3_prolific']
    run_questionnaires_analysis(studies)