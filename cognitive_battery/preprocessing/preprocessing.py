from . import enumeration_ana_results as enumeration
from . import taskswitch_ana_results as ts
from . import gonogo_ana_results as gonogo
from . import moteval_ana_results as mot
from . import memorability_ana_results as memora
from . import loadblindness_ana_results as lb
from . import workingmemory_ana_results as wm
from . import ufov_ana_results as ufov


def pre_process_all(study):
    enumeration.preprocess_and_save(study=study)
    gonogo.preprocess_and_save(study=study)
    ts.preprocess_and_save(study=study)
    mot.preprocess_and_save(study=study)
    memora.preprocess_and_save(study=study)
    lb.preprocess_and_save(study=study)
    wm.preprocess_and_save(study=study)
    ufov.preprocess_and_save(study=study)


if __name__ == '__main__':
    pre_process_all('v3_prolific')
