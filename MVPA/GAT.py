import pathlib
import numpy as np
import logging
logger = logging.getLogger('MVPA')

import mne

# from sklearn.linear_model import LinearRegression, RidgeClassifier, LogisticRegression
# from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from mne.decoding import GeneralizingEstimator, cross_val_multiscore 

# prevent commandline output from getting cluttered by repetitive warnings
from warnings import filterwarnings
from sklearn.exceptions import ConvergenceWarning
filterwarnings("once", category=ConvergenceWarning)

# this is where all user-defined parameters (constants) are located
from .PARAMETERS import *



####################
# ----- MVPA ----- #
####################

def GAT_main():
    # remove existing performance matrix files from directory
    for _f in SAVE_DIR_TMP.glob('*.npy'): # FOR SAFETY, ONLY TOP-LEVEL FILE GLOB
        if _f.is_file(): _f.unlink()
        else: raise Exception(f"Please do not alter contents of {SAVE_DIR_TMP}")

    if DATA_PATH.is_file():
        files = [DATA_PATH]
        logger.info(f"Loading file {DATA_PATH}")
    elif DATA_PATH.is_dir():
        files = list(DATA_PATH.glob('*.vhdr'))
        logger.info(f"Loading files from {DATA_PATH} (found {len(files)} files)")

    files = list(SAVE_DIR_TMP.glob('*-epo.fif'))

    # run Time-Generalised Decoding for each subject individually
    for i, f in enumerate(files, start=1):
        logger.info(f"Performing Generalization Across Time MVPA for file {i}/{len(files)} : {f.name}")

        data_epochs = mne.read_epochs(f)

        pipeline = make_pipeline(StandardScaler(),
                                DECODER_MODEL)

        generalizer = GeneralizingEstimator(pipeline,
                                            scoring=SCORING,
                                            n_jobs=N_JOBS,
                                            verbose=logger.level)

        # TODO: balanced no. of trials across conditions

        # TODO: 1 event seems to get dropped here
        data_matrix = data_epochs.get_data(picks='data') # pick only good EEG channels

        # produce labels based on user-indicated condition/marker mapping
        labels = np.empty(shape=(len(data_epochs.events[:,-1])), dtype=object)
        for cond, markers in CONDITION_STIMULI.items():
            for marker in markers:
                labels[data_epochs.events[:,-1] == marker] = cond

        # run fitting/cross validation
        logger.info("Performing fitting and cross-validation")
        scores = cross_val_multiscore(generalizer,
                                    data_matrix,
                                    labels,
                                    cv=CROSS_VAL_FOLDS,
                                    n_jobs=N_JOBS
                                    )

        # store results in temp folder, to be imported later
        with open(SAVE_DIR_TMP / (f.stem[:-len('-epo')] + '-gat.npy'), 'wb') as tmp:
            np.save(tmp, scores)
            logger.info(f"Wrote cross validation scores to {tmp.name}")


    # read scores from tmp files, preserving file names in dict keys
    logger.info("Consolidating GAT results")
    tmp_results = {}
    for f in SAVE_DIR_TMP.glob('*' + '-gat.npy'):
        logger.debug(f"Unpacking {f.name}")
        with open(f, 'rb') as tmp:
            tmp_results[f.stem] = np.load(tmp)

    # pickle results for (optional) later use
    import pickle
    _f = SAVE_DIR / 'GAT_results.pickle'
    with open(_f, 'wb') as f:
        pickle.dump(tmp_results, f)

    logger.info(f"Stored GAT results at {_f}")


if __name__ == ' __main__':
    print('THIS SCRIPT IS NOT MEANT TO BE RUN INDEPENDENTLY')
