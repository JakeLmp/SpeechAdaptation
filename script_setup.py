#%%
import pathlib
import numpy as np

import mne

from sklearn.linear_model import LinearRegression, RidgeClassifier, LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from mne.decoding import GeneralizingEstimator, cross_val_multiscore 

from plotting import GeneralizationScoreMatrix

# setup logging level for this script
import logging
logging.basicConfig(level=logging.INFO)
mne.set_log_level(logging.ERROR)

# prevent interactive plot windows from opening
import matplotlib
matplotlib.use('Agg')

# prevent commandline output from getting cluttered by repetitive warnings
from warnings import filterwarnings
from sklearn.exceptions import ConvergenceWarning
filterwarnings("once", category=ConvergenceWarning)



#################################
# ----- SCRIPT PARAMETERS ----- #
#################################

# Data-related
DATA_PATH = r"C:\Users\Jakob\Documents\repositories\SpeechAdaptation\data\practice_data"  # Location of data file(s), may be a single file or a directory containing multiple files.
SAVE_DIRECTORY = r"C:\Users\Jakob\Documents\repositories\SpeechAdaptation\results"          # Directory name, NOT a file

# Data import arguments
# see https://mne.tools/stable/generated/mne.io.read_raw_brainvision.html#mne.io.read_raw_brainvision
DATA_ARGS = {'eog' :    ('Up', 'Down', 'Left', 'Right'),
             'misc' :   'auto',
             'scale' :  1.0
             }

CONDITION_STIMULI = {'ambi_after_EE': [103, 104, 105, 106, 107],    # 'condition name' : [stimulus markers]
                     'ambi_after_EU': [203, 204, 205, 206, 207]}

REFERENCE_ELECTRODES = ['M1', 'M2']                    # List of reference electrode names
BAD_ELECTRODES = []                                    # List electrode names that should be ignored
# BAD_SUBJECTS = []                                      # NOT IMPLEMENTED : List subject IDs that should be ignored

T_MIN = -0.2                                           # Starting time of trials in seconds (if baseline segment should be included, starting time can be negative)
T_MAX = 1.0                                            # Ending time of trials in seconds

# Resampling parameters (STRONGLY RECOMMENDED TO RESAMPLE)   
# For resampling approach see https://mne.tools/stable/auto_tutorials/preprocessing/30_filtering_resampling.html#resampling
RESAMPLE_FREQUENCY = 100                               # Frequency to downsample to. Reduces computational load of decoding procedure.
RESAMPLE_AT = 'epoch'                                  # At what point should resampling occur? (choose 'raw', 'epoch', or 'do_not_resample')

ERP_CHECK_ELECTRODES = ['Fz', 'Cz', 'Pz']              # Electrodes to plot the ERP of, as a preliminary check, saving it in the indicated directory. Leave empty to plot all channels.


# Model selection
# See https://scikit-learn.org/stable/modules/classes.html#module-sklearn.linear_model
# or  https://scikit-learn.org/stable/modules/classes.html#module-sklearn.svm

RAND_STATE = 1              # random state seed

GENERAL_ARGS = {"class_weight":'balanced',
                }

# available methods (DO NOT EDIT THIS LIST)
MODELS = {"OLS":    LinearRegression(),                                                     # Ordinary Least Squares Regression
          "LogRes": LogisticRegression(solver="liblinear", **GENERAL_ARGS),                 # Logistic Regression
          "Ridge":  RidgeClassifier(**GENERAL_ARGS),                                        # Ridge Regression / Tikhonov regularisation
          "SVC":    SVC(kernel='linear', random_state=RAND_STATE, **GENERAL_ARGS),          # Linear Support Vector Machine
          "SVM":    SVC(kernel='rbf', random_state=RAND_STATE, **GENERAL_ARGS),             # Non-linear Support Vector Machine
          }

DECODER_MODEL = MODELS['Ridge']            # choose desired model from the above list

CROSS_VAL_FOLDS = 5                         # no. of cross-validation folds to use

N_JOBS = -1                                 # no. of jobs to run in parallel. If -1, equal to number of CPU cores

# available classification scoring methods: https://scikit-learn.org/stable/modules/model_evaluation.html
SCORING = 'accuracy'



############################
# ----- DATA IMPORTS ----- #
############################

DATA_PATH = pathlib.Path(DATA_PATH)
SAVE_DIR = pathlib.Path(SAVE_DIRECTORY)

if not SAVE_DIR.exists():
    raise OSError('Save directory does not exist')

# create subdirectories in the results dir
SAVE_DIR_ERP = SAVE_DIR / "ERP"
if not SAVE_DIR_ERP.exists():
    SAVE_DIR_ERP.mkdir()

SAVE_DIR_GEN_MATRIX = SAVE_DIR / "Temporal Generalization Matrix"
if not SAVE_DIR_GEN_MATRIX.exists():
    SAVE_DIR_GEN_MATRIX.mkdir()

# TODO: rewrite this into context manager for graceful exception handling
# tmp results directory will hold intermediate results, as back up for between-subject script crashes
SAVE_DIR_TMP = SAVE_DIR / "tmp"
if not SAVE_DIR_TMP.exists():
    SAVE_DIR_TMP.mkdir()
else:
    # remove existing files from directory
    for _f in SAVE_DIR_TMP.glob('*'): # FOR SAFETY, ONLY TOP-LEVEL FILE GLOB
        if _f.is_file(): _f.unlink()
        else: raise Exception(f"Please do not alter contents of {SAVE_DIR_TMP}")

if DATA_PATH.is_file():
    files = [DATA_PATH]
    logging.info(f"Loading file {DATA_PATH}")
elif DATA_PATH.is_dir():
    files = list(DATA_PATH.glob('*.vhdr'))
    logging.info(f"Loading files from {DATA_PATH} (found {len(files)} files)")

# run Time-Generalised Decoding for each subject individually
for i, f in enumerate(files, start=1):
    logging.info(f"Processing file {i}/{len(files)} : {f.name}")
    data_raw = mne.io.read_raw_brainvision(f, **DATA_ARGS)

    data_raw.load_data()
    data_raw.set_eeg_reference(REFERENCE_ELECTRODES)
    data_raw.info['bads'] = BAD_ELECTRODES

    # resample if selected to do so here, or check if another valid option was given.
    if RESAMPLE_AT.lower() == 'raw':
        _t = data_raw.info['sfreq']
        data_raw.resample(RESAMPLE_FREQUENCY)
        logging.info(f"Succesfully resampled RAW data (was {_t} Hz, is now {data_raw.info['sfreq']} Hz)")
    elif RESAMPLE_AT.lower() == 'epoch':
        pass
    elif RESAMPLE_AT.lower() == 'do_not_resample':
        pass
    else:
        s = f"Invalid resampling method parameter ({RESAMPLE_AT})"
        logging.critical(s)
        raise ValueError(s)

    # get stimulus events
    try:
        if len(data_raw.annotations) > 0:
            events = mne.events_from_annotations(data_raw)
        else:
            events = mne.find_events(data_raw)
    except:
        s = "Unable to segment data (could not find event markers/annotations)"
        logging.critical(s)
        raise Exception(s)

    logging.info(f"Found event IDs: {events[1]}")

    # constructing hierarchical condition/stimulus event_ids 
    # (looks more complicated than it is, in order to get it more user-friendly)
    event_id = {}
    for event_key, event_val in events[1].items():
        for cond_key, cond_val in CONDITION_STIMULI.items():
            # if user chose the marker value as indicator
            if event_val in cond_val:
                event_id[cond_key + '/' + str(event_val)] = event_val
            # if user chose the marker name as indicator
            elif event_key in cond_val:
                event_id[cond_key + '/' + event_key.replace('/', '_')] = event_val

    data_epochs = mne.Epochs(data_raw, events[0],
                             event_id=event_id,
                             tmin = T_MIN,
                             tmax = T_MAX,
                             preload=True
                             )

    # resample if selected to do so here
    if RESAMPLE_AT.lower() == 'epoch':
        data_epochs.load_data()
        _t = data_epochs.info['sfreq']
        data_epochs.resample(RESAMPLE_FREQUENCY)
        logging.info(f"Succesfully resampled EPOCH data (was {_t} Hz, is now {data_epochs.info['sfreq']} Hz)")

    # ERP plot as reference check for successful data import/conversion
    data_epochs.average(picks=ERP_CHECK_ELECTRODES if len(ERP_CHECK_ELECTRODES) > 0 else None) \
               .plot() \
               .savefig(SAVE_DIR_ERP / f.with_suffix('.png').name, dpi=300)



    ####################
    # ----- MVPA ----- #
    ####################

    pipeline = make_pipeline(StandardScaler(),
                             DECODER_MODEL)

    generalizer = GeneralizingEstimator(pipeline,
                                        scoring=SCORING,
                                        n_jobs=N_JOBS,
                                        verbose=logging.root.level)

    # TODO: balanced no. of trials across conditions

    # TODO: 1 event seems to get dropped here
    data_matrix = data_epochs.get_data(picks='data') # pick only good EEG channels

    # produce labels based on user-indicated condition/marker mapping
    labels = np.empty(shape=(len(data_epochs.events[:,-1])), dtype=object)
    for cond, markers in CONDITION_STIMULI.items():
        for marker in markers:
            labels[data_epochs.events[:,-1] == marker] = cond

    # run fitting/cross validation
    logging.info("Performing fitting/cross-validation")
    scores = cross_val_multiscore(generalizer,
                                  data_matrix,
                                  labels,
                                  cv=CROSS_VAL_FOLDS,
                                  n_jobs=N_JOBS
                                  )

    # store results in temp folder, to be imported later
    with open(SAVE_DIR_TMP / f.with_suffix('.npy').name, 'wb') as tmp:
        # scores.tofile(tmp)
        np.save(tmp, scores)
        logging.info(f"Wrote cross validation scores to {tmp.name}")

    # take mean score over folds
    mean_scores = scores.mean(0)

    # plot the scoring matrix of this subject
    fig, ax = GeneralizationScoreMatrix(scores_matrix = mean_scores, 
                                        times_limits = data_epochs.times[[0, -1]],
                                        score_method = 'accuracy')
    
    fig.savefig(SAVE_DIR_GEN_MATRIX / f.with_suffix('.png').name, dpi=450)

#%%

# read scores from tmp files, preserving file names in dict keys
tmp_results = {}
for f in SAVE_DIR_TMP.glob('*.npy'):
    print(f"Unpacking {f.name}")
    with open(f, 'rb') as tmp:
        tmp_results[f.stem] = np.load(tmp)

# aggregate into numpy array, with axes (subject, fold, n_times, n_times)
agg_results = np.empty(shape=(len(tmp_results, *tmp_results[0].shape)))    
for i, (key, val) in enumerate(tmp_results.items()):
    agg_results[i] = val