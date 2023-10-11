#%%
from sklearn.linear_model import LinearRegression, RidgeClassifier, LogisticRegression
from sklearn.svm import SVC
import pathlib

#################################
# ----- SCRIPT PARAMETERS ----- #
#################################

# Data-related
DATA_FILE = r"C:\Users\Jakob\Documents\repositories\SpeechAdaptation\data\practice_data\EE_EU_adaptation_ERP_1_Ambi_after_EE_34567_BC.vhdr"
# DATA_FILE = "path/to/data/file.vhdr"                   # Expects BrainVision format header file (.vhdr extension)
SAVE_DIRECTORY = r"C:\Users\Jakob\Documents\repositories\SpeechAdaptation\results"
# SAVE_DIRECTORY = "path/to/save/directory"              # Directory name, NOT a file 

# Data import args
# see https://mne.tools/stable/generated/mne.io.read_raw_brainvision.html#mne.io.read_raw_brainvision
DATA_ARGS = {'eog' :    ('HEOGL', 'HEOGR', 'VEOGb'),
             'misc' :   'auto',
             'scale' :  1.0}

TARGET_STIM_ONSET_MARKER_NAME = 'Time 0'               # Name or value of marker indicating target stimulus onset

REFERENCE_ELECTRODES = ['M1', 'M2']                    # List of reference electrode names
BAD_ELECTRODES = []                                    # List electrode names that should be ignored
# BAD_SUBJECTS = []                                      # List subject IDs that should be ignored

T_MIN = -0.2                                           # Starting time of trials in seconds (if baseline segment should be included, starting time can be negative)
T_MAX = 1.0                                            # Ending time of trials in seconds

# Model selection
# See https://scikit-learn.org/stable/modules/classes.html#module-sklearn.linear_model
# or  https://scikit-learn.org/stable/modules/classes.html#module-sklearn.svm

GENERAL_ARGS = {"class_weight":'balanced'}

RAND_STATE = 1

MODELS = {"OLS":    LinearRegression(),                                       # Ordinary Least Squares Regression
          "LogRes": LogisticRegression(**GENERAL_ARGS),                                     # Logistic Regression
          "Ridge":  RidgeClassifier(**GENERAL_ARGS),                                        # Ridge Regression / Tikhonov regularisation
          "SVC":    SVC(kernel='linear', random_state=RAND_STATE, **GENERAL_ARGS),          # Linear Support Vector Machine
          "SVM":    SVC(kernel='rbf', random_state=RAND_STATE, **GENERAL_ARGS),             # Non-linear Support Vector Machine
          }

DECODER_MODEL = MODELS['LogRes']            # choose desired model from the above list

SCORING = 'roc_auc'                         # check scikit-learn docs (links above) for available metrics

N_JOBS = -1                                 # if -1, equal to number of CPU cores



############################
# ----- DATA IMPORTS ----- #
############################

import mne
mne.set_log_level('ERROR')

DATA_PATH = pathlib.Path(DATA_FILE)

data_raw = mne.io.read_raw_brainvision(DATA_PATH, 
                                       **DATA_ARGS)

data_raw.load_data()
data_raw.set_eeg_reference(REFERENCE_ELECTRODES)
data_raw.info['bads'] = BAD_ELECTRODES

GOOD_CHANNELS = list(set(data_raw.ch_names) - set(REFERENCE_ELECTRODES) - set(BAD_ELECTRODES))

_allowed_channel_types = ['ecg', 'eeg', 'emg', 'eog', 'exci', 
                          'ias', 'misc', 'resp', 'seeg', 'dbs', 
                          'stim', 'syst', 'ecog', 'hbo', 'hbr', 
                          'fnirs_cw_amplitude', 'fnirs_fd_ac_amplitude', 
                          'fnirs_fd_phase', 'fnirs_od', 'eyetrack_pos', 
                          'eyetrack_pupil', 'temperature', 'gsr']

try:
    if len(data_raw.annotations) > 0:
        events = mne.events_from_annotations(data_raw)
    else:
        events = mne.find_events(data_raw)
except:
    raise Exception('Unable to segment data (could not find event annotations)')

from pprint import pprint
print("Found event IDs:")
pprint(events[1])

marker_present = False
if isinstance(TARGET_STIM_ONSET_MARKER_NAME, str):
    for key in events[1].keys():
        if TARGET_STIM_ONSET_MARKER_NAME in key:
            marker_present = True
elif isinstance(TARGET_STIM_ONSET_MARKER_NAME, int):
    for val in events[1].values():
        if TARGET_STIM_ONSET_MARKER_NAME == val:
            marker_present = True
else:
    raise ValueError(f"Unrecognised marker value ({TARGET_STIM_ONSET_MARKER_NAME})")

try:
    assert marker_present
except:
    raise ValueError(f"Could not find target stimulus marker in event IDs ({TARGET_STIM_ONSET_MARKER_NAME})")

#%%
data_epochs = mne.Epochs(data_raw, events[0],
                         event_id=[val for key, val in events[1].items() if 'Stimulus' in key],
                         tmin = T_MIN,
                         tmax = T_MAX)

 #%%



#############################
# ----- MVPA PIPELINE ----- #
#############################

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# from sklearn.cross_validation import StratifiedKFold

#from sklearn.model_selection import cross_val_multiscore # updated
from mne.decoding import SlidingEstimator, cross_val_multiscore, Scaler, Vectorizer
from mne.decoding import GeneralizingEstimator


pipeline = make_pipeline(StandardScaler(),
                         DECODER_MODEL)

generalizer = GeneralizingEstimator(pipeline,
                                    scoring=SCORING,
                                    n_jobs=N_JOBS)
