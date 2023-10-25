import pathlib

from sklearn.linear_model import LinearRegression, RidgeClassifier, LogisticRegression
from sklearn.svm import SVC


######################################
# ----- PARAMETERS DEFINITIONS ----- #
######################################

# Data-related
DATA_PATH = r"C:\Users\Jakob\Documents\repositories\SpeechAdaptation\data\Export from BVA"  # Location of data file(s), may be a single file or a directory containing multiple files.
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




#######################################################
# ----- IO DEFINITIONS - DO NOT EDIT CODE BELOW ----- #
#######################################################

DATA_PATH = pathlib.Path(DATA_PATH)
SAVE_DIR = pathlib.Path(SAVE_DIRECTORY)

if not SAVE_DIR.exists():
    raise OSError('Save directory does not exist')

# create subdirectories in the results dir
_subdirs = ["ERP",
            "Temporal Generalization Matrix"]

SAVE_DIRS_DICT = {}

for d in _subdirs:
    SAVE_DIRS_DICT[d] = SAVE_DIR / d
    if not SAVE_DIRS_DICT[d].exists():
        SAVE_DIRS_DICT[d].mkdir()


# TODO: rewrite this for graceful exception handling (use intermediate files only in case of error)
# .tmp results directory will hold intermediate results
SAVE_DIR_TMP = SAVE_DIR / ".tmp"
if not SAVE_DIR_TMP.exists():
    SAVE_DIR_TMP.mkdir(mode=0o700)
