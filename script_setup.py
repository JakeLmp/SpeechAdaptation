from sklearn.linear_model import LinearRegression, RidgeClassifier, LogisticRegression
from sklearn.svm import SVC
import pathlib

#################################
# ----- SCRIPT PARAMETERS ----- #
#################################

# Data-related
DATA_FILE = r"C:\Users\Jakob\Documents\repositories\SpeechAdaptation\data\data\dbc_data.csv"
# DATA_FILE = "path/to/data/file.csv"                    # Expected format provided elsewhere
SAVE_DIRECTORY = "path/to/save/directory"              # Directory name, NOT a file 

BAD_ELECTRODES = ['Electrode1', 'Electrode2']          # List electrode names that should be ignored
BAD_SUBJECTS = ['ExampleID1', 'ExampleID2']            # List subject IDs that should be ignored

# Model selection
# See https://scikit-learn.org/stable/modules/classes.html#module-sklearn.linear_model
# or  https://scikit-learn.org/stable/modules/classes.html#module-sklearn.svm

GENERAL_ARGS = {"class_weight":'balanced'}

RAND_STATE = 1

MODELS = {"OLS":    LinearRegression(**GENERAL_ARGS),                                       # Ordinary Least Squares Regression
          "LogRes": LogisticRegression(**GENERAL_ARGS),                                     # Logistic Regression
          "Ridge":  RidgeClassifier(**GENERAL_ARGS),                                        # Ridge Regression / Tikhonov regularisation
          "SVC":    SVC(kernel='linear', random_state=RAND_STATE, **GENERAL_ARGS),          # Linear Support Vector Machine
          "SVM":    SVC(kernel='rbf', random_state=RAND_STATE, **GENERAL_ARGS),             # Non-linear Support Vector Machine
          }

DECODER_MODEL = MODELS['LogRes']            # choose desired model from the above list

SCORING = 'roc_auc'                         # check scikit-learn docs (links above) for available metrics


############################
# ----- DATA IMPORTS ----- #
############################

import pandas as pd
import mne
mne.set_log_level('ERROR')

DATA_PATH = pathlib.Path(DATA_FILE)

df = pd.read_csv(DATA_PATH)

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
                                    scoring=SCORING)

