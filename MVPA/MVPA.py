import logging
logger = logging.getLogger(__name__)

import pickle
import numpy as np

import mne
from mne.decoding import GeneralizingEstimator, cross_val_multiscore 

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# prevent commandline output from getting cluttered by repetitive warnings
from warnings import filterwarnings
from sklearn.exceptions import ConvergenceWarning
filterwarnings("once", category=ConvergenceWarning)

from .utils import config_prep, SubjectFiles
CONFIG = config_prep()

class MVPA_manager:
    def __init__(self, data_path=CONFIG['PATHS']['DATA'], save_path=CONFIG['PATHS']['SAVE']):
        if data_path.is_file():
            self.subjects = [SubjectFiles(data_path, save_path)]
        elif data_path.is_dir():
            self.subjects = [SubjectFiles(f, save_path) for f in data_path.glob('*.vhdr')]

    #############################
    # ----- PREPROCESSING ----- #
    #############################

    def _preprocess(self, subject: SubjectFiles):
        data_raw = mne.io.read_raw_brainvision(subject.raw, **CONFIG['MNE']['IMPORT_ARGS'])

        data_raw.load_data()
        data_raw.set_eeg_reference(CONFIG['MNE']['ELECTRODES']['REFERENCE'])
        data_raw.info['bads'] = CONFIG['MNE']['ELECTRODES']['BAD']

        # resample if selected to do so here, or check if another valid option was given.
        if CONFIG['MNE']['RESAMPLE']['METHOD'].lower() == 'raw':
            t_ = data_raw.info['sfreq']
            data_raw.resample(CONFIG['MNE']['RESAMPLE']['FREQUENCY'])
            logger.debug(f"Succesfully resampled RAW data (was {t_} Hz, is now {data_raw.info['sfreq']} Hz)")
        elif CONFIG['MNE']['RESAMPLE']['METHOD'].lower() == 'epoch':
            pass
        elif CONFIG['MNE']['RESAMPLE']['METHOD'].lower() == 'do_not_resample':
            pass
        else:
            s = f"Invalid resampling method parameter ({CONFIG['MNE']['RESAMPLE']['METHOD']})"
            logger.critical(s)
            raise ValueError(s)

        # get stimulus events
        try:
            if len(data_raw.annotations) > 0:
                events = mne.events_from_annotations(data_raw)
            else:
                events = mne.find_events(data_raw)
        except:
            s = "Unable to segment data (could not find event markers/annotations)"
            logger.critical(s)
            raise Exception(s)

        logger.debug(f"Found event IDs: {events[1]}")

        # constructing hierarchical condition/stimulus event_ids 
        # (looks more complicated than it is, in order to get it more user-friendly)
        event_id = {}
        for event_key, event_val in events[1].items():
            for cond_key, cond_val in CONFIG['CONDITION_STIMULI'].items():
                # if user chose the marker value as indicator
                if event_val in cond_val:
                    event_id[cond_key + '/' + str(event_val)] = event_val
                # if user chose the marker name as indicator
                elif event_key in cond_val:
                    event_id[cond_key + '/' + event_key.replace('/', '_')] = event_val

        data_epochs = mne.Epochs(data_raw, events[0],
                                 event_id=event_id,
                                 tmin = CONFIG['MNE']['T_MIN'],
                                 tmax = CONFIG['MNE']['T_MAX'],
                                 preload=True
                                 )

        # resample if selected to do so here
        if CONFIG['MNE']['RESAMPLE']['METHOD'].lower() == 'epoch':
            data_epochs.load_data()
            t_ = data_epochs.info['sfreq']
            data_epochs.resample(CONFIG['MNE']['RESAMPLE']['FREQUENCY'])
            logger.debug(f"Succesfully resampled EPOCH data (was {t_} Hz, is now {data_epochs.info['sfreq']} Hz)")

        # ERP plot as reference check for successful data import/conversion
        data_epochs.average(picks=CONFIG['MNE']['ELECTRODES']['ERP_CHECK'] if len(CONFIG['MNE']['ELECTRODES']['ERP_CHECK']) > 0 else None) \
                .plot() \
                .savefig(subject.png('ERP'), dpi=300)
        
        # store files in tmp folder
        logger.debug(f"Storing processed data at {subject.epoch}")
        data_epochs.save(subject.epoch)

    def preprocess_all(self, rm_existing=True):
        # remove existing epoch files from tmp directory
        if rm_existing:
            # remove existing epoch files from directory
            for f_ in CONFIG['PATHS']['TMP'].glob('*-epo.fif'): # FOR SAFETY, ONLY TOP-LEVEL FILE GLOB
                if f_.is_file(): f_.unlink()
                else: raise Exception(f"Please do not alter contents of {CONFIG['PATHS']['TMP']}")

        # run preprocessing for all raw files
        for i, subject in enumerate(self.subjects, start=1):
            logger.info(f"Preprocessing file {i}/{len(self.subjects)} : {subject.raw}")
            self._preprocess(subject)

    ########################
    # ----- GAT MVPA ----- #
    ########################

    def _gat(self, subject: SubjectFiles):
        logging.debug(f"Loading file {subject.epoch}")
        data_epochs = mne.read_epochs(subject.epoch)

        from sklearn.linear_model import LinearRegression, RidgeClassifier, LogisticRegression
        from sklearn.svm import SVC

        # available methods
        models = dict(OLS  = LinearRegression(),                                                     # Ordinary Least Squares Regression
                    LogRes = LogisticRegression(solver="liblinear", **CONFIG['DECODING']['MODEL_ARGS']),                 # Logistic Regression
                    Ridge  = RidgeClassifier(**CONFIG['DECODING']['MODEL_ARGS']),                                        # Ridge Regression / Tikhonov regularisation
                    SVC    = SVC(kernel='linear', random_state=CONFIG['DECODING']['RAND_STATE'], **CONFIG['DECODING']['MODEL_ARGS']),          # Linear Support Vector Machine
                    SVM    = SVC(kernel='rbf', random_state=CONFIG['DECODING']['RAND_STATE'], **CONFIG['DECODING']['MODEL_ARGS']),             # Non-linear Support Vector Machine
                    )
        
        if CONFIG['DECODING']['MODEL'] not in [m for m in models.keys()]:
            raise ValueError(f"Unrecognised decoder model (got {CONFIG['DECODING']['MODEL']}, expected one of {[m for m in models.keys()]}")

        pipeline = make_pipeline(StandardScaler(),
                                 models[CONFIG['DECODING']['MODEL']])

        generalizer = GeneralizingEstimator(pipeline,
                                            scoring=CONFIG['DECODING']['SCORING'],
                                            n_jobs=CONFIG['DECODING']['N_JOBS'],
                                            verbose=logger.level)

        # TODO: 1 event seems to get dropped here
        data_matrix = data_epochs.get_data(picks='data') # pick only good EEG channels

        # produce labels based on user-indicated condition/marker mapping
        labels = np.empty(shape=(len(data_epochs.events[:,-1])), dtype=object)
        for cond, markers in CONFIG['CONDITION_STIMULI'].items():
            for marker in markers:
                labels[data_epochs.events[:,-1] == marker] = cond

        # run fitting/cross validation
        logger.info("Performing fitting and cross-validation")
        scores = cross_val_multiscore(generalizer,
                                      data_matrix,
                                      labels,
                                      cv=CONFIG['DECODING']['CROSS_VAL_FOLDS'],
                                      n_jobs=CONFIG['DECODING']['N_JOBS']
                                      )

        # store results in temp folder, to be imported later
        with open(subject.gat, 'wb') as tmp:
            np.save(tmp, scores)
            logger.debug(f"Wrote cross validation scores to {tmp.name}")
    
    def aggregate_gat_results(self, keep_names=False):
        # read scores from tmp files
        results_dict = {}
        for subject in self.subjects:
            logger.debug(f"Loading {subject.gat}")
            with open(subject.gat, 'rb') as f:
                results_dict[subject.stem] = np.load(f)

        # store results for later use
        if keep_names:
            f = CONFIG['PATHS']['SAVE'] / 'GAT_results.pickle'
            with open(f, 'wb') as f_:
                pickle.dump(results_dict, f_)
        else:
            # aggregate all results into numpy array, with axes (subject, fold, n_times, n_times)
            results_mat = np.empty(shape=(len(results_dict), *next(iter(results_dict.values())).shape))
            for i, res in enumerate(results_dict.values()):
                results_mat[i] = res
            
            f = CONFIG['PATHS']['SAVE'] / 'GAT_results.npy'
            with open(f, 'wb') as f_:
                np.save(f_, results_mat)

        logger.info(f"Stored aggregated GAT results at {f}")

    def run_all_gat(self, rm_existing=True, aggregate_results=True):
        # remove existing GAT files from tmp directory
        if rm_existing:
            # remove existing GAT files from directory
            for f_ in CONFIG['PATHS']['TMP'].glob('*-gat.npy'): # FOR SAFETY, ONLY TOP-LEVEL FILE GLOB
                if f_.is_file(): f_.unlink()
                else: raise Exception(f"Please do not alter contents of {CONFIG['PATHS']['TMP']}")
        
        # run Generalized Across Time (GAT) Decoding for each subject individually
        for i, subject in enumerate(self.subjects, start=1):
            logger.info(f"Fitting GAT decoding for subject {i}/{len(self.subjects)}")
            self._gat(subject)
        
        logger.debug("Consolidating GAT results")
        self.aggregate_gat_results()

