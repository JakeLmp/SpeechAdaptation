import logging
logger = logging.getLogger('MVPA')

import pickle
import numpy as np
import matplotlib.pyplot as plt

import mne
import sklearn.pipeline, sklearn.preprocessing

from tqdm import tqdm

# prevent commandline output from getting cluttered by repetitive warnings
from warnings import filterwarnings
from sklearn.exceptions import ConvergenceWarning
filterwarnings("once", category=ConvergenceWarning)

import MVPA.utils, MVPA.stat_utils
CONFIG = MVPA.utils.config_prep()

# neighbour definition for searchlight (64 electrodes)
montage_1020_64 = np.array([
    ['--', '--', '--', '--', '--', '--', 'Nz', '--', '--', '--', '--',  '--','--'],
    ['--', '--', '--', '--','Fp1', '--','Fpz', '--','Fp2', '--', '--',  '--','--'],
    ['--', '--','AF7', '--','AF3', '--','AFz', '--','AF4', '--','AF8',  '--','--'],
    ['--', 'F9', 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8', 'F10','--'],
    ['--','FT9','FT7','FC5','FC3','FC1','FCz','FC2','FC4','FC6','FT8','FT10','--'],
    ['A1', 'T9', 'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8', 'T10','A2'],
    ['--','TP9','TP7','CP5','CP3','CP1','CPz','CP2','CP4','CP6','TP8','TP10','--'],
    ['--', 'P9', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'P10','--'],
    ['--', '--','PO7', '--','PO3', '--','POz', '--','PO4', '--','PO8',  '--','--'],
    ['--', '--', '--', '--', 'O1', '--', 'Oz', '--', 'O2', '--', '--',  '--','--'],
    ['--', '--', '--', '--', '--', '--', 'Iz', '--', '--', '--', '--',  '--','--']
])
# replace '--' with np.nan
np.place(montage_1020_64, montage_1020_64=='--', np.nan)

def get_neighbours(idx: int, jdx: int, radius:int=1) -> np.ndarray:
    """
    Get neighbour electrodes from montage matrix.

    Args:
        idx (int): row index
        jdx (int): column index
        radius (int, optional): radius of square neighbourhood in indices. Defaults to 1.

    Returns:
        np.ndarray: submatrix of size (radius*2+1, radius*2+1), 
            containing electrode at (idx, jdx) and its neighbours
    """
    imax, jmax = montage_1020_64.shape
    
    nbs = [[montage_1020_64[i][j] if  i >= 0 and i < imax and j >= 0 and j < jmax else np.nan
                for j in range(jdx-radius, jdx+1+radius)]
                    for i in range(idx-radius, idx+1+radius)]

    return np.array(nbs)


class DecodingManager:
    def __init__(self, data_path=CONFIG['PATHS']['DATA'], 
                 save_path=CONFIG['PATHS']['SAVE']):
        if data_path.is_file():
            self.subjects = [MVPA.utils.SubjectFiles(data_path, save_path)]
        elif data_path.is_dir():
            self.subjects = [MVPA.utils.SubjectFiles(f, save_path) for f in data_path.glob('*.vhdr')]

    #############################
    # ----- PREPROCESSING ----- #
    #############################

    def _preprocess(self, subject: MVPA.utils.SubjectFiles):
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
        
        # store files in tmp folder
        logger.debug(f"Storing processed data at {subject.epoch}")
        data_epochs.save(subject.epoch)
        
        # store subject evoked as well (all conditions bunched together)
        evoked = data_epochs.average(picks=CONFIG['MNE']['ELECTRODES']['ERP_CHECK'] if len(CONFIG['MNE']['ELECTRODES']['ERP_CHECK']) > 0 else None)
        evoked.save(subject.evoked)

        # ERP plot as reference check for successful data import/conversion
        evoked.plot() \
              .savefig(subject.png('ERP'), dpi=300)
        plt.close()

    def _grand_avg(self,):
        # read subject evokeds
        evokeds = []
        for subject in self.subjects:
            evokeds.extend(mne.read_evokeds(subject.evoked))
        
        # calculate and return grand average
        return mne.grand_average(evokeds)

    def preprocess_all(self, rm_existing=True):
        # remove existing epoch files from tmp directory
        if rm_existing:
            # remove existing epoch files from directory
            for f_ in CONFIG['PATHS']['TMP'].glob('*-epo.fif'): # FOR SAFETY, ONLY TOP-LEVEL FILE GLOB
                if f_.is_file(): f_.unlink()
                else: raise Exception(f"Please do not alter contents of {CONFIG['PATHS']['TMP']}")

            # remove existing evoked files from directory
            for f_ in CONFIG['PATHS']['TMP'].glob('*_processed-ave.fif'): # FOR SAFETY, ONLY TOP-LEVEL FILE GLOB
                if f_.is_file(): f_.unlink()
                else: raise Exception(f"Please do not alter contents of {CONFIG['PATHS']['TMP']}")

        # run preprocessing for all raw files
        for i, subject in enumerate(self.subjects, start=1):
            logger.info(f"Preprocessing file {i}/{len(self.subjects)} : {subject.raw}")
            self._preprocess(subject)

        # store info object of first subject for easy later use
        mne.read_epochs(self.subjects[0].epoch) \
           .info.save(CONFIG['PATHS']['INFO_OBJ'])
        
        # calculate, plot and store grand average (all conditions bunched together)
        grand_average = self._grand_avg()

        f = CONFIG['PATHS']['PLOT'] / 'grand_avg.png'
        grand_average.plot(picks=CONFIG['MNE']['ELECTRODES']['ERP_CHECK'] if len(CONFIG['MNE']['ELECTRODES']['ERP_CHECK']) > 0 else None) \
                     .savefig(f, dpi=300)
        plt.close()
        logger.debug(f"Wrote grand avg plot to {f}")

        f = CONFIG['PATHS']['RESULTS']['GRAND_AVG']
        grand_average.save(f, overwrite=True)

    def spoof_single_subject(self, rm_existing=True):
        """
        Combine all subject data into single file, and continue analysis
        as if there is just one subject. 
        
        WARNING:
        This greatly increases memory demands, since all data is now loaded at once.
        """
        
        # create list of epoch objects without loading data
        epochs_list = [mne.read_epochs(subject.epoch, preload=False) for subject in self.subjects]

        # combine epochs (this loads data)
        subjects_combined = mne.concatenate_epochs(epochs_list, verbose=logger.level)

        # save object to file, then delete the object
        f = CONFIG['PATHS']['TMP'] / 'subjects_combined-epo.fif'
        
        from copy import deepcopy
        spoof_subject = deepcopy(self.subjects[0])
        spoof_subject.path = CONFIG['PATHS']['TMP'] / 'subjects_combined.fakesuffix'

        subjects_combined.save(spoof_subject.epoch, overwrite=rm_existing, verbose=True)
        del subjects_combined
        
        # remove old epoch files
        # for subject in self.subjects:
        #     subject.epoch.unlink()

        # replace subject list with spoofed subject
        self.subjects = [spoof_subject]

    ######################
    # ----- MODELS ----- #
    ######################

    def _get_model_object(self, model):
        from sklearn.linear_model import LinearRegression, RidgeClassifier, LogisticRegression
        from sklearn.svm import SVC

        # available methods
        models = dict(OLS  = LinearRegression(),                                                                                            # Ordinary Least Squares Regression
                    LogRes = LogisticRegression(solver="liblinear", **CONFIG['DECODING']['MODEL_ARGS']),                                    # Logistic Regression
                    Ridge  = RidgeClassifier(**CONFIG['DECODING']['MODEL_ARGS']),                                                           # Ridge Regression / Tikhonov regularisation
                    SVC    = SVC(kernel='linear', random_state=CONFIG['DECODING']['RAND_STATE'], **CONFIG['DECODING']['MODEL_ARGS']),       # Linear Support Vector Machine
                    SVM    = SVC(kernel='rbf', random_state=CONFIG['DECODING']['RAND_STATE'], **CONFIG['DECODING']['MODEL_ARGS']),          # Non-linear Support Vector Machine
                    )
        
        if model not in [m for m in models.keys()]:
            raise ValueError(f"Unrecognised decoder model (got {model}, expected one of {[m for m in models.keys()]})")
        
        return models[model]

    ########################
    # ----- GAT MVPA ----- #
    ########################

    def _gat(self, subject: MVPA.utils.SubjectFiles, 
             model=CONFIG['DECODING']['MODEL'],
             cv_folds=CONFIG['DECODING']['CROSS_VAL_FOLDS'],
             scoring=CONFIG['DECODING']['SCORING'],
             n_jobs=CONFIG['DECODING']['N_JOBS'],
             ):
        logger.debug(f"Loading file {subject.epoch}")
        data_epochs = mne.read_epochs(subject.epoch)

        pipeline = sklearn.pipeline.make_pipeline(sklearn.preprocessing.StandardScaler(),
                                                  self._get_model_object(model)
                                                  )

        generalizer = mne.decoding.GeneralizingEstimator(pipeline,
                                                         scoring=scoring,
                                                         n_jobs=n_jobs,
                                                         verbose=logger.level
                                                         )

        # TODO: 1 event seems to get dropped here
        data_matrix = data_epochs.get_data(picks='data') # pick only good EEG channels

        # produce labels based on user-indicated condition/marker mapping
        labels = np.empty(shape=(len(data_epochs.events[:,-1])), dtype=object)
        for cond, markers in CONFIG['CONDITION_STIMULI'].items():
            for marker in markers:
                labels[data_epochs.events[:,-1] == marker] = cond

        # labels should be encoded to facilitate all models
        labels = sklearn.preprocessing.LabelEncoder().fit_transform(labels)

        # run fitting/cross validation
        logger.info("Performing fitting and cross-validation")
        scores = mne.decoding.cross_val_multiscore(generalizer,
                                                   data_matrix,
                                                   labels,
                                                   cv=cv_folds,
                                                   n_jobs=n_jobs
                                                   )

        # store results in temp folder, to be imported later
        with open(subject.gat, 'wb') as tmp:
            np.save(tmp, scores)
            logger.debug(f"Wrote cross validation scores to {tmp.name}")
    
    def _aggregate_gat_results(self, keep_names=False):
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
            
            f = CONFIG['PATHS']['RESULTS']['GAT_RESULTS']
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
        self._aggregate_gat_results()

        if len(self.subjects) > 1:
            # p-value calculation with FDR correction
            logger.info("Calculating p-values")

            with open(CONFIG['PATHS']['RESULTS']['GAT_RESULTS'], 'rb') as f:
                scores = np.load(f)
        
            p_values = MVPA.stat_utils.get_p_scores(scores.mean(1), # use average of folds
                                                    chance= 1/len(CONFIG['CONDITION_STIMULI']), # 1/no_of_conditions
                                                    tfce=False) # use FDR instead of TFCE

            # store p-values
            f = CONFIG['PATHS']['RESULTS']['GAT_PVALUES']
            with open(f, 'wb') as tmp:
                np.save(tmp, p_values)
                logger.debug(f"Wrote GAT p-values to {tmp.name}")

    #############################
    # ----- TEMPORAL MVPA ----- #
    #############################
    
    def _temporal_decoding(self, subject,
                           model=CONFIG['DECODING']['MODEL'],
                           cv_folds=CONFIG['DECODING']['CROSS_VAL_FOLDS'],
                           scoring=CONFIG['DECODING']['SCORING'],
                           n_jobs=CONFIG['DECODING']['N_JOBS']
                           ):
        """
        See https://mne.tools/stable/auto_examples/decoding/linear_model_patterns.html
        and also https://mne.tools/stable/auto_tutorials/machine-learning/50_decoding.html#temporal-decoding

        Use channels as features, time points as samples.
        """
        logger.debug(f"Loading file {subject.epoch}")
        data_epochs = mne.read_epochs(subject.epoch)

        pipeline = sklearn.pipeline.make_pipeline(
                                                #   mne.decoding.Vectorizer(),
                                                  sklearn.preprocessing.StandardScaler(),
                                                  mne.decoding.LinearModel(self._get_model_object(model))
                                                  )

        estimator = mne.decoding.SlidingEstimator(pipeline,
                                                  n_jobs=n_jobs,
                                                  scoring=scoring,
                                                  verbose=logger.level
                                                  )

        # TODO: 1 event seems to get dropped here
        data_matrix = data_epochs.get_data(picks='data') # pick only good EEG channels

        # produce labels based on user-indicated condition/marker mapping
        labels = np.empty(shape=(len(data_epochs.events[:,-1])), dtype=object)
        for cond, markers in CONFIG['CONDITION_STIMULI'].items():
            for marker in markers:
                labels[data_epochs.events[:,-1] == marker] = cond

        # labels should be encoded to facilitate all models
        labels = sklearn.preprocessing.LabelEncoder().fit_transform(labels)

        logger.info("Performing fitting")
        
        # --- First obtaining the scores
        scores = mne.decoding.cross_val_multiscore(estimator,
                                                   data_matrix,
                                                   labels,
                                                   cv=cv_folds,
                                                   n_jobs=n_jobs
                                                   )

        # store results in temp folder, to be imported later
        with open(subject.temporal_scores, 'wb') as tmp:
            np.save(tmp, scores)
            logger.debug(f"Wrote cross validation scores to {tmp.name}")

        # --- Now obtaining filters and patterns
        estimator.fit(data_matrix, labels)

        # inverse-transform results for storage
        coef = mne.decoding.get_coef(estimator, 'filters_', inverse_transform=True)
        evoked = mne.EvokedArray(coef, data_epochs.pick('eeg', exclude=['bads', 'eog']).info, 
                                 tmin=data_epochs.tmin)
        logger.debug(f"Storing filter evokeds at {subject.temporal_filter}")
        evoked.save(subject.temporal_filter)
        
        coef = mne.decoding.get_coef(estimator, 'patterns_', inverse_transform=True)
        evoked = mne.EvokedArray(coef, data_epochs.pick('eeg', exclude=['bads', 'eog']).info, 
                                 tmin=data_epochs.tmin)
        logger.debug(f"Storing pattern evokeds at {subject.temporal_pattern}")
        evoked.save(subject.temporal_pattern)

    def _aggregate_temporal_scores(self):
        # get first subject's scores as template for array size
        with open(self.subjects[0].temporal_scores, 'rb') as tmp:
            scores = np.load(tmp)

        # insert dimension for multiple subjects
        scores.resize((len(self.subjects), *scores.shape))

        # read remaining subject scores
        for i, subject in enumerate(self.subjects[1:], start=1):
            with open(subject.temporal_scores, 'rb') as tmp:
                scores[i] = np.load(tmp)

        f = CONFIG['PATHS']['RESULTS']['TEMPORAL_SCORES']
        with open(f, 'wb') as tmp:
            np.save(tmp, scores)
            logger.debug(f"Wrote aggregated temporal scores to {tmp.name}")

    def _temporal_grand_avg(self,):
        # read subject evokeds (only patterns are directly interpretable)
        evokeds = []
        for subject in self.subjects:
            evokeds.extend(mne.read_evokeds(subject.temporal_pattern))
        
        # calculate and store grand average
        grand_average = mne.grand_average(evokeds)

        f = CONFIG['PATHS']['RESULTS']['TEMPORAL_GRAND_AVG']
        grand_average.save(f, overwrite=True)
        logger.debug(f"Wrote temporal pattern grand avg to {f}")

    def run_all_temporal_decoding(self, rm_existing=True):
        # remove existing -temporal_* files from tmp directory
        if rm_existing:
            # remove existing scoring files from directory
            for f_ in CONFIG['PATHS']['TMP'].glob('*-temporal_scores.npy'):# FOR SAFETY, ONLY TOP-LEVEL FILE GLOB
                if f_.is_file(): f_.unlink()
                else: raise Exception(f"Please do not alter contents of {CONFIG['PATHS']['TMP']}")

            # remove existing pattern/filter files from directory
            for f_ in CONFIG['PATHS']['TMP'].glob('*-temporal_*-ave.fif'): # FOR SAFETY, ONLY TOP-LEVEL FILE GLOB
                if f_.is_file(): f_.unlink()
                else: raise Exception(f"Please do not alter contents of {CONFIG['PATHS']['TMP']}")

        # run temporal decoding for each subject individually
        for i, subject in enumerate(self.subjects, start=1):
            logger.info(f"Fitting temporal decoding for subject {i}/{len(self.subjects)}")
            self._temporal_decoding(subject)
        
        # aggregate and save results
        logger.debug("Consolidating results")
        self._aggregate_temporal_scores()
        self._temporal_grand_avg()

    ############################
    # ----- CHANNEL MVPA ----- #
    ############################

    def _channel_decoding(self, subject,
                          model=CONFIG['DECODING']['MODEL'],
                          cv_folds=CONFIG['DECODING']['CROSS_VAL_FOLDS'],
                          scoring=CONFIG['DECODING']['SCORING'],
                          n_jobs=CONFIG['DECODING']['N_JOBS']
                          ):
        """
        Same as temporal decoding, but using time points as features, channels as samples.
        Difference is, that this is done with one channel at a time. 
        """
        logger.debug(f"Loading file {subject.epoch}")
        data_epochs = mne.read_epochs(subject.epoch)

        # produce labels based on user-indicated condition/marker mapping
        labels = np.empty(shape=(len(data_epochs.events[:,-1])), dtype=object)
        for cond, markers in CONFIG['CONDITION_STIMULI'].items():
            for marker in markers:
                labels[data_epochs.events[:,-1] == marker] = cond

        # labels should be encoded to facilitate all models
        labels = sklearn.preprocessing.LabelEncoder().fit_transform(labels)

        # predefined scores array, shape (n_channels, k_folds, n_times)
        scores = np.empty(shape=(len(data_epochs.pick('data').ch_names), cv_folds, len(data_epochs.times)))

        logger.info("Performing fitting")

        # we're doing this for every channel (yes, it's somewhat hacky)
        for i, channel in tqdm(enumerate(data_epochs.pick('data').ch_names)):
            # TODO: 1 event seems to get dropped here
            data_matrix = data_epochs.get_data(picks=channel) # pick only this channel's data

            pipeline = sklearn.pipeline.make_pipeline(sklearn.preprocessing.StandardScaler(),
                                                      mne.decoding.LinearModel(self._get_model_object(model))
                                                      )

            estimator = mne.decoding.SlidingEstimator(pipeline,
                                                      n_jobs=n_jobs,
                                                      scoring=scoring,
                                                      verbose=logging.ERROR
                                                      )

            # bug in mne/fixes.py (in BaseEstimator.get_params) causes this to crash unpredictably
            # catch IndexErrors, and retry (up to a max number subsequent retries)
            err_count = 0
            while err_count < 10:
                try:
                    scores[i] = mne.decoding.cross_val_multiscore(estimator,
                                                                data_matrix,
                                                                labels,
                                                                cv=cv_folds,
                                                                n_jobs=n_jobs
                                                                )
                except IndexError:
                    logging.debug("Caught MNE error, retrying")
                    continue
                break
            
        # store results in temp folder, to be imported later
        with open(subject.channel_scores, 'wb') as tmp:
            np.save(tmp, scores)
            logger.debug(f"Wrote cross validation scores to {tmp.name}")

    def _aggregate_channel_scores(self,):
        # get first subject's scores as template for array size
        with open(self.subjects[0].channel_scores, 'rb') as tmp:
            scores = np.load(tmp)

        # insert dimension for multiple subjects
        scores.resize((len(self.subjects), *scores.shape))

        # read remaining subject scores
        for i, subject in enumerate(self.subjects[1:], start=1):
            with open(subject.channel_scores, 'rb') as tmp:
                scores[i] = np.load(tmp)

        f = CONFIG['PATHS']['RESULTS']['CHANNEL_SCORES']
        with open(f, 'wb') as tmp:
            np.save(tmp, scores)
            logger.debug(f"Wrote aggregated channel scores to {tmp.name}")

    def run_all_channel_decoding(self, rm_existing=True):
        # remove existing -channel_* files from tmp directory
        if rm_existing:
            # remove existing scoring files from directory
            for f_ in CONFIG['PATHS']['TMP'].glob('*-channel_scores.npy'):# FOR SAFETY, ONLY TOP-LEVEL FILE GLOB
                if f_.is_file(): f_.unlink()
                else: raise Exception(f"Please do not alter contents of {CONFIG['PATHS']['TMP']}")

        # run channel decoding for each subject individually
        for i, subject in enumerate(self.subjects, start=1):
            logger.info(f"Fitting channel decoding for subject {i}/{len(self.subjects)}")
            self._channel_decoding(subject)
        
        # aggregate and save results
        logger.debug("Consolidating results")
        self._aggregate_channel_scores()

        if len(self.subjects) > 1:
            # p-value calculation with FDR correction
            logger.info("Calculating p-values")

            with open(CONFIG['PATHS']['RESULTS']['CHANNEL_SCORES'], 'rb') as f:
                scores = np.load(f)
        
            p_values = MVPA.stat_utils.get_p_scores(scores.mean(2), # use average of folds
                                                    chance= 1/len(CONFIG['CONDITION_STIMULI']), # 1/no_of_conditions
                                                    tfce=False) # use FDR instead of TFCE

            # store p-values
            f = CONFIG['PATHS']['RESULTS']['CHANNEL_PVALUES']
            with open(f, 'wb') as tmp:
                np.save(tmp, p_values)
                logger.debug(f"Wrote channel p-values to {tmp.name}")

    ################################
    # ----- SEARCHLIGHT MVPA ----- #
    ################################

    def _searchlight_decoding(self, subject,
                              model=CONFIG['DECODING']['MODEL'],
                              cv_folds=CONFIG['DECODING']['CROSS_VAL_FOLDS'],
                              scoring=CONFIG['DECODING']['SCORING'],
                              n_jobs=CONFIG['DECODING']['N_JOBS']
                              ):
        pass
