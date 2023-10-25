import mne
import logging
logger = logging.getLogger('MVPA')

# this is where all user-defined parameters (constants) are located
from .PARAMETERS import *



############################
# ----- DATA IMPORTS ----- #
############################

def preprocessing_main():
    # remove existing epoch files from directory
    for _f in SAVE_DIR_TMP.glob('*-epo.fif'): # FOR SAFETY, ONLY TOP-LEVEL FILE GLOB
        if _f.is_file(): _f.unlink()
        else: raise Exception(f"Please do not alter contents of {SAVE_DIR_TMP}")

    if DATA_PATH.is_file():
        files = [DATA_PATH]
        logger.info(f"Loading file {DATA_PATH}")
    elif DATA_PATH.is_dir():
        files = list(DATA_PATH.glob('*.vhdr'))
        logger.info(f"Loading files from {DATA_PATH} (found {len(files)} files)")

    # run Time-Generalised Decoding for each subject individually
    for i, f in enumerate(files, start=1):
        logger.info(f"Processing file {i}/{len(files)} : {f.name}")
        data_raw = mne.io.read_raw_brainvision(f, **DATA_ARGS)

        data_raw.load_data()
        data_raw.set_eeg_reference(REFERENCE_ELECTRODES)
        data_raw.info['bads'] = BAD_ELECTRODES

        # resample if selected to do so here, or check if another valid option was given.
        if RESAMPLE_AT.lower() == 'raw':
            _t = data_raw.info['sfreq']
            data_raw.resample(RESAMPLE_FREQUENCY)
            logger.info(f"Succesfully resampled RAW data (was {_t} Hz, is now {data_raw.info['sfreq']} Hz)")
        elif RESAMPLE_AT.lower() == 'epoch':
            pass
        elif RESAMPLE_AT.lower() == 'do_not_resample':
            pass
        else:
            s = f"Invalid resampling method parameter ({RESAMPLE_AT})"
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

        logger.info(f"Found event IDs: {events[1]}")

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
            logger.info(f"Succesfully resampled EPOCH data (was {_t} Hz, is now {data_epochs.info['sfreq']} Hz)")

        # ERP plot as reference check for successful data import/conversion
        data_epochs.average(picks=ERP_CHECK_ELECTRODES if len(ERP_CHECK_ELECTRODES) > 0 else None) \
                .plot() \
                .savefig(SAVE_DIRS_DICT["ERP"] / f.with_suffix('.png').name, dpi=300)
        
        # store files in tmp folder
        logger.info(f"Storing processed data at {SAVE_DIR_TMP / (f.stem + '-epo.fif')}")
        data_epochs.save(SAVE_DIR_TMP / (f.stem + '-epo.fif'))

if __name__ == ' __main__':
    print('THIS SCRIPT IS NOT MEANT TO BE RUN INDEPENDENTLY')
