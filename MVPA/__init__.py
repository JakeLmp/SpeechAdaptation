import logging, sys
logger = logging.getLogger('MVPA')
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(stream=sys.stdout))

from matplotlib.pyplot import set_loglevel;     set_loglevel('error')
from mne import set_log_level;                  set_log_level(logging.ERROR)

from . import MVPA, plotting

import argparse

def main():
    # parser = argparse.ArgumentParser(prog="MVPA",
    #                                  description="Does MVPA for EEG data.",
    #                                  )

    manager = MVPA.DecodingManager()
    manager.preprocess_all()
    # manager.spoof_single_subject()
    manager.run_all_gat()
    manager.run_all_temporal_decoding()
    manager.run_all_channel_decoding()

    # at this point, all calculation work should be finished
    # and we can generate plots using the results
    plotting.generate_all_plots()

    print('Success!')