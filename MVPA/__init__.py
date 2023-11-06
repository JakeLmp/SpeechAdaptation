import logging, sys
logger = logging.getLogger('MVPA')
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(stream=sys.stdout))

from matplotlib.pyplot import set_loglevel;     set_loglevel('error')
from mne import set_log_level;                  set_log_level(logging.ERROR)

from .MVPA import MVPA_manager

def main():
    mvpa_manager = MVPA_manager()
    mvpa_manager.preprocess_all()
    mvpa_manager.run_all_gat()
    mvpa_manager.run_all_sensor_space_decoding()
