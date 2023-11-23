import logging, sys
logger = logging.getLogger('MVPA')
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(stream=sys.stdout))

from matplotlib.pyplot import set_loglevel;     set_loglevel('error')
from mne import set_log_level;                  set_log_level(logging.ERROR)

from . import MVPA, plotting

def main():
    manager = MVPA.DecodingManager()
    manager.preprocess_all()
    manager.run_all_gat()
    manager.run_all_sensor_space_decoding()

    # at this point, all calculation work should be finished
    # and we can generate plots using the results
    plotting.generate_all_plots()