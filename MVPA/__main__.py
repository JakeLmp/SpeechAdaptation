import logging, sys
logger = logging.getLogger('MVPA')
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(stream=sys.stdout))

from matplotlib.pyplot import set_loglevel;     set_loglevel('error')
from mne import set_log_level;                  set_log_level(logging.ERROR)

# prevent interactive plot windows from opening
import matplotlib
matplotlib.use('Agg')

from MVPA.preprocessing import preprocessing_main
from MVPA.GAT import GAT_main
from MVPA.plotting import plotting_main

if __name__ == '__main__':
    preprocessing_main()
    GAT_main()
    plotting_main()
