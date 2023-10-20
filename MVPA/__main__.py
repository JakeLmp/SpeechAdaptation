import logging, sys
logger = logging.getLogger('MVPA')
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(stream=sys.stdout))

from matplotlib.pyplot import set_loglevel;     set_loglevel('error')
from mne import set_log_level;                  set_log_level(logging.ERROR)

# prevent interactive plot windows from opening
import matplotlib
matplotlib.use('Agg')

if __name__ == '__main__':
    from MVPA.preprocessing import *
    from MVPA.GAT import *
