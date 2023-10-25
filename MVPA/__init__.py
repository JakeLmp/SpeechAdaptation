import logging, sys
logger = logging.getLogger('MVPA')
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(stream=sys.stdout))

from matplotlib.pyplot import set_loglevel;     set_loglevel('error')
from mne import set_log_level;                  set_log_level(logging.ERROR)

from .preprocessing import preprocessing_main
from .GAT import GAT_main
from .plotting import plotting_main

def main():
    preprocessing_main()
    GAT_main()
    plotting_main()
