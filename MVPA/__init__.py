import logging, sys
logging.basicConfig(level=logging.WARNING,
                    stream=sys.stdout)
logger = logging.getLogger('MVPA')

logging.getLogger('matplotlib').setLevel(logging.ERROR)
from mne import set_log_level; set_log_level(logging.ERROR)

from . import MVPA, plotting

import argparse

def main():
    parser = argparse.ArgumentParser(prog=__name__,
                                     description="""
Package for MVPA (decoding) for EEG data. Includes Generalisation Across Time, Temporal Decoding (all channels are features), 
Channel Decoding (searchlight decoding with single-channel features) and Searchlight Decoding (searchlight with primitive neighbour definition). 
Generates plots using calculation outcomes.

See the package configuration file 'PARAMETERS.toml' for parameter specification.

""")
    
    parser.add_argument('-a', '--all', action='store_true', help="Default behaviour. Run preprocessing, do all calculation work, and generate plots.")
    parser.add_argument('-p', '--plotting-only', action='store_true', help="Skip all calculations, only generate plots. Only works if calculations were done previously and and files are still in the tmp folder. Choosing this option overwrites all other options.")
    parser.add_argument('-v', '--verbose', action='store_true', help="Print logging info")
    parser.add_argument('--spoof-subject', action='store_true', help="Aggregates all subjects into one 'spoofed' subject. Calculations are performed with this subject only. Doing this voids validity of p-value calculations.")
    parser.add_argument('--skip-preprocessing', action='store_true', help="Skip preprocessing steps, and use previously generated files. Only works if preprocessing was performed previously, and files are still in the tmp folder.")
    parser.add_argument('--skip-gat', action='store_true', help="Skip Generalised Across Time calculations.")
    parser.add_argument('--skip-temporal', action='store_true', help="Skip Temporal Decoding calculations.")
    parser.add_argument('--skip-channel', action='store_true', help="Skip Channel Decoding calculations.")
    parser.add_argument('--skip-searchlight', action='store_true', help="Skip Searchlight Decoding calculations.")

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.WARNING)

    manager = MVPA.DecodingManager()

    if not args.plotting_only:
        if not args.skip_preprocessing:
            manager.preprocess_all()
        if args.spoof_subject:
            manager.spoof_single_subject()
        if not args.skip_gat:
            manager.run_all_gat()
        if not args.skip_temporal:
            manager.run_all_temporal_decoding()
        if not args.skip_channel:
            manager.run_all_channel_decoding()
        if not args.skip_searchlight:
            manager.run_all_searchlight_decoding()
    
    # at this point, all calculation work should be finished
    # and we can generate plots using the results
    plotting.generate_all_plots(spoofed_subject=args.spoof_subject)
