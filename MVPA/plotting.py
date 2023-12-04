import logging
logger = logging.getLogger('MVPA')

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import mne
# import pathlib, pickle

# from .stat_utils import get_p_scores
from .utils import config_prep
CONFIG = config_prep()

def GeneralizationScoreMatrix(scores_matrix, times_limits, score_method, p_values=None, p_value_threshold=0.05, imshow_kwargs={}):
    """
    Plot the Time-Generalised Decoding score matrix.

    Args:
        scores_matrix (numpy ndarray): 
            Matrix of shape (n_subjects, n_times, n_times)
            containing scores of decoders for each subject, 
            at each testing time, for each training time
        times_limits (tuple): 
            (t_start, t_end) the start and end times of the x axis
        score_method (str): 
            scoring method used in calculating the decoder scores
        p_values (numpy ndarray):
            Default None, else a numpy ndarray of shape (n_times, n_times) 
            containing p-values to be used in plotting a 
            significance contour
        p_value_threshold (float):
            Default 0.05, p-value threshold value to use in masking
        imshow_kwargs (dict): 
            keyword arguments to be passed to matplotlib.pyplot.imshow

    Returns:
        fig, ax: matplotlib Figure and Axes objects containing the plot
    """
    
    kwargs = dict(interpolation = "lanczos",
                  origin        = "lower",
                  cmap          = "RdBu_r"
                  )

    kwargs.update(imshow_kwargs)
    
    # we need the mean over subjects
    data = scores_matrix.mean(0)

    # create mask for significant area
    if isinstance(p_values, np.ndarray):
        mask = p_values < p_value_threshold
    # if no p-values, everything is significant
    else:
        mask = np.ones_like(data, dtype='int')

    fig, ax = plt.subplots(1, 1)

    # create alpha value array, where non-significant elements get lower alpha
    alpha = np.ones_like(data, dtype='float32')
    alpha[~mask] = 0.5

    im = ax.imshow(
        data,
        alpha=alpha,
        extent=(*times_limits, *times_limits),
        **kwargs
    )
    
    # lastly, we draw contour lines around significant areas (if p_values is given)
    if isinstance(p_values, np.ndarray):
        # matplotlib requires weird coordinate formats, this complies to that.
        x = np.linspace(*times_limits, data.shape[0])
        y = np.linspace(*times_limits, data.shape[1])
        x, y = np.meshgrid(x, y)
        
        ax.contour(x, y, mask, 
                   levels=[0.5],
                   linewidths=[1],
                   colors=['k'],
                   corner_mask=False,
                  )

    ax.set_xlabel("Testing Time (s)")
    ax.set_ylabel("Training Time (s)")
    ax.set_title("Temporal generalization")
    ax.axvline(0, color="k", linestyle=':')
    ax.axhline(0, color="k", linestyle=':')
    ax.plot(times_limits, times_limits, color="k", linestyle=":")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(score_method)

    return fig, ax

def LineBandPlot(x, y, err_lower, err_upper, plot_kwargs={}):
    """
    Plot the line defined by (x, y), with bands defined by (y ± y_err)

    Args:
        x (list-like): x axis values
        y (list-like): y axis values
        err_lower (list-like): lower bound of bandwidth around y
        err_upper (list-like): upper bound of bandwidth around y
        plot_kwargs (dict): 
            keyword arguments to be passed to matplotlib.pyplot.plot

    Returns:
        fig, ax: matplotlib Figure and Axes objects containing the plot
    """

    kwargs = dict()

    kwargs.update(plot_kwargs)

    fig, ax = plt.subplots(1, 1)

    ax.plot(x, y, **kwargs)
    ax.fill_between(x, err_lower, err_upper, alpha=.25)
    ax.set_xlim(left=x[0], right=x[-1])
    ax.grid(visible=True)

    return fig, ax

def GeneralizationDiagonal(scores_matrix, times_limits, score_method, p_values=None, p_value_threshold=0.05, plot_kwargs={}):
    """
    Plot the diagonal of the Time-Generalised Decoding score matrix.

    Args:
        scores_matrix (numpy array): 
            Matrix of shape (n_subjects, n_times, n_times) 
            containing scores of decoders for each subject, 
            at each testing time, for each training time
        times_limits (tuple): 
            (t_start, t_end) the start and end times of the x axis
        score_method (str): 
            scoring method used in calculating the decoder scores
        p_values (numpy array):
            Default None, else a numpy ndarray of shape (n_times, n_times) 
            containing p-values to be used in plotting 
            significance indicators
        p_value_threshold (float | list of floats):
            Default 0.05, float or list of floats containing
            values to be used for significance level indication
        plot_kwargs (dict): 
            keyword arguments to be passed to matplotlib.pyplot.plot

    Returns:
        fig, ax: matplotlib Figure and Axes objects containing the plot
    """

    # kwargs = {'title' : 'Decoding accuracy on GAT diagonal',
    #           'xlabel' : 'Time (s)',
    #           'ylabel' : score_method, 
    #           }

    kwargs = dict()

    kwargs.update(plot_kwargs)

    # we need the mean over subjects
    data = scores_matrix.mean(0)

    x = np.linspace(*times_limits, data.shape[0])
    y = np.diagonal(data)                               # diagonal of mean over subjects
    
    # if more than 1 subject, we can calculate a confidence interval
    if scores_matrix.shape[0] > 1:
        res = scipy.stats.bootstrap((np.diagonal(scores_matrix, axis1=1, axis2=2),), np.mean, axis=0,  
                                    confidence_level=0.95,
                                    n_resamples=1000
                                    )
        ci_l, ci_u = res.confidence_interval    # lower and upper bound of 95% conf. int. of mean
    else: # use a band of width 0
        ci_l, ci_u = y.copy(), y.copy()

    fig, ax = LineBandPlot(x, y, ci_l, ci_u, plot_kwargs=kwargs)

    # now for the significance indicator lines (if p_values is given)
    if isinstance(p_values, np.ndarray):
        p_values = np.diagonal(p_values)

        # if not iterable already, make it so
        try:
            _ = iter(p_value_threshold)
        except TypeError:
            p_value_threshold = [p_value_threshold]
        
        # descending order
        p_value_threshold = np.flip(np.sort(p_value_threshold))

        # putting height of significance indicators at 1/8th of the plot
        lims = ax.get_ylim()
        sign_y = lims[0] + (lims[1] - lims[0])/8

        # plot significance indicators
        for i, p in enumerate(p_value_threshold, start=1):
            intervals = np.ma.masked_where(~(p_values < p), x) # transparent where significant
            ax.plot(intervals, np.full_like(intervals, fill_value=sign_y), 
                    linewidth = i*3, # increasing thickness for increasing significance
                    color='k',
                    solid_capstyle='round'
                    )

    ax.set_title('Decoding Score over Time')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(score_method)

    return fig, ax

def ChannelScoresMatrix(scores_matrix, channel_labels, times_limits, score_method, 
                        p_values=None, p_value_threshold=0.05, imshow_kwargs={}):
    """_summary_

    Args:
        scores_matrix (numpy ndarray): 
            Matrix of shape (n_subjects, n_channels, n_times)
            containing scores of decoders for each subject, 
            for each channel, at each time point
        channel_labels (list of str):
            List containing channel labels in the same order
            as the corresponding axis of scores_matrix
        times_limits (tuple): 
            (t_start, t_end) the start and end times of the x axis
        score_method (str): 
            scoring method used in calculating the decoder scores
        p_values (numpy ndarray):
            Default None, else a numpy ndarray of shape (n_channels, n_times) 
            containing p-values to be used in plotting a 
            significance contour
        p_value_threshold (float):
            Default 0.05, p-value threshold value to use in masking
        imshow_kwargs (dict): 
            keyword arguments to be passed to matplotlib.pyplot.imshow
            Defaults to {}.
    """
    
    kwargs = dict(interpolation = "lanczos",
                  cmap          = "RdBu_r"
                  )

    kwargs.update(imshow_kwargs)

    # we need the mean over subjects
    data = scores_matrix.mean(0)

    # create mask for significant area
    if isinstance(p_values, np.ndarray):
        mask = p_values < p_value_threshold
    # if no p-values, everything is significant
    else:
        mask = np.ones_like(data, dtype='int')
    
    fig, ax = plt.subplots(1, 1)

    # create alpha value array, where non-significant elements get lower alpha
    alpha = np.ones_like(data, dtype='float32')
    alpha[~mask] = 0.5

    im = ax.imshow(
        data,
        alpha=alpha,
        extent=(*times_limits, *times_limits),
        **kwargs
    )

    # we draw contour lines around significant areas (if p_values is given)
    if isinstance(p_values, np.ndarray):
        # matplotlib requires weird coordinate formats, this complies to that.
        x = np.linspace(*times_limits, data.shape[0])
        y = np.linspace(*times_limits, data.shape[1])
        x, y = np.meshgrid(x, y)
        
        ax.contour(x, y, mask, 
                   levels=[0.5],
                   linewidths=[1],
                   colors=['k'],
                   corner_mask=False,
                  )

    # y-axis tick positions
    pos = np.linspace(*times_limits, len(channel_labels))
    # pos = pos + (pos[1]-pos[0])/2 # offset by half the tick distance

    # stagger every other label (order is reversed due to imshow's y-axis treatment)
    channel_labels = [ch + 8*' ' if i%2==0 else ch for i, ch in enumerate(reversed(channel_labels))]
    ax.set_yticks(pos, channel_labels,
                  fontsize='xx-small',
                  ha='right')

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Channel")
    ax.set_title("Channel Decoding Score")
    ax.axvline(0, color="k", linestyle=':')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(score_method)

    return fig, ax


def ChannelAccuracyTopo():
    """
    TO DO
    """
    
    pass    
    
    # fig, ax = plt.subplots(1, 1)

    # im, cn = mne.viz.plot_topomap(
    #     data=scores.mean(0), 
    #     pos=np.array([ch['loc'][:2] for ch in data_epochs.pick('data').info['chs']]),
    #     names=[ch['ch_name'] for ch in data_epochs.pick('data').info['chs']],
    #     res=1000,
    #     size=3,
    #     axes=ax
    #     )

    # cbar = plt.colorbar(im, ax=ax)

def generate_all_plots(spoofed_subject=False, save_kwargs={}):
    logger.info("Generating plots")

    kwargs = dict(dpi=450,
                  )
    
    kwargs.update(save_kwargs)
    
    # --- GAT plots
    f = CONFIG['PATHS']['RESULTS']['GAT_RESULTS']
    with open(f, 'rb') as tmp:
        GAT_results = np.load(tmp)
    
    if not spoofed_subject:
        f = CONFIG['PATHS']['RESULTS']['GAT_PVALUES']
        with open(f, 'rb') as tmp:
            GAT_pvalues = np.load(tmp)
    else:
        GAT_pvalues = None
    
    fig, ax = GeneralizationScoreMatrix(GAT_results.mean(1), 
                                        times_limits=(CONFIG['MNE']['T_MIN'], CONFIG['MNE']['T_MAX']),
                                        score_method=CONFIG['DECODING']['SCORING'],
                                        p_values=GAT_pvalues,
                                        p_value_threshold=0.05
                                        )
    f = CONFIG['PATHS']['PLOT'] / 'GAT_matrix.png'
    fig.savefig(f, **kwargs)
    logger.debug(f"Wrote GAT matrix plot to {f}")
    plt.close(fig)
    
    fig, ax = GeneralizationDiagonal(GAT_results.mean(1), 
                                     times_limits=(CONFIG['MNE']['T_MIN'], CONFIG['MNE']['T_MAX']),
                                     score_method=CONFIG['DECODING']['SCORING'],
                                     p_values=GAT_pvalues,
                                     p_value_threshold=[0.05,0.01]
                                     )
    f = CONFIG['PATHS']['PLOT'] / 'GAT_diagonal.png'
    fig.savefig(f, **kwargs)
    logger.debug(f"Wrote GAT diagonal plot to {f}")
    plt.close(fig)

    # --- temporal plots

    pass

    # --- channel plots

    f = CONFIG['PATHS']['RESULTS']['CHANNEL_SCORES']
    with open(f, 'rb') as tmp:
        channel_results = np.load(tmp)
    
    if not spoofed_subject:
        f = CONFIG['PATHS']['RESULTS']['CHANNEL_PVALUES']
        with open(f, 'rb') as tmp:
            channel_pvalues = np.load(tmp)
    else:
        channel_pvalues = None

    f = CONFIG['PATHS']['INFO_OBJ']
    info = mne.io.read_info(f)

    # make channel plotting order
    data_channels = [(ch['ch_name'], idx) for idx, ch in enumerate(info['chs'])] # all channels present in data, with their index
    channel_labels = dict([tup for ch in CONFIG['DEFAULT']['CHANNEL_ORDER'] for tup in data_channels if tup[0] == ch]) # (ch_name, idx) dict

    # reordering matrix according to plotting order
    channel_results = channel_results[:, list(channel_labels.values()), ...]
    if not spoofed_subject:
        channel_pvalues = channel_pvalues[list(channel_labels.values()), ...]

    fig, ax = ChannelScoresMatrix(channel_results.mean(2),
                                  channel_labels=list(channel_labels.keys()),
                                  times_limits=(CONFIG['MNE']['T_MIN'], CONFIG['MNE']['T_MAX']),
                                  score_method=CONFIG['DECODING']['SCORING'],
                                  )
    f = CONFIG['PATHS']['PLOT'] / 'channel_scores.png'
    fig.savefig(f, **kwargs)
    logger.debug(f"Wrote channel scores plot to {f}")
    plt.close(fig)

    # butterfly w/ all channels
    pass

    # just topo
    pass

if __name__ == '__main__':
    print('THIS SCRIPT IS NOT MEANT TO BE RUN INDEPENDENTLY')
