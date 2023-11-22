import logging
logger = logging.getLogger('MVPA')

import matplotlib.pyplot as plt
import numpy as np
# import pathlib, pickle

# from .stat_utils import get_p_scores
from .utils import config_prep
CONFIG = config_prep()

def GeneralizationScoreMatrix(scores_matrix, times_limits, score_method, p_values=None, p_value_threshold=0.01, imshow_kwargs={}):
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
            Default 0.01, p-value threshold value to use in masking
        imshow_kwargs (dict): 
            keyword arguments to be passed to matplotlib.pyplot.imshow

    Returns:
        fig, ax: matplotlib Figure and Axes objects containing the plot
    """
    
    kwargs = {"interpolation":  "lanczos",
              "origin":         "lower",
              "cmap":           "RdBu_r"
              }

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

    # if isinstance(p_values, np.ndarray):
    #     ax.contour(mask, 
    #                levels=[0.5],
    #                linewidths=[1],
    #                colors=['k'],
    #                corner_mask=False,
    #               )

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
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(score_method)

    return fig, ax

def LineBandPlot(x, y, y_err):
    """
    Plot the line defined by (x, y), with bands defined by (y ± y_err)

    Args:
        x (list-like): x axis values
        y (list-like): y axis values
        y_err (list-like): band width around y

    Returns:
        fig, ax: matplotlib Figure and Axes objects containing the plot
    """
    fig, ax = plt.subplots(1, 1)

    ax.plot(x, y)
    ax.fill_between(x, y-y_err, y+y_err, alpha=.25)
    ax.set_xlim(left=x[0], right=x[-1])
    ax.grid(visible=True)
    ax.axis('tight')

    return fig, ax

def GeneralizationDiagonal(scores_matrix, times_limits, score_method, plot_kwargs={}):
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
        plot_kwargs (dict): 
            keyword arguments to be passed to matplotlib.pyplot.imshow

    Returns:
        fig, ax: matplotlib Figure and Axes objects containing the plot
    """
    if len(scores_matrix.shape) != 3:
        raise ValueError(f"Expected array of shape (n_subjects, n_times, n_times), got array with shape {scores_matrix.shape} instead")
    
    x = np.linspace(*times_limits, scores_matrix.shape[1])
    y = np.diagonal(scores_matrix.mean(axis=0))             # diagonal of mean over subjects
    y_err = np.diagonal(np.std(scores_matrix, axis=0))      # diagonal of standard deviations over subjects

    # kwargs = {'title' : 'Decoding accuracy on diagonal',
    #           'xlabel' : 'Time (s)',
    #           'ylabel' : score_method,
    #           'axis' : 'tight' 
    #           }

    # kwargs.update(plot_kwargs)
    
    p_ = get_p_scores(scores_matrix, chance=.5, tfce=False) 

    fig, ax = LineBandPlot(x, y, y_err)

    ax.set_title('Decoding accuracy on diagonal')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(score_method)

    return fig, ax

def generate_all_plots():
    # --- GAT plots
    f = CONFIG['PATHS']['SAVE'] / 'GAT_results.npy'
    with open(f, 'rb') as tmp:
        GAT_results = np.load(tmp)
    
    f = CONFIG['PATHS']['SAVE'] / 'GAT_pvalues.npy'
    with open(f, 'rb') as tmp:
        GAT_pvalues = np.load(tmp)
    
    fig, ax = GeneralizationScoreMatrix(GAT_results.mean(1))
    

if __name__ == '__main__':
    print('THIS SCRIPT IS NOT MEANT TO BE RUN INDEPENDENTLY')
