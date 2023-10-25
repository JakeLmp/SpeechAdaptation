import matplotlib.pyplot as plt
import numpy as np
import pathlib, pickle

from .stat_utils import get_p_scores

from .PARAMETERS import *

def GeneralizationScoreMatrix(scores_matrix, times_limits, score_method, p_val_contour=True, imshow_kwargs={}):
    """
    Plot the Time-Generalised Decoding score matrix.

    Args:
        scores_matrix (numpy array): 
            Matrix of shape (n_subjects, n_times, n_times) 
            containing scores of decoders for each subject, 
            at each testing time, for each training time
        times_limits (tuple): 
            (t_start, t_end) the start and end times of the x axis
        score_method (str): 
            scoring method used in calculating the decoder scores
        p_val_contour (bool):
            whether to plot a significance contour (p<0.01)
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

    if p_val_contour:
        mask = get_p_scores(scores_matrix) < 0.01
        # TODO: implement contouring in imshow

    fig, ax = plt.subplots(1, 1)
    im = ax.imshow(
        scores_matrix.mean(axis=0),
        extent=(*times_limits, *times_limits),
        **kwargs
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


def plotting_main(GAT_file = SAVE_DIR / 'GAT_results.pickle'):
    with open(GAT_file, 'rb') as f:
        all_scores = pickle.load(f)
    
    agg_results = np.empty(shape=(len(all_scores), *next(iter(all_scores.values())).shape))
    for i, (key, scores) in enumerate(all_scores.items()):       
        # aggregate all results into numpy array, with axes (subject, fold, n_times, n_times)
        agg_results[i] = scores

        # take mean score over folds
        mean_scores = scores.mean(0)

        # plot the scoring matrix of this subject
        fig, ax = GeneralizationScoreMatrix(scores_matrix = mean_scores, 
                                            times_limits = (T_MIN, T_MAX),
                                            score_method = SCORING)
        
        fig.savefig(SAVE_DIRS_DICT["Temporal Generalization Matrix"] / (key + '.png'), dpi=450)

    # plot the average scoring matrix of all subjects
    fig, ax = GeneralizationScoreMatrix(scores_matrix = agg_results.mean(axis=(0,1)), 
                                        times_limits = (T_MIN, T_MAX),
                                        score_method = 'accuracy')

    fig.savefig(SAVE_DIRS_DICT["Temporal Generalization Matrix"] / 'average_over_subjects.png', dpi=450)


if __name__ == '__main__':
    print('THIS SCRIPT IS NOT MEANT TO BE RUN INDEPENDENTLY')
    plotting_main()
