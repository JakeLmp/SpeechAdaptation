import matplotlib.pyplot as plt

def GeneralizationScoreMatrix(scores_matrix, times_limits, score_method, imshow_kwargs={}):
    """
    Plot the Time-Generalised Decoding score matrix.

    Args:
        scores_matrix (numpy array): 
            n*n matrix containing scores of decoders at each testing time, for each trainning time
        times_limits (iterable): 
            iterable containing the start and end times of the matrix axes
        score_method (str): 
            scoring method used in calculating the decoder scores
        imshow_kwargs (dict): 
            keyword arguments to be passed to matplotlib.pyplot.imshow

    Returns:
        fig, ax: matplotlib Figure and Axes objects containing the plot
    """
    
    kwargs = {"interpolation":"lanczos",
              "origin":"lower",
              "cmap":"RdBu_r"}

    kwargs.update(imshow_kwargs)

    fig, ax = plt.subplots(1, 1)
    im = ax.imshow(
        scores_matrix,
        extent=(*times_limits, *times_limits),
        **kwargs
    )
    ax.set_xlabel("Testing Time (s)")
    ax.set_ylabel("Training Time (s)")
    ax.set_title("Temporal generalization")
    ax.axvline(0, color="k")
    ax.axhline(0, color="k")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(score_method)

    return fig, ax