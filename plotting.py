import matplotlib.pyplot as plt

def GeneralizationScoreMatrix(scores_matrix, times_limits, score_method, imshow_kwargs={}):
    """
    Plot the Time-Generalised Decoding score matrix.

    Args:
        scores_matrix (_numpy array_): _n*n matrix containing scores of decoders at each testing time, for each trainning time_
        times_limits (_iterable_): _iterable containing the start and end times of the matrix axes_
        score_method (_str_): _scoring method used in calculating the decoder scores_
        imshow_kwargs (_dict_): _keyword arguments to be passed to matplotlib.pyplot.imshow_

    Returns:
        _fig, ax_: _matplotlib Figure and Axes objects containing the plot_
    """
    
    kwargs = {"interpolation":"lanczos",
              "origin":"lower",
              "cmap":"RdBu_r"}

    kwargs.update(imshow_kwargs)

    fig, ax = plt.subplots(1, 1)
    im = ax.imshow(
        scores_matrix,
        extent=(*times_limits, *times_limits),
        # vmin=0.0,
        # vmax=1.0,
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