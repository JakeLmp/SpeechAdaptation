import matplotlib.pyplot as plt

def GeneralizationScoreMatrix(scores_matrix, times_limits, score_method):
    fig, ax = plt.subplots(1, 1)
    im = ax.imshow(
        scores_matrix,
        interpolation="lanczos",
        origin="lower",
        cmap="RdBu_r",
        extent=(*times_limits, *times_limits),
        # vmin=0.0,
        # vmax=1.0,
    )
    ax.set_xlabel("Testing Time (s)")
    ax.set_ylabel("Training Time (s)")
    ax.set_title("Temporal generalization")
    ax.axvline(0, color="k")
    ax.axhline(0, color="k")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(score_method)