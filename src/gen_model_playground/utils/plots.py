import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap


def plot(zhat, z, name=""):
    """
    Plots a 2D histogram comparing two sets of data points.

    Args:
        zhat (array-like): The first set of 2D points to be plotted.
        z (array-like): The second set of 2D points to be plotted.
        name (str, optional): Title for the plot.

    Returns:
        matplotlib.figure.Figure, matplotlib.axes.Axes: The figure and axes objects of the plot.
    """
    fig, ax = plt.subplots(1, 1, figsize=(6.4, 6.4))
    cmap = LinearSegmentedColormap.from_list("", ["white", *plt.cm.Blues(np.arange(255))])

    _, xx, yy, _ = ax.hist2d(z[:, 0], z[:, 1], bins=(100, 100), density=1)
    plt.close()
    fig, ax = plt.subplots(1, 1, figsize=(6.4, 6.4))
    ax.hist2d(zhat[:, 0], zhat[:, 1], bins=(xx, yy), density=1, cmap=cmap)
    ax.axis("off")
    plt.tight_layout()
    return fig, ax

def plot_series(z, name="", bins=(100, 100)):
    """
    Plots a series of 2D histograms for a sequence of data points.

    Args:
        z (list of array-like): A sequence of 2D data points to be plotted.
        name (str, optional): Title for the plot.
        bins (tuple, optional): The bin size for the histograms. Default is (100, 100).

    Returns:
        matplotlib.figure.Figure, matplotlib.axes.Axes: The figure and axes objects of the plot.
    """
    n = len(z)
    fig, ax = plt.subplots(1, n, figsize=(n * 6.4, 6.4))
    for i in range(n - 1):
        _, xx, yy, _ = ax[i].hist2d(z[i][:, 0].cpu().numpy(), z[i][:, 1].cpu().numpy(), bins=bins)
        ax[i].axis("off")
    _, xx, yy, _ = ax[-1].hist2d(z[-1][:, 0].cpu().numpy(), z[-1][:, 1].cpu().numpy(), bins=bins)
    plt.axis("off")
    plt.tight_layout()
    return fig, ax


def plot_latent(z, z2, name="latent.pdf"):
    """
    Plots two overlapping 2D histograms of data points in latent space, useful for visualizing distributions.

    Args:
        z (array-like): 2D data points representing the first set in latent space.
        z2 (array-like, optional): 2D data points representing the second set in latent space.
        name (str, optional): Filename for saving the plot.

    Returns:
        matplotlib.figure.Figure, matplotlib.axes.Axes: The figure and axes objects of the plot.
    """
    fig, ax = plt.subplots(1, 1, figsize=(6.4, 6.4))
    cmap_red = LinearSegmentedColormap.from_list("", ["white", "#EC3D5D"])
    cmap_blue = LinearSegmentedColormap.from_list("", ["white", "#3D5DEC"])
    # Define the binning for the histograms
    x=torch.cat((torch.from_numpy(z),torch.from_numpy(z2)),dim=0).numpy()
    bins = [np.linspace(x[:, 0].min(), x[:, 0].max(), 100),
            np.linspace(x[:, 1].min(), x[:, 1].max(), 100)]



    # Plot the first histogram
    ax.hist2d(z[:, 0], z[:, 1], bins=bins, cmap=cmap_blue, alpha=0.5, label="Lower Moon",density=1,vmin=0.01)


    # Plot the second histogram if it exists
    ax.hist2d(z2[:, 0], z2[:, 1], bins=bins, cmap=cmap_red, alpha=0.5,  label="Upper Moon",density=1,vmin=0.01)


    plt.legend()
    ax.axis("off")
    plt.tight_layout()

    return fig, ax

def plot_logprob(x, y, z):
    """
    Plots logarithmic probabilities on a 2D grid.

    Args:
        x (array-like): 1D array representing the x-coordinates of the grid.
        y (array-like): 1D array representing the y-coordinates of the grid.
        z (array-like): 2D array representing the probabilities at each grid point.

    Returns:
        Tuple of figures and axes objects for the normal and logarithmic plots.
    """
    fig, ax = plt.subplots()
    cmap = LinearSegmentedColormap.from_list("", ["white", *plt.cm.Blues(np.arange(255))])

    plt.pcolor(x, y, z,  cmap=cmap)
    plt.axis("off")
    plt.tight_layout()
    logfig, logax = plt.subplots()
    plt.pcolor(x, y, z, norm=mpl.colors.LogNorm(), cmap=cmap)
    plt.axis("off")
    plt.tight_layout()
    return fig, ax, logfig, logax
