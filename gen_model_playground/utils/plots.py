import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap


def plot(zhat, z,lims=[[-2,2],[-2,2]]):
    """
    Plots a 2D histogram comparing two sets of data points.

    Args:
        zhat (array-like): The first set of 2D points to be plotted.
        z (array-like): The second set of 2D points to be plotted.
        name (str, optional): Title for the plot.
        lims (tuple, optional): Limits for the axes of the plot.

    Returns:
        matplotlib.figure.Figure, matplotlib.axes.Axes: The figure and axes objects of the plot.
    """
    fig, ax = plt.subplots(1, 1, figsize=(6.4, 6.4))
    cmap = LinearSegmentedColormap.from_list("", ["white", *plt.cm.Blues(np.arange(255))])

    _, xx, yy, _ = ax.hist2d(z[:, 0], z[:, 1], bins=(np.linspace(lims[0][0],lims[0][1],100), np.linspace(lims[1][0],lims[1][1],100) ), density=1)
    plt.close()
    fig, ax = plt.subplots(1, 1, figsize=(6.4, 6.4))
    if zhat.shape[1]==2:
        ax.hist2d(zhat[:, 0], zhat[:, 1], bins=(xx, yy), density=1, cmap=cmap)
    else:
        ax.hist(zhat, bins=100, density=1, alpha=1)
    ax.axis("off")
    plt.xlim(lims[0])
    plt.ylim(lims[1])
    plt.tight_layout()
    return fig, ax

def plot_series(z, bins=(100, 100),lims=[[-2,2],[-2,2]]):
    """
    Plots a series of 2D histograms for a sequence of data points.

    Args:
        z (list of array-like): A sequence of 2D data points to be plotted.
        name (str, optional): Title for the plot.
        bins (tuple, optional): The bin size for the histograms. Default is (100, 100).
        lims (tuple, optional): Limits for the axes of the plot.

    Returns:
        matplotlib.figure.Figure, matplotlib.axes.Axes: The figure and axes objects of the plot.
    """
    cmap = LinearSegmentedColormap.from_list("", ["white", *plt.cm.Blues(np.arange(255))])
    n = len(z)
    fig, ax = plt.subplots(1, n, figsize=(n * 6.4, 6.4))
    for i in range(n - 1):
        _, xx, yy, _ = ax[i].hist2d(z[i][:, 0].cpu().numpy(), z[i][:, 1].cpu().numpy(), bins=bins, cmap=cmap)
        ax[i].axis("off")
    _, xx, yy, _ = ax[-1].hist2d(z[-1][:, 0].cpu().numpy(), z[-1][:, 1].cpu().numpy(), bins=bins, cmap=cmap)
    plt.axis("off")
    plt.tight_layout()
    plt.xlim(lims[0])
    plt.ylim(lims[1])
    return fig


def plot_latent(z, z2,):
    """
    Plots two overlapping 2D histograms of data points in latent space, useful for visualizing distributions.

    Args:
        z (array-like): 2D data points representing the first set in latent space.
        z2 (array-like, optional): 2D data points representing the second set in latent space.
        name (str, optional): Filename for saving the plot.
        lims (tuple, optional): Limits for the axes of the plot.

    Returns:
        matplotlib.figure.Figure, matplotlib.axes.Axes: The figure and axes objects of the plot.
    """
    fig, ax = plt.subplots(1, 1, figsize=(6.4, 6.4))
    cmap_red = LinearSegmentedColormap.from_list("", ["white", "#EC3D5D"])
    cmap_blue = LinearSegmentedColormap.from_list("", ["white", "#3D5DEC"])
    # Define the binning for the histograms
    x=torch.cat((torch.from_numpy(z),torch.from_numpy(z2)),dim=0).numpy()

    if x.shape[1]==2:
        bins = [np.linspace(-3,3, 100),
                np.linspace(-3,3, 100)]
        # Plot the lower moon
        ax.hist2d(z[:, 0], z[:, 1], bins=bins, cmap=cmap_blue, alpha=0.5, label="Lower Moon",density=1,vmin=0.01)
        # Plot the upper moon
        ax.hist2d(z2[:, 0], z2[:, 1], bins=bins, cmap=cmap_red, alpha=0.5,  label="Upper Moon",density=1,vmin=0.01)

    else:
        bins = 100
        ax.hist(z, bins=bins, alpha=0.5, label="Lower Moon",density=1,color="#EC3D5D")
        ax.hist(z2, bins=bins, alpha=0.5, label="Upper Moon",density=1, color="#3D5DEC")



    plt.legend()
    ax.axis("off")
    plt.tight_layout()

    return fig, ax

def plot_logprob(x, y, z, lims=[[-2,2],[-2,2]]):
    """
    Plots logarithmic probabilities on a 2D grid.

    Args:
        x (array-like): 1D array representing the x-coordinates of the grid.
        y (array-like): 1D array representing the y-coordinates of the grid.
        z (array-like): 2D array representing the probabilities at each grid point.

    Returns:
        Tuple of figures and axes objects for the normal and logarithmic plots.
    """
    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    cmap = LinearSegmentedColormap.from_list("", ["white", *plt.cm.Blues(np.arange(255))])

    plt.pcolor(x, y, z,  cmap=cmap)
    plt.axis("off")
    plt.tight_layout()
    logfig, logax = plt.subplots(figsize=(6.4, 6.4))
    plt.pcolor(x, y, z, norm=mpl.colors.LogNorm(), cmap=cmap)
    plt.axis("off")
    plt.tight_layout()
    plt.xlim(lims[0])
    plt.ylim(lims[1])
    return fig, ax, logfig, logax

