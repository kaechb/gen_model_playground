import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap
from sklearn.datasets import make_moons


def plot(zhat, z, name=""):
    fig, ax = plt.subplots(1, 1, figsize=(6.4, 6.4))
    cmap = LinearSegmentedColormap.from_list(
        "", ["white", *plt.cm.Blues(np.arange(255))]
    )
    cmap2 = LinearSegmentedColormap.from_list(
        "", ["white", *plt.cm.Reds(np.arange(255))]
    )

    _, xx, yy, _ = ax.hist2d(z[:, 0], z[:, 1], bins=(100, 100), density=1)
    plt.close()
    fig, ax = plt.subplots(1, 1, figsize=(6.4, 6.4))
    ax.hist2d(zhat[:, 0], zhat[:, 1], bins=(xx, yy), density=1, cmap=cmap)
    ax.axis("off")
    plt.axis("off")
    plt.tight_layout()
    return fig, ax


def plot_series(z, name="", bins=(100, 100)):
    n = len(z)
    fig, ax = plt.subplots(1, n, figsize=(n * 6.4, 6.4))
    x = make_moons(len(z), noise=0.05)[0]
    for i in range(n - 1):
        _, xx, yy, _ = ax[i].hist2d(
            z[i][:, 0].cpu().numpy(), z[i][:, 1].cpu().numpy(), bins=bins
        )
        ax[i].axis("off")
    _, xx, yy, _ = ax[-1].hist2d(
        z[-1][:, 0].cpu().numpy(), z[-1][:, 1].cpu().numpy(), bins=bins
    )
    plt.axis("off")
    plt.tight_layout()
    return fig, ax


def plot_latent(z, z2=None, name="latent.pdf"):
    fig, ax = plt.subplots(1, 1, figsize=(6.4, 6.4))
    cmap = LinearSegmentedColormap.from_list(
        "", ["white", *plt.cm.Blues(np.arange(255))]
    )
    cmap2 = LinearSegmentedColormap.from_list(
        "", ["white", *plt.cm.Reds(np.arange(255))]
    )

    if z.shape[1] == 1:
        fig, ax = plt.subplots(1, 1, figsize=(6.4, 6.4))

        plt.hist(z[:1000, 0], bins=100, label="Lower Moon", alpha=0.01, cmap=cmap)
        plt.hist(z2[:1000, 0], bins=100, label="Upper Moon", alpha=0.01, cmap=cmap2)
    else:
        x = torch.normal(0, 1, size=(len(z), 2)).detach().numpy()
        # create a colormap object
        _, xx, yy, _ = ax.hist2d(x[:, 0], x[:, 1], bins=(100, 100), alpha=0.5)
        plt.close()
        fig, ax = plt.subplots(1, 1, figsize=(6.4, 6.4))

        ax.scatter(z2[:5000, 0], z2[:5000, 1], label="Upper Moon", cmap=cmap, alpha=0.1)
        h = ax.scatter(
            z[:5000, 0], z[:5000, 1], label="Lower Moon", cmap=cmap2, alpha=0.1
        )

    plt.legend()
    ax.axis("off")
    plt.axis("off")
    plt.tight_layout()
    return fig, ax


def plot_logprob(x, y, z):
    fig, ax = plt.subplots()

    plt.pcolor(x, y, z, vmin=0.01, vmax=1)
    plt.axis("off")
    #     plt.title(r'$p_Z(\mathbf{f}^{-1}(\mathbf{z}))$')
    plt.tight_layout()
    logfig, logax = plt.subplots()
    cmap = LinearSegmentedColormap.from_list(
        "", ["white", *plt.cm.Blues(np.arange(255))]
    )
    plt.pcolor(x, y, z, norm=mpl.colors.LogNorm(), cmap=cmap)
    #     plt.title(r'log$p_Z(\mathbf{f}(\mathbf{z}))$')
    plt.axis("off")
    plt.tight_layout()
    return fig, ax, logfig, logax
