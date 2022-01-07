from typing import  List
import numpy as np
import matplotlib.pyplot as plt
import os


def plot_V(Gamma: List) -> None:
    """
    Plot the value function represented by a set of alpha vectors (assuming alpha vectors are 2-dimensional

    :param Gamma: the set of alpha vectors
    :return: None
    """
    belief_space = np.arange(0, 1, 0.01)
    y = []
    for b0 in belief_space:
        max_v = -np.inf
        for alpha in Gamma:
            v = np.dot(np.array(alpha), np.array([b0, 1 - b0]))

            if v > max_v:
                max_v = v
        y.append(max_v)

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    labelsize = 9
    fontsize = 10
    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{amsfonts,amsmath}')
    plt.rcParams['font.family'] = ['serif']
    plt.rcParams['axes.titlepad'] = 50
    plt.rcParams['ytick.major.pad'] = 0.05
    plt.rcParams['axes.labelpad'] = 2
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams.update({'font.size': fontsize})

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 2.5))

    colors = plt.cm.viridis(np.linspace(0.3, 1, 2))[-2:]
    markevery = 5

    ax.plot(belief_space, y, label=r"$V(b)$", ls='-', color=colors[0], marker="s",
            markevery=markevery, markersize=2.5)

    # ax[0].set_xlabel(r"$t$", fontsize=labelsize)
    ax.set_ylabel(r"$V(b)$", fontsize=labelsize)
    ax.set_xlabel(r"$b(s_1)$", fontsize=labelsize)

    # tweak the axis labels
    xlab = ax.xaxis.get_label()
    ylab = ax.yaxis.get_label()

    xlab.set_size(labelsize)
    ylab.set_size(labelsize)

    # change the color of the top and right spines to opaque gray
    ax.spines['right'].set_color((.8, .8, .8))
    ax.spines['top'].set_color((.8, .8, .8))
    ax.tick_params(axis='both', which='major', labelsize=labelsize, length=2.2, width=0.6)
    ax.tick_params(axis='both', which='minor', labelsize=labelsize, length=2.2, width=0.6)

    fig.suptitle(r"Piece-wise Linear Convex Optimal Value Function $V^{*}$ for Belief-MDP", fontsize=fontsize, y=0.95,
                 x=0.55)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.505, 0.165),
               ncol=3, fancybox=True, shadow=True)

    fig.tight_layout()
    # plt.show()

    fig.subplots_adjust(wspace=0.1, hspace=0.25, bottom=0.285)
    fig.savefig("value_fun" + ".png", format="png", dpi=600)
    fig.savefig("value_fun" + ".pdf", format='pdf', dpi=600, bbox_inches='tight', transparent=True)

