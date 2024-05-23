from matplotlib import pyplot as plt
import numpy as np
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_median_mean(prob_t, prob_l, plot_all_curves=False, save_path=None, title='', y_label='Probability'):
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(20, 5), sharex=True)

    # Original scale subplot
    if plot_all_curves:
        alpha = prob_t.shape[1]/42600.0
        ax1.plot(prob_t, color='tab:blue', alpha=alpha)
        ax1.plot(prob_l, color='tab:orange', alpha=alpha)
    ax1.plot(prob_t.median(axis=1).values, color='tab:blue', label='truth median')
    ax1.plot(prob_l.median(axis=1).values, color='tab:orange', label='lie median')
    ax1.plot(prob_t.mean(axis=1), color='tab:blue', label='truth mean', linestyle='--')
    ax1.plot(prob_l.mean(axis=1), color='tab:orange', label='lie mean', linestyle='--')
    ax1.grid()
    ax1.set_xlabel("Layer")
    ax1.set_ylabel(y_label)
    ax1.set_title(title + ' (Linear Scale)')
    ax1.legend()

    # Log scale subplot
    if plot_all_curves:
        ax2.plot(prob_t, color='tab:blue', alpha=alpha)
        ax2.plot(prob_l, color='tab:orange', alpha=alpha)
    ax2.plot(prob_t.median(axis=1).values, color='tab:blue', label='truth median')
    ax2.plot(prob_l.median(axis=1).values, color='tab:orange', label='lie median')
    ax2.plot(prob_t.mean(axis=1), color='tab:blue', label='truth mean', linestyle='--')
    ax2.plot(prob_l.mean(axis=1), color='tab:orange', label='lie mean', linestyle='--')
    ax2.set_yscale('log')
    ax2.grid()
    ax2.set_xlabel("Layer")
    ax2.set_ylabel(y_label)
    ax2.set_title(title + ' (Log Scale)')
    ax2.legend()

    # Save figure if path provided
    if save_path:
        fig.savefig(save_path)

    plt.show()


def plot_h_bar(prob_truth, prob_lie, selected_layers, title, y_label="top tokens"):
    width = 0.5
    k = prob_truth.shape[0]
    fig, axs = plt.subplots(1, len(selected_layers), figsize=(len(selected_layers)*2.5, 5))

    prob_truth_means, prob_truth_medians = prob_truth.mean(dim=-1), prob_truth.median(dim=-1).values
    prob_lie_means, prob_lie_medians = prob_lie.mean(dim=-1), prob_lie.median(dim=-1).values

    for i, l in enumerate(selected_layers):
        y = np.arange(k)
        axs[i].barh(y - width/2, prob_truth_medians[:, l], height=width/3, color='tab:blue', align='center', label='Truth median', edgecolor='black')
        axs[i].barh(y - width/4, prob_truth_means[:, l], height=width/3, color='tab:blue', align='center', label='Truth mean',hatch='//', edgecolor='black')
        axs[i].barh(y + width/4, prob_lie_medians[:, l], height=width/3, color='tab:orange', align='center', label='Lie median', edgecolor='black')
        axs[i].barh(y + width/2, prob_lie_means[:, l], height=width/3, color='tab:orange', align='center', label='Lie mean', hatch = '//', edgecolor='black')
        axs[i].grid('off')
        axs[i].set_yticks(np.arange(k))
        axs[i].set_yticklabels([])
        if i == 0:
            axs[i].set_ylabel(y_label)
            axs[i].set_yticklabels(np.arange(1, k+1).astype(int))
        if i ==  len(selected_layers)-1:
            axs[i].legend(loc='best')
        axs[i].set_xlabel(f'\nlayer_id: {l}')

    fig.align_labels()
    fig.suptitle(title)
    plt.show()


def plot_distance_matrix(truth_token_dist, lie_token_dist, sub_titles=['truth tokens', 'lie tokens'], sup_title="Pairwise distances", norm=None):

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    truth_token_dist.fill_diagonal_(float('nan')).numpy()
    lie_token_dist.fill_diagonal_(float('nan')).numpy()

    if norm is not None:
        vmin = min(np.nanmin(truth_token_dist), np.nanmin(lie_token_dist))
        vmax = max(np.nanmax(truth_token_dist), np.nanmax(lie_token_dist))
        norm = colors.LogNorm(vmin=vmin, vmax=vmax)

    # Function to plot with colorbars
    def plot_matrix(ax, data, title, norm):
        im = ax.imshow(data, cmap='viridis', norm=norm)
        ax.set_title(title)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

    # Plot first distance matrix
    plot_matrix(axes[0], truth_token_dist, sub_titles[0], norm)
    # Plot average distance matrix
    plot_matrix(axes[1], lie_token_dist, sub_titles[1], norm)

    fig.suptitle(sup_title)

    plt.tight_layout()
    plt.show()