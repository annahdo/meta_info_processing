from matplotlib import pyplot as plt


def plot_prob(prob_t, prob_l, plot_all_curves=False, save_path=None, title='', y_label='Probability'):
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
