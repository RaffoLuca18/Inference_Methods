####################################################################################################
####################################################################################################
#                                                                                                  #
# importing the libraries                                                                          #
#                                                                                                  #
####################################################################################################
####################################################################################################



import numpy as np
import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
import J_inference
import J_sampler
import GGM_sampler



####################################################################################################
####################################################################################################
#                                                                                                  #
# some utilities                                                                                   #
#                                                                                                  #
####################################################################################################
####################################################################################################



def mask(J, t = 0.1):
    """ masking everything under t to zero and everything above to one """


    n_spins = J.shape[0]
    mask = (jnp.abs(J) >= t).astype(jnp.float32)

    return mask



####################################################################################################



def roc_prc(true_J, hat_J, thresholds=None):
    """ROC curve + AUC, PRC curve + AUC-PR, and PR baseline for edge selection"""

    n_spins = true_J.shape[0]

    # ground-truth edges (no self-loops, symmetrized)
    true_edges = (jnp.abs(true_J) > 1e-12).astype(jnp.float32)
    true_edges = true_edges * (1.0 - jnp.eye(n_spins))
    true_edges = jnp.maximum(true_edges, true_edges.T)

    # ---- PR baseline on upper triangle (avoid double counting) ----
    upper = jnp.triu(jnp.ones((n_spins, n_spins), dtype=jnp.float32), k=1)
    P = jnp.sum(true_edges * upper)          
    N = jnp.sum((1.0 - true_edges) * upper) 
    pr_baseline = P / jnp.maximum(P + N, 1.0)

    # thresholds grid
    if thresholds is None:
        maxval = float(jnp.max(jnp.abs(hat_J)))
        thresholds = jnp.linspace(0.0, maxval, 50)

    fpr_list, tpr_list = [], []
    precision_list, recall_list = [], []

    for t in thresholds:
        est_edges = (jnp.abs(hat_J) > t).astype(jnp.float32)
        est_edges = est_edges * (1.0 - jnp.eye(n_spins))
        est_edges = jnp.maximum(est_edges, est_edges.T)

        TP = jnp.sum((true_edges == 1) & (est_edges == 1))
        FP = jnp.sum((true_edges == 0) & (est_edges == 1))
        FN = jnp.sum((true_edges == 1) & (est_edges == 0))
        TN = jnp.sum((true_edges == 0) & (est_edges == 0))

        TPR = TP / jnp.maximum(TP + FN, 1.0)  # recall
        FPR = FP / jnp.maximum(FP + TN, 1.0)
        PREC = TP / jnp.maximum(TP + FP, 1.0)
        REC = TPR

        fpr_list.append(FPR)
        tpr_list.append(TPR)
        precision_list.append(PREC)
        recall_list.append(REC)

    # arrays
    fpr = jnp.array(fpr_list)
    tpr = jnp.array(tpr_list)
    precision = jnp.array(precision_list)
    recall = jnp.array(recall_list)

    # add ROC endpoints
    fpr = jnp.concatenate([jnp.array([0.0]), fpr, jnp.array([1.0])])
    tpr = jnp.concatenate([jnp.array([0.0]), tpr, jnp.array([1.0])])

    # add PRC endpoints (precision=1 at recall=0 by convention)
    precision = jnp.concatenate([jnp.array([1.0]), precision])
    recall = jnp.concatenate([jnp.array([0.0]), recall])

    # sort
    roc_idx = jnp.argsort(fpr)
    fpr, tpr = fpr[roc_idx], tpr[roc_idx]

    prc_idx = jnp.argsort(recall)
    recall, precision = recall[prc_idx], precision[prc_idx]

    # AUC via trapezoid
    auc_roc = np.trapezoid(tpr, fpr)
    auc_prc = np.trapezoid(precision, recall)

    # return also the PR baseline
    return fpr, tpr, auc_roc, recall, precision, auc_prc, float(pr_baseline)



####################################################################################################
####################################################################################################
#                                                                                                  #
# copmlete experiment                                                                              #
#                                                                                                  #
####################################################################################################
####################################################################################################


def complete_experiment(J, n_samples, method, use_tol=True):
    """ full experiment """

    beta_c = J_sampler.find_critical(J)

    betas_grid = np.concatenate([
        np.geomspace(beta_c/10, beta_c, 10)[:-1],
        np.geomspace(beta_c, 10*beta_c, 10)
    ])

    n_spins = int(J.shape[0])

    samples_grid = np.unique(
        np.linspace(n_spins, int(n_samples), 10, dtype=int)
    )

    grid = np.zeros((len(betas_grid), len(samples_grid)), dtype=float)
    grid_2 = np.zeros((len(betas_grid), len(samples_grid)), dtype=float)

    for i, beta in enumerate(betas_grid):
        true_J = beta * J
        h = jnp.diag(true_J)

        for j, n_samples in enumerate(samples_grid):
            samples_n = J_sampler.J_sampler(n_samples, true_J, h)
            histogram_n = J_sampler._samples_to_histogram(samples_n)

            if method in ["RISE", "logRISE", "RPLE", "MPF", "CSM", "EMHT"]:
                out, _ = J_inference.inverse_ising(method, 0.1, "Y", histogram_n)
            else:
                raise ValueError(f"Unknown method '{method}'")

            J_hat = out[0] if isinstance(out, tuple) else out
            _2, _3, grid[i, j], _a, _b, grid_2[i, j], baseline = roc_prc(true_J, J_hat)

    # indice di beta_c nella griglia
    idx_beta_c = np.argmin(np.abs(betas_grid - beta_c))

    # plotting: two heatmaps (ROC on top, PR below)
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(6.2, 9), sharex=True)

    # ROC
    im1 = axes[0].imshow(grid, origin='lower', aspect='auto', vmin=0.0, vmax=1.0)
    axes[0].set_xticks(np.arange(len(samples_grid)))
    axes[0].set_xticklabels(samples_grid)
    axes[0].set_yticks(np.arange(len(betas_grid)))
    axes[0].set_yticklabels([f"{b:.1f}" for b in betas_grid])
    axes[0].set_ylabel("beta")
    axes[0].set_title(f"AUC-ROC heatmap ({method}, use_tol={bool(use_tol)})")
    cbar1 = plt.colorbar(im1, ax=axes[0], label="AUC-ROC")
    cbar1.ax.tick_params(labelsize=10)

    # linea rossa in corrispondenza di beta_c
    axes[0].hlines(idx_beta_c, xmin=-0.5, xmax=len(samples_grid)-0.5,
                   colors="red", linewidth=2)

    for i in range(len(betas_grid)):
        for j in range(len(samples_grid)):
            axes[0].text(j, i, f"{grid[i,j]:.2f}",
                         ha="center", va="center", color="w", fontsize=7)

    # PR
    im2 = axes[1].imshow(grid_2, origin='lower', aspect='auto', vmin=0.0, vmax=1.0)
    axes[1].set_xticks(np.arange(len(samples_grid)))
    axes[1].set_xticklabels(samples_grid)
    axes[1].set_yticks(np.arange(len(betas_grid)))
    axes[1].set_yticklabels([f"{b:.1f}" for b in betas_grid])
    axes[1].set_xlabel("n_samples")
    axes[1].set_ylabel("beta")
    axes[1].set_title("AUC-PR heatmap")
    cbar2 = plt.colorbar(im2, ax=axes[1], label="AUC-PR")
    cbar2.ax.tick_params(labelsize=10)

    # baseline PR
    cbar2.ax.hlines(baseline, xmin=0.0, xmax=1.0, colors="red", linewidth=5, clip_on=False)

    # linea rossa anche qui
    axes[1].hlines(idx_beta_c, xmin=-0.5, xmax=len(samples_grid)-0.5,
                   colors="red", linewidth=2)

    for i in range(len(betas_grid)):
        for j in range(len(samples_grid)):
            axes[1].text(j, i, f"{grid_2[i,j]:.2f}",
                         ha="center", va="center", color="w", fontsize=7)

    plt.tight_layout()
    plt.show()




####################################################################################################



def complete_experiment_noise(J, n_samples, method, use_tol=True):
    """ full experiment with sign sampler """


    betas_grid = np.linspace(0.01, 3.0, 10, dtype=float)

    n_spins = int(J.shape[0])

    samples_grid = np.unique(
        np.linspace(n_spins, int(n_samples), 20, dtype=int)
    )

    grid = np.zeros((len(betas_grid), len(samples_grid)), dtype=float)

    for i, beta in enumerate(betas_grid):
        true_J = beta * J
        h = jnp.diag(true_J) * 0

        for j, n_samples in enumerate(samples_grid):
            samples_n = J_sampler.J_sampler_noise(n_samples, true_J, h)
            histogram_n = J_sampler._samples_to_histogram(samples_n)

            if method == "RISE":
                out, _ = J_inference.inverse_ising(method, 0.1, "Y", histogram_n)

            elif method == "logRISE":
                out, _ = J_inference.inverse_ising(method, 0.1, "Y", histogram_n)

            elif method == "MPF":
                out, _ = J_inference.inverse_ising(method, 0.1, "Y", histogram_n)

            elif method == "CSM":
                out, _ = J_inference.inverse_ising(method, 0.1, "Y", histogram_n)

            elif method == "EMHT":
                out, _ = J_inference.inverse_ising(method, 0.1, "Y", histogram_n)

            else:
                raise ValueError(f"Unknown method '{method}'")

            J_hat = out[0] if isinstance(out, tuple) else out
            _2, _3, grid[i, j] = roc_auc(true_J, J_hat)

    # heatmap: X = n_samples, Y = beta
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    im = ax.imshow(grid, origin='lower', aspect='auto', vmin=0.3, vmax=1.0)

    ax.set_xticks(np.arange(len(samples_grid))); ax.set_xticklabels(samples_grid)
    ax.set_yticks(np.arange(len(betas_grid)));  ax.set_yticklabels([f"{b:.1f}" for b in betas_grid])

    ax.set_xlabel("n_samples")
    ax.set_ylabel("beta")
    plt.colorbar(im, ax=ax, label="AUC")
    ax.set_title(f"AUC heatmap vs beta & n_samples ({method}, use_tol={bool(use_tol)})")
    plt.tight_layout(); plt.show()

