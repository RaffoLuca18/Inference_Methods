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
import GGM_inference
import GGM_sampler



####################################################################################################
####################################################################################################
#                                                                                                  #
# some utilities                                                                                   #
#                                                                                                  #
####################################################################################################
####################################################################################################



def mask(precision, t = 0.1):
    """ masking everything under t to zero and everything above to one """


    n_spins = precision.shape[0]
    mask = (jnp.abs(precision) >= t).astype(jnp.float32)
    mask = mask * (1.0 - jnp.eye(n_spins))
    mask = mask + jnp.eye(n_spins)

    return mask



####################################################################################################



def roc_prc(true_precision, hat_precision, thresholds=None):
    """ROC curve + AUC, PRC curve + AUC-PR, and PR baseline for edge selection"""

    n_spins = true_precision.shape[0]

    # ground-truth edges (no self-loops, symmetrized)
    true_edges = (jnp.abs(true_precision) > 1e-12).astype(jnp.float32)
    true_edges = true_edges * (1.0 - jnp.eye(n_spins))
    true_edges = jnp.maximum(true_edges, true_edges.T)

    # ---- PR baseline on upper triangle (avoid double counting) ----
    upper = jnp.triu(jnp.ones((n_spins, n_spins), dtype=jnp.float32), k=1)
    P = jnp.sum(true_edges * upper)          
    N = jnp.sum((1.0 - true_edges) * upper) 
    pr_baseline = P / jnp.maximum(P + N, 1.0)

    # thresholds grid
    if thresholds is None:
        maxval = float(jnp.max(jnp.abs(hat_precision)))
        thresholds = jnp.linspace(0.0, maxval, 50)

    fpr_list, tpr_list = [], []
    precision_list, recall_list = [], []

    for t in thresholds:
        est_edges = (jnp.abs(hat_precision) > t).astype(jnp.float32)
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



def complete_experiment(precision, n_samples, method, use_tol=True):
    """ full experiment """

    betas_grid = np.linspace(0.1, 3.0, 15, dtype=float)
    n_spins = int(precision.shape[0])

    samples_grid = np.unique(
        np.linspace(n_spins, int(n_samples), 15, dtype=int)
    )

    grid_auc_roc = np.zeros((len(betas_grid), len(samples_grid)), dtype=float)
    grid_auc_pr = np.zeros((len(betas_grid), len(samples_grid)), dtype=float)

    for i, beta in enumerate(betas_grid):
        true_precision = beta * precision

        for j, n_samples in enumerate(samples_grid):
            samples_n = GGM_sampler.precision_sampler(true_precision, n_samples)

            if method == "mle":
                out = GGM_inference.naive_mle(samples_n)
            elif method == "graphical_lasso":
                out = GGM_inference.graphical_lasso(samples_n)
            elif method == "graphical_score_matching":
                out = GGM_inference.graphical_score_matching(samples_n)
            elif method == "clime":
                out = GGM_inference.clime(samples_n)
            else:
                raise ValueError(f"Unknown method '{method}'")

            precision_hat = out[0] if isinstance(out, tuple) else out
            _, _, grid_auc_roc[i, j], _, _, grid_auc_pr[i, j], baseline = roc_prc(true_precision, precision_hat)

    # plotting: two heatmaps (ROC on top, PR below)
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(6.2, 9), sharex=True)

    # ROC
    im1 = axes[0].imshow(grid_auc_roc, origin='lower', aspect='auto', vmin=0.0, vmax=1.0)
    axes[0].set_xticks(np.arange(len(samples_grid)))
    axes[0].set_xticklabels(samples_grid)
    axes[0].set_yticks(np.arange(len(betas_grid)))
    axes[0].set_yticklabels([f"{b:.1f}" for b in betas_grid])
    axes[0].set_ylabel("beta")
    axes[0].set_title(f"AUC-ROC heatmap ({method}, use_tol={bool(use_tol)})")
    midbar = plt.colorbar(im1, ax=axes[0], label="AUC-ROC")
    midbar.ax.tick_params(labelsize=10)

    midbar.ax.hlines(0.5, xmin=0.0, xmax=1.0, colors="red", linewidth=5, clip_on=False)

    for i in range(len(betas_grid)):
        for j in range(len(samples_grid)):
            axes[0].text(j, i, f"{grid_auc_roc[i,j]:.2f}",
                        ha="center", va="center", color="w", fontsize=7)

    # PR
    im2 = axes[1].imshow(grid_auc_pr, origin='lower', aspect='auto', vmin=0.0, vmax=1.0)
    axes[1].set_xticks(np.arange(len(samples_grid)))
    axes[1].set_xticklabels(samples_grid)
    axes[1].set_yticks(np.arange(len(betas_grid)))
    axes[1].set_yticklabels([f"{b:.1f}" for b in betas_grid])
    axes[1].set_xlabel("n_samples")
    axes[1].set_ylabel("beta")
    axes[1].set_title("AUC-PR heatmap")

    cbar = plt.colorbar(im2, ax=axes[1], label="AUC-PR")
    cbar.ax.tick_params(labelsize=10)

    cbar.ax.hlines(baseline, xmin=0.0, xmax=1.0, colors="red", linewidth=5, clip_on=False)

    for i in range(len(betas_grid)):
        for j in range(len(samples_grid)):
            axes[1].text(j, i, f"{grid_auc_pr[i,j]:.2f}",
                        ha="center", va="center", color="w", fontsize=7)

    plt.tight_layout()
    plt.show()
