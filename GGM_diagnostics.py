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



def roc_auc(true_precision, hat_precision, thresholds=None):
    """ ROC curve + AUC for edge selection """


    n_spins = true_precision.shape[0]

    true_edges = (jnp.abs(true_precision) > 1e-12).astype(jnp.float32)
    true_edges = true_edges * (1.0 - jnp.eye(n_spins))
    true_edges = jnp.maximum(true_edges, true_edges.T)

    if thresholds is None:
        maxval = float(jnp.max(jnp.abs(hat_precision)))
        thresholds = jnp.linspace(0.0, maxval, 50)

    fpr_list = []
    tpr_list = []

    for t in thresholds:
        est_edges = mask(hat_precision, t=t)

        TP = jnp.sum((true_edges == 1) & (est_edges == 1))
        FP = jnp.sum((true_edges == 0) & (est_edges == 1))
        FN = jnp.sum((true_edges == 1) & (est_edges == 0))
        TN = jnp.sum((true_edges == 0) & (est_edges == 0))

        TPR = TP / jnp.maximum(TP + FN, 1.0)
        FPR = FP / jnp.maximum(FP + TN, 1.0)

        fpr_list.append(FPR)
        tpr_list.append(TPR)

    fpr = jnp.array(fpr_list)
    tpr = jnp.array(tpr_list)

    fpr = jnp.concatenate([jnp.array([0.0]), fpr, jnp.array([1.0])])
    tpr = jnp.concatenate([jnp.array([0.0]), tpr, jnp.array([1.0])])

    idx = jnp.argsort(fpr)

    fpr = fpr[idx]
    tpr = tpr[idx]

    auc = np.trapezoid(tpr, fpr)

    return fpr, tpr, auc



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

    grid = np.zeros((len(betas_grid), len(samples_grid)), dtype=float)

    for i, beta in enumerate(betas_grid):
        true_precision = beta * precision

        for j, n_samples in enumerate(samples_grid):
            samples_n = GGM_sampler.precision_sampler(true_precision, n_samples)

            if method == "MPF":
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
            _, _2, grid[i, j] = roc_auc(true_precision, precision_hat)

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