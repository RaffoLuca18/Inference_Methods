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
    mask = mask * (1.0 - jnp.eye(n_spins))
    mask = mask + jnp.eye(n_spins)

    return mask



####################################################################################################



def roc_auc(true_J, hat_J, thresholds=None):
    """ ROC curve + AUC for edge selection """


    n_spins = true_J.shape[0]

    true_edges = (jnp.abs(true_J) > 1e-12).astype(jnp.float32)
    true_edges = true_edges * (1.0 - jnp.eye(n_spins))
    true_edges = jnp.maximum(true_edges, true_edges.T)

    if thresholds is None:
        maxval = float(jnp.max(jnp.abs(hat_J)))
        thresholds = jnp.linspace(0.0, maxval, 50)

    fpr_list = []
    tpr_list = []

    for t in thresholds:
        est_edges = mask(hat_J, t=t)

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



def complete_experiment(J, n_samples, method, use_tol=True):
    """ full experiment """


    betas_grid = np.linspace(0.01, 3.0, 10, dtype=float)

    n_spins = int(J.shape[0])

    samples_grid = np.unique(
        np.linspace(n_spins, int(n_samples), 10, dtype=int)
    )

    grid = np.zeros((len(betas_grid), len(samples_grid)), dtype=float)

    for i, beta in enumerate(betas_grid):
        true_J = beta * J
        h = jnp.diag(true_J)

        for j, n_samples in enumerate(samples_grid):
            samples_n = J_sampler.J_sampler(n_samples, true_J, h)
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



####################################################################################################



def complete_experiment_sign(prec, n_samples, method, use_tol=True):
    """ full experiment with sign sampler"""


    betas_grid = np.linspace(0.01, 3.0, 10, dtype=float)

    n_spins = int(prec.shape[0])

    samples_grid = np.unique(
        np.linspace(n_spins, int(n_samples), 10, dtype=int)
    )

    grid = np.zeros((len(betas_grid), len(samples_grid)), dtype=float)

    for i, beta in enumerate(betas_grid):
        true_prec = beta * prec
        h = jnp.diag(true_prec) * 0

        for j, n_samples in enumerate(samples_grid):
            samples_n = GGM_sampler.precision_sampler_sign(true_prec, n_samples)
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

            prec_hat = out[0] if isinstance(out, tuple) else out
            _2, _3, grid[i, j] = roc_auc(true_prec, prec_hat)

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



####################################################################################################



def complete_experiment_landau(prec, n_samples, method, use_tol=True):
    """ full experiment with sign sampler"""


    betas_grid = np.linspace(0.01, 3.0, 10, dtype=float)

    n_spins = int(prec.shape[0])

    samples_grid = np.unique(
        np.linspace(n_spins, int(n_samples), 20, dtype=int)
    )

    grid = np.zeros((len(betas_grid), len(samples_grid)), dtype=float)

    for i, beta in enumerate(betas_grid):
        true_prec = beta * prec
        h = jnp.diag(true_prec) * 0

        for j, n_samples in enumerate(samples_grid):
            samples_n = GGM_sampler.precision_sampler_landau(true_prec, n_samples)
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

            prec_hat = out[0] if isinstance(out, tuple) else out
            _2, _3, grid[i, j] = roc_auc(true_prec, prec_hat)

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