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
import pulp
import math
from typing import Optional, Tuple, Literal, Union, Sequence, Dict, Any

import J_sampler



####################################################################################################
####################################################################################################
#                                                                                                  #
# some utilities                                                                                   #
#                                                                                                  #
####################################################################################################
####################################################################################################



def _compute_lambda(alpha, n_spins, n_samples):
    """ compute the regularization strength lambda from the user-supplied coefficient alpha """


    return alpha * math.sqrt(math.log((n_spins ** 2) / 0.05) / n_samples)



####################################################################################################
####################################################################################################
#                                                                                                  #
# loss functions                                                                                   #
#                                                                                                  #
####################################################################################################
####################################################################################################



def _rise_loss(h):
    """
    rise loss: exp(-h)
    """


    return jnp.exp(-h)



####################################################################################################



def _logrise_loss(h):
    """
    log-rise loss: same exponential inside, the log is taken
    in the caller for numerical stability
    """


    return jnp.exp(-h)



####################################################################################################




def _rple_loss(h):
    """
    rple loss: log(1 + exp(-2h))
    """

    
    return jnp.log1p(jnp.exp(-2.0 * h))



####################################################################################################



def _mpf_loss(h):
    """
    l_mpf ∝ exp(-ΔE / 2) with ΔE = 2h in the node-centric notation,
    hence exp(-h)
    """


    return jnp.exp(-h)  # equivalent to rise when written per‑node



####################################################################################################



def _csm_loss(h):
    """
    l_csm ∝ exp(-2ΔE) - 2 exp(+ΔE) + 1
    con ΔE = 2h, quindi exp(-4h) - 2 exp(2h) + 1
    """


    return jnp.exp(-4.0 * h) - 2.0 * jnp.exp(2.0 * h)



####################################################################################################
####################################################################################################
#                                                                                                  #
# reconstruct single spin, no emht                                                                 #
#                                                                                                  #
####################################################################################################
####################################################################################################



def _reconstruct_single_spin(s, freq, configs, method, lam, adj_row: Optional[jnp.ndarray],
                            n_steps = 500, lr = 1e-2, record_history = True):
    """
    reconstructs row w_{s,·} for spin s
    returns (w_full_final, history) where:
      history[k] = w_full (np.ndarray, shape (num_spins,)) after k steps
    """

    # get counts and dimensionality
    num_conf, num_spins = configs.shape
    n_samples = freq.sum()

    # batch_sise
    BATCH_SIZE = min(500, num_conf)
    REPLACE = True
    key = jax.random.PRNGKey(int(s))

    # extract target spin column
    y = configs[:, s]

    # build nodal statistics y * x with self term kept as y
    nodal_stat_full = (y[:, None] * configs).at[:, s].set(y).astype(jnp.float32)

    # l1 mask: penalize all except the self index
    l1_mask = jnp.ones(num_spins, dtype=jnp.float32).at[s].set(0.0)

    # zero mask from adjacency: forbid edges where adj is zero
    zero_mask = (
        (adj_row == 0) & (jnp.arange(num_spins) != s)
        if adj_row is not None
        else jnp.zeros(num_spins, dtype=bool)
    )

    # free indices are those not hard-zeroed
    free_idx = jnp.where(~zero_mask)[0]

    # l1 mask restricted to free parameters
    l1_mask_free = l1_mask[free_idx]

    # to sample batches
    probs = (freq / n_samples).astype(jnp.float32)
    is_logrise = (method == "logRISE")

    # loss per sample
    def per_sample_loss(w_free, nodal_stat_batch):
        w_full = jnp.zeros(num_spins, dtype=jnp.float32).at[free_idx].set(w_free)
        h = nodal_stat_batch @ w_full
        if method == "RISE":
            return _rise_loss(h)
        elif method == "logRISE":
            return _logrise_loss(h)
        elif method == "RPLE":
            return _rple_loss(h)
        elif method == "MPF":
            return _mpf_loss(h)
        elif method == "CSM":
            return _csm_loss(h)
        else:
            raise ValueError(f"unknown method: {method}")

    # loss on batches
    def batch_loss(w_free, nodal_stat_batch):
        vals = per_sample_loss(w_free, nodal_stat_batch)  # shape (batch,)
        if is_logrise:
            return jnp.log(jnp.mean(vals) + 1e-12)
        else:
            return jnp.mean(vals)

    # composite objective = smooth(batch) + l1 su free coords
    def objective_on_batch(w_free, nodal_stat_batch):
        smooth = batch_loss(w_free, nodal_stat_batch)
        return smooth + lam * jnp.sum(l1_mask_free * jnp.abs(w_free))

    # initialize free params to zero
    params = jnp.zeros((free_idx.size,), dtype=jnp.float32)

    # optimizer
    optimizer = optax.sgd(learning_rate = lr)
    opt_state = optimizer.init(params)

    # optionally record iterates
    history = []

    # loop
    for t in range(1, n_steps + 1):
        key, k2 = jax.random.split(key)
        idx = jax.random.choice(k2, num_conf, shape=(BATCH_SIZE,), p=probs, replace=REPLACE)
        nodal_stat_batch = nodal_stat_full[idx, :]

        # value and gradient su mini-batch
        val, grads = jax.value_and_grad(objective_on_batch)(params, nodal_stat_batch)

        # update
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        # reconstruction and adjacency matrix
        w_full = jnp.zeros(num_spins, dtype=jnp.float32).at[free_idx].set(params)
        if adj_row is not None:
            w_full = w_full.at[zero_mask].set(0.0)

        # save current iterate if requested
        if record_history:
            history.append(np.asarray(w_full))

    # final full vector (post loop) with hard zeros
    w_full_final = jnp.zeros(num_spins, dtype=jnp.float32).at[free_idx].set(params)
    if adj_row is not None:
        w_full_final = w_full_final.at[zero_mask].set(0.0)

    return w_full_final, history



####################################################################################################
####################################################################################################
#                                                                                                  #
# reconstruct single spin, emht                                                                    #
#                                                                                                  #
####################################################################################################
####################################################################################################



def _reconstruct_single_spin_em(s, freq, configs, eps = 0.01, lam = 0.1, adj_row: Optional[jnp.ndarray] = None,
                                n_steps = 500, lr = 0.1, record_history = True):
    """
    em single-spin: returns (w_final, history_list)
    history_list[k] = w_full after k steps
    """


    # sizes and counts
    num_conf, num_spins = configs.shape
    n_samples = freq.sum()

    # target spin column
    y = configs[:, s]

    # nodewise statistics: y * x, self-term equals y
    nodal_stat = (y[:, None] * configs).at[:, s].set(y).astype(jnp.float32)

    # hard-zero mask from adjacency (exclude self)
    zero_mask = (
        (adj_row == 0) & (jnp.arange(num_spins) != s)
        if adj_row is not None
        else jnp.zeros(num_spins, dtype=bool)
    )

    # initialize weights to zero
    w_full = jnp.zeros(num_spins, dtype=jnp.float32)

    # l1 proximal operator on off-diagonal entries
    def prox_l1_offdiag(w, tau):
        # keep self untouched
        off = w.at[s].set(0.0)
        # soft-threshold off-diagonal
        shrunk = jnp.sign(off) * jnp.maximum(jnp.abs(off) - tau, 0.0)
        # restore self
        return shrunk.at[s].set(w[s])

    # adam optimizer
    optimizer = optax.adam(learning_rate=lr)
    opt_state = optimizer.init(w_full)

    # optional trajectory
    history = []

    # em-style updates
    for t in range(1, n_steps + 1):
        # local field h_s(x) = stats · w
        h = nodal_stat @ w_full

        # e-step: responsibilities (tempered by eps)
        local_logw = -(1.0 - eps) * h  # lower is better
        r_unnorm = (freq / n_samples) * jnp.exp(local_logw - jnp.max(local_logw))  # stabilize
        r = r_unnorm / (r_unnorm.sum() + 1e-12)  # normalize

        # data moments under r
        data_m_s = (r * y).sum()                  # first moment for node s
        data_C_sj = (r[:, None] * nodal_stat).sum(axis=0)  # cross terms

        # model moments (simple linearized form)
        model_m_s = eps * w_full[s]
        model_C_sj = eps * w_full

        # gradient of m-step objective
        grad = jnp.zeros_like(w_full)
        grad = grad.at[s].set(data_m_s - model_m_s)  # self term
        grad = grad + (data_C_sj - model_C_sj) * (jnp.arange(num_spins) != s)  # off-diag

        # adam step (maximize, hence minus grad in update call)
        updates, opt_state = optimizer.update(-grad, opt_state, w_full)
        w_full = optax.apply_updates(w_full, updates)

        # l1 prox on off-diagonals if requested
        if lam > 0.0:
            w_full = prox_l1_offdiag(w_full, tau=lr * lam)

        # enforce hard zeros from adjacency
        if adj_row is not None:
            w_full = w_full.at[zero_mask].set(0.0)

        # record iterate
        if record_history:
            history.append(np.asarray(w_full))

    # return final weights and history
    return w_full, history



####################################################################################################
####################################################################################################
#                                                                                                  #
# inverse ising                                                                                    #
#                                                                                                  #
####################################################################################################
####################################################################################################



def _read_adjacency(adjacency_path, num_spins):
    """ to be implemented """


    return None



####################################################################################################



def inverse_ising(method: str, regularizing_value, symmetrization, histogram,
                adjacency_path = None, n_steps = 500, eps = 0.1, lr = 1e-2, record_history = False):
    """ returns (W_np_finale, history) """


    method = method.strip()
    symmetrization = symmetrization.strip().upper()

    freq, configs = J_sampler._histogram_to_freq_configs(histogram)
    num_conf, num_spins = configs.shape
    num_samples = float(freq.sum())
    n_samples = freq.sum()

    adj = None
    if adjacency_path is not None:
        adj = _read_adjacency(adjacency_path, num_spins)

    lam = _compute_lambda(regularizing_value, num_spins, num_samples)
    print(f"λ = {lam:.5g}  (reg = {regularizing_value})")


    W_snapshots: Dict[int, np.ndarray] = {}

    rows = []

    if method == "EMHT":
        for s in range(num_spins):
            print(f"[{s+1}/{num_spins}] reconstruction spin {s}")
            adj_row = adj[s] if adj is not None else None
            w_row_final, hist_row = _reconstruct_single_spin_em(
                s, freq, configs, eps, lam, adj_row,
                n_steps=n_steps, lr=lr,
                record_history=record_history,
            )
            rows.append(w_row_final)

            if record_history:
                for step, w_vec in enumerate(hist_row):
                    W_snapshots[step][s, :] = w_vec

        W = jnp.stack(rows)

    else:
        for s in range(num_spins):
            print(f"[{s+1}/{num_spins}] reconstruction spin {s}")
            adj_row = adj[s] if adj is not None else None
            w_row_final, hist_row = _reconstruct_single_spin(
                s, freq, configs, method, lam, adj_row,
                n_steps=n_steps, lr=lr,
                record_history=record_history,
            )
            rows.append(w_row_final)

            if record_history:
                for step, w_vec in enumerate(hist_row):
                    if step not in W_snapshots:
                        W_snapshots[step] = np.zeros((num_spins, num_spins), dtype=np.float32)
                    W_snapshots[step][s, :] = np.asarray(w_vec, dtype=np.float32)

        W = jnp.stack(rows)

    if symmetrization == "Y":
        W = 0.5 * (W + W.T)
        if record_history:
            for k in list(W_snapshots.keys()):
                W_snapshots[k] = 0.5 * (W_snapshots[k] + W_snapshots[k].T)

    W_np = np.asarray(W)

    history = {int(k): np.asarray(v) for k, v in W_snapshots.items()}
    return W_np, history

    