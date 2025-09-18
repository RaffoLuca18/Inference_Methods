####################################################################################################
####################################################################################################
#                                                                                                  #
# importing the functions                                                                          #
#                                                                                                  #
####################################################################################################
####################################################################################################



import numpy as np
import jax
import jax.numpy as jnp



####################################################################################################
####################################################################################################
#                                                                                                  #
# making the precision matrix                                                                      #
#                                                                                                  #
####################################################################################################
####################################################################################################



def precision_maker(n_spins, p = 0.2, eps = 0.01, minval = 0.5, maxval = 1.0, seed = 0):
    """ to make precision matrix from a erdos-renyi graph """

    key = jax.random.PRNGKey(seed)
    precision = jnp.zeros((n_spins, n_spins))

    for i in range(n_spins):
        for j in range(0, i):
            key, key_2, key_3 = jax.random.split(key, 3)
            u = jax.random.uniform(key_2)
            if u < p:
                rnd = jax.random.uniform(key_3, minval = minval, maxval = maxval)
                precision = precision.at[i, j].set(rnd)
                precision = precision.at[j, i].set(rnd)

    lambda_min = jnp.min(jnp.linalg.eigvalsh(precision))
    if lambda_min < eps:
        precision = precision + (eps - lambda_min) * jnp.eye(n_spins)

    return precision



####################################################################################################
####################################################################################################
#                                                                                                  #
# making the samples from the precision matrix                                                     #
#                                                                                                  #
####################################################################################################
####################################################################################################



def precision_sampler(precision, n_samples, beta = 1.0, seed = 0):
    """ to sample the data points """

    key = jax.random.PRNGKey(seed)
    precision = precision * beta
    cov = jnp.linalg.inv(precision)
    n_spins = precision.shape[0]
    mean = jnp.zeros(n_spins)
    samples = jax.random.multivariate_normal(key, mean = mean, cov = cov, shape = (n_samples,))

    return samples



####################################################################################################



def precision_sampler_sign(precision, n_samples, beta=1.0, seed=0):
    """ sample sign vectors from gaussian with precision matrix """

    key = jax.random.PRNGKey(seed)
    precision = precision * beta
    cov = jnp.linalg.inv(precision)
    n_spins = precision.shape[0]
    mean = jnp.zeros(n_spins)

    # gaussian samples
    samples = jax.random.multivariate_normal(
        key, mean=mean, cov=cov, shape=(n_samples,)
    )

    # take sign: {-1, +1}
    spin_samples = jnp.sign(samples)

    # convention: replace 0 with +1 (just in case)
    spin_samples = jnp.where(spin_samples == 0, 1, spin_samples)

    return spin_samples



####################################################################################################



def precision_sampler_landau(precision, n_samples, beta=1.0, lam_q=0.0,
                             step_size=1e-2, n_steps=1000, seed=0, record_history=False):
    """ landaun-ginzburg samplers """


    key = jax.random.PRNGKey(seed)
    precision = precision * beta
    cov = jnp.linalg.inv(precision)
    d = precision.shape[0]
    mean = jnp.zeros(d)
    key, sub = jax.random.split(key)
    x = jax.random.multivariate_normal(sub, mean=mean, cov=cov, shape=(n_samples,))
    P_eff = precision + lam_q * jnp.eye(d)
    s = step_size
    accepts = []

    if record_history:
        traj = []

    for _ in range(n_steps):
        key, sub = jax.random.split(key)
        logpi_x = -0.5 * jnp.einsum('bi,ij,bj->b', x, P_eff, x)
        g_x = -(x @ P_eff.T)
        mean_fwd = x + s * g_x
        noise = jax.random.normal(sub, shape=x.shape)
        y = mean_fwd + jnp.sqrt(2.0 * s) * noise
        logpi_y = -0.5 * jnp.einsum('bi,ij,bj->b', y, P_eff, y)
        g_y = -(y @ P_eff.T)
        mean_bwd = y + s * g_y
        diff_yx = y - mean_fwd
        diff_xy = x - mean_bwd
        log_q_y_given_x = -0.5 / (2.0 * s) * jnp.sum(diff_yx * diff_yx, axis=-1)
        log_q_x_given_y = -0.5 / (2.0 * s) * jnp.sum(diff_xy * diff_xy, axis=-1)
        log_alpha = (logpi_y - logpi_x) + (log_q_x_given_y - log_q_y_given_x)
        u = jax.random.uniform(sub, shape=(n_samples,))
        accept = (jnp.log(u + 1e-12) < log_alpha)
        x = jnp.where(accept[:, None], y, x)
        accepts.append(jnp.mean(accept.astype(jnp.float32)))
        if record_history:
            traj.append(x)

    accept_rate = float(jnp.mean(jnp.stack(accepts))) if accepts else 0.0
    info = {"accept_rate": accept_rate}

    if record_history:
        info["traj"] = jnp.stack(traj)

    #return x, info
    return x



####################################################################################################



def diag_generation(precision: jnp.ndarray,
                    n_samples: int,
                    seed: int = 0,
                    jitter: float = 1e-9) -> jnp.ndarray:
    """ to sample gaussians in the langevin matching framework """

    Theta = 0.5 * (precision + precision.T)

    eigvals, eigvecs = jnp.linalg.eigh(Theta)

    eigvals_clipped = jnp.clip(eigvals, a_min=jitter)

    inv_sqrt = 1.0 / jnp.sqrt(eigvals_clipped)
    A = eigvecs * inv_sqrt[None, :]     # same as eigvecs @ diag(inv_sqrt)

    key = jax.random.PRNGKey(seed)
    z = jax.random.normal(key, shape=(n_samples, precision.shape[0]))
    samples = z @ A.T


    return samples
