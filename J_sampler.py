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
# some utilities                                                                                   #
#                                                                                                  #
####################################################################################################
####################################################################################################



def _int_to_spin(ints, n):
    """ convert integers (0…2^n - 1) to {-1,+1} spin vectors of length n """


    bits = ((ints[:, None] >> jnp.arange(n)) & 1)
    
    return 2 * bits - 1  # 0→‑1, 1→+1



####################################################################################################



def _spin_to_int(spins):
    """ does the opposite of the previous function """


    n = spins.shape[1]
    # map: -1 -> 0, +1 -> 1
    bits = (spins > 0).astype(jnp.uint32)  # (B,n)
    weights = (1 << jnp.arange(n, dtype=jnp.uint32))  # (n,)
    ids = (bits * weights).sum(axis=1, dtype=jnp.uint32)

    return ids



####################################################################################################



def _samples_to_histogram(samples):
    """ build the histogram from the samples """


    n = samples.shape[1]
    ids = _spin_to_int(samples)                               # (B,)
    counts = jnp.bincount(ids, length=(1 << n))              # (2^n,)
    nonzero = jnp.nonzero(counts)[0]                         # (K,)
    counts_nz = counts[nonzero][:, None]                     # (K,1)
    spins_nz = _int_to_spin(nonzero, n)                       # (K,n)
    hist = jnp.concatenate([counts_nz, spins_nz], axis=1)    # (K, n+1)

    return np.asarray(hist, dtype=np.int64)



####################################################################################################



def _histogram_to_samples(histogram):
    """ reconstruct the samples from the histogram """


    H = jnp.asarray(histogram)
    counts = H[:, 0].astype(jnp.int32)
    spins = H[:, 1:].astype(jnp.int32)        # (-1,+1)
    idx = jnp.arange(spins.shape[0], dtype=jnp.int32)
    idx_rep = jnp.repeat(idx, counts)         # (B,)
    samples = spins[idx_rep]                  # (B,n)

    return samples



####################################################################################################



def _histogram_to_freq_configs(histogram):
    """ TO REVIEW """


    # convert histogram to (freq, configs) without expanding samples
    H = jnp.asarray(histogram)
    freq = H[:, 0].astype(jnp.int32)        # (K,)
    configs = H[:, 1:].astype(jnp.int32)    # (K, n)
    
    return freq, configs



####################################################################################################



def find_critical(J):
    """" to find the ecritical beta, for a given J, and supposing that h = 0 """


    lam_max = jnp.max(jnp.linalg.eigvalsh(J))

    beta_c = 1/lam_max

    return beta_c



####################################################################################################
####################################################################################################
#                                                                                                  #
# making the J matrix                                                                              #
#                                                                                                  #
####################################################################################################
####################################################################################################



def J_maker(n_spins, p = 0.2, minval = -1.0, maxval = 1.0, seed = 0):
    """ making the J matrix from a erdos-renyi graph structure """


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

    return precision



####################################################################################################



def J_maker_01(n_spins, p = 0.2, seed = 0):
    """ only 0-1 entries in the J matrix """


    key = jax.random.PRNGKey(seed)
    J = jnp.zeros((n_spins, n_spins))
    for i in range(n_spins):
        for j in range(0, i):
            key, key_2 = jax.random.split(key, 2)
            u = jax.random.uniform(key_2)
            if u < p:
                J = J.at[i, j].set(1.0)
                J = J.at[j, i].set(1.0)

    return J



####################################################################################################
####################################################################################################
#                                                                                                  #
# sampling from the J matrix                                                                       #
#                                                                                                  #
####################################################################################################
####################################################################################################



def J_sampler(n_samples, J, h, seed = 0):
    """ sample from the boltzmann distribution with mean zero and prescribed couplings matrix J """
    

    key = jax.random.PRNGKey(seed)
    n_spins = J.shape[0]
    J = J * (1 - jnp.eye(n_spins))

    configs = jnp.arange(2 ** n_spins, dtype=jnp.uint32)
    spins   = _int_to_spin(configs, n_spins)

    energies = 0.5 * jnp.einsum('bi,ij,bj->b', spins, J, spins) + spins @ h
    logw = energies - jnp.max(energies)
    probs = jnp.exp(logw) / jnp.sum(jnp.exp(logw))

    sampled_configs = jax.random.choice(key, configs, shape=(n_samples,), p=probs, replace=True)
    sampled_spins   = _int_to_spin(sampled_configs, n_spins)

    return sampled_spins



####################################################################################################



def J_sampler_noise(n_samples, J, h, p = 0.1, seed = 0):
    """ J is made of zeros and ones only. At each batch: independent flip with prob. p of every off-diagonal entry of J (0<->1),
    keeping symmetry and diag = 0; then sample n_samples from e^{-e(s)} by enumeration """


    key = jax.random.PRNGKey(seed)
    n = J.shape[0]

    batches = n_samples
    J = (J > 0).astype(jnp.int8)
    J = jnp.triu(J, 1)
    J = (J + J.T).astype(jnp.int8)

    configs = jnp.arange(2 ** n, dtype=jnp.uint32)
    spins   = _int_to_spin(configs, n)

    for _ in range(batches):
        key, kmask = jax.random.split(key)
        mask_up = jax.random.bernoulli(kmask, p=p, shape=(n, n))
        mask_up = jnp.triu(mask_up, 1)

        J_up = jnp.triu(J, 1)
        J_up_flipped = jnp.bitwise_xor(J_up, mask_up.astype(J_up.dtype))

        J = (J_up_flipped + J_up_flipped.T).astype(jnp.int8)

        samples = J_sampler(n_samples, J, h)

    return samples