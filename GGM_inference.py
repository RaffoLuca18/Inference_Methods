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
from sklearn.covariance import graphical_lasso as skl_graphical_lasso

import GGM_sampler



####################################################################################################
####################################################################################################
#                                                                                                  #
# some utilities                                                                                   #
#                                                                                                  #
####################################################################################################
####################################################################################################



def _soft_abs(x, tau = 1e-12):
    """ to make the l1 loss differentiable in zero """


    abs = jnp.sqrt(x * x + tau * tau)

    return abs



####################################################################################################



def _symmetrize(matrix):
    """ to symmetrize a matrix """


    return 0.5 * (matrix + matrix.T)



####################################################################################################



def _l1_offdiag(matrix, lam, tau):
    """ to calculate the l1 of a matrix, without considering the diagonal """


    n_spins = len(matrix[0])
    mask = 1.0 - jnp.eye(n_spins)

    return lam * jnp.sum(_soft_abs(matrix * mask, tau))



####################################################################################################



def _project_to_pd(matrix, eps=1e-9):
    """ take a matrix and project it onto the space of pd matrices """


    matrix = _symmetrize(matrix)
    lam_min = jnp.linalg.eigvalsh(matrix).min()
    delta = jnp.maximum(0.0, eps - lam_min)

    return matrix + delta * jnp.eye(matrix.shape[0], dtype=matrix.dtype)



####################################################################################################



def _to_np(x):
    """ make the list an np array """


    try:
        return np.asarray(x)
    except Exception:
        return x
    


####################################################################################################



def _to_jnp(x):
    """ make the list a jax.numpy array """


    return jnp.asarray(x)



####################################################################################################
####################################################################################################
#                                                                                                  #
# naive mle                                                                                        #
#                                                                                                  #
####################################################################################################
####################################################################################################



def naive_mle(samples):
    """ the complete algorithm of the naive MLE estimator """


    n_samples = len(samples)

    mean = jnp.mean(samples, axis=0)
    centered_samples = samples - mean
    empirical_covariance = (centered_samples.T @ centered_samples) / n_samples

    hat_precision = jnp.linalg.inv(empirical_covariance)

    return hat_precision



####################################################################################################
####################################################################################################
#                                                                                                  #
# graphical lasso                                                                                  #
#                                                                                                  #
####################################################################################################
####################################################################################################



def _glasso_loss(precision, empirical_covariance, lam=0.1, tau=1e-9):
    """ graphical lasso loss function """


    precision = _symmetrize(precision)
    sign, logdet = jnp.linalg.slogdet(precision)
    #penalty = jnp.where(sign <= 0, 1e12, 0.0)
    #smooth = -logdet + jnp.trace(empirical_covariance @ precision) + penalty
    smooth = -logdet + jnp.trace(empirical_covariance @ precision)
    reg = _l1_offdiag(precision, lam, tau)

    return smooth + reg



####################################################################################################



def graphical_lasso_old(samples, lam=0.1, lr=1e-2,
                              max_steps=1000, tol=1e-3,
                              tau=1e-9, eps=1e-9,
                              init_from_naive=True,
                              use_tol = True):
    """ projected algorithm for graphical lasso method """


    n_samples, n_spins = samples.shape

    mean = jnp.mean(samples, axis=0)
    centered_samples = samples - mean
    empirical_covariance = (centered_samples.T @ centered_samples) / n_samples

    if init_from_naive:
        precision0 = jnp.linalg.inv(empirical_covariance + eps * jnp.eye(n_spins))
    else:
        precision0 = jnp.eye(n_spins)

    precision = precision0

    def loss_fn(precision):
        return _glasso_loss(precision, empirical_covariance, lam=lam, tau=tau)

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(precision)

    @jax.jit
    def step(precision, opt_state):
        loss, grads = jax.value_and_grad(loss_fn)(precision)
        updates, opt_state = optimizer.update(grads, opt_state, params=precision)
        precision_new = optax.apply_updates(precision, updates)
        precision_new = _project_to_pd(precision_new, eps=eps)
        upd_norm = jnp.linalg.norm(precision_new - precision)
        return precision_new, opt_state, loss, upd_norm

    history = [precision]
    for _ in range(max_steps):
        precision_new, opt_state, loss_value, upd = step(precision, opt_state)
        history.append(precision_new)
        if use_tol:
            if upd < tol:
                precision = precision_new
                break
        precision = precision_new

    precision_hat = _symmetrize(precision)

    return precision_hat, history



####################################################################################################



def graphical_lasso_sk(samples, lam=0.1, tol=1e-4, max_iter=100):
    """ graphical lasso via scikit-learn """


    samples = np.asarray(samples, dtype=np.float64)
    n_samples = len(samples)
    mean = np.mean(samples, axis = 0)
    centered_samples = samples - mean
    empirical_covariance = (centered_samples.T @ centered_samples) / n_samples
    _, precision = skl_graphical_lasso(empirical_covariance, alpha=lam, tol=tol, max_iter=max_iter)

    return precision



####################################################################################################



def solve_lasso_cd(W11, s12, lam,
                   max_iter = 100, tol = 1e-3):
    """ intermediate lasso solver """


    V = np.asarray(0.5 * (W11 + W11.T))
    u = np.asarray(s12).astype(V.dtype, copy=False)

    p = V.shape[0]
    beta = np.zeros(p, dtype=V.dtype)

    def soft(x, t):
        ax = np.abs(x)
        return np.sign(x) * np.maximum(ax - t, 0.0)

    Vjj = np.clip(np.diag(V), 1e-12, None)

    for _ in range(max_iter):
        beta_old = beta.copy()

        for j in range(p):
            zj = u[j] - V[j, :].dot(beta) + Vjj[j] * beta[j]
            beta[j] = soft(zj, lam) / Vjj[j]

        if np.max(np.abs(beta - beta_old)) < tol:
            break

    return jnp.asarray(beta)



####################################################################################################



def graphical_lasso(samples, lam = 0.1, lr = 1e-2,
                    max_steps = 100, tol = 1e-3,
                    tau = 1e-9, use_tol = True):
    """ graphical lasso implemented with linear algebra operations """


    n_samples, n_spins = samples.shape

    mean = jnp.mean(samples, axis=0)
    centered_samples = samples - mean
    empirical_covariance = (centered_samples.T @ centered_samples) / n_samples
    current_cov = empirical_covariance + lam * jnp.eye(n_spins)

    history = [current_cov]

    for t in range(max_steps):
        W_prev = current_cov

        for j in range(n_spins):
            idx = np.arange(n_spins)
            sub = idx[idx != j]

            W11 = current_cov[np.ix_(sub, sub)]
            s12 = empirical_covariance[sub, j]
            beta = solve_lasso_cd(W11, s12, lam)
            w12 = W11 @ beta
            current_cov = current_cov.at[sub, j].set(w12)
            current_cov = current_cov.at[j, sub].set(w12)

        history.append(current_cov)

        if use_tol:
            delta = jnp.max(jnp.abs(current_cov - W_prev))
            if float(delta) < tol:
                break

    prec = jnp.linalg.inv(current_cov)

    return prec



####################################################################################################
####################################################################################################
#                                                                                                  #
# graphical score matching lasso                                                                   #
#                                                                                                  #
####################################################################################################
####################################################################################################



def _gsm_loss(precision, empirical_covariance, lam=0.1, tau=1e-9):
    """ graphical score matching lasso loss function """


    precision = _symmetrize(precision)
    smooth = 0.5 * jnp.trace(empirical_covariance @ (precision @ precision)) - jnp.trace(precision)
    reg = _l1_offdiag(precision, lam, tau)

    return smooth + reg



####################################################################################################



def graphical_score_matching_old(samples, lam=0.1, lr=1e-2,
                             max_steps=1000, tol=1e-3,
                             tau=1e-9, init_from_naive=False, use_tol = True):
    """ graphical score matching lasso complete algorithm """


    n_samples, n_spins = samples.shape

    mean = jnp.mean(samples, axis=0)
    centered_samples = samples - mean
    empirical_covariance = (centered_samples.T @ centered_samples) / n_samples

    if init_from_naive:
        precision0 = jnp.linalg.inv(empirical_covariance + 1e-9 * jnp.eye(n_spins))
    else:
        precision0 = jnp.eye(n_spins)

    def loss_fn(precision):
        return _gsm_loss(precision, empirical_covariance, lam=lam, tau=tau)

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(precision0)

    @jax.jit
    def step(precision, opt_state):
        loss, grads = jax.value_and_grad(loss_fn)(precision)
        updates, opt_state = optimizer.update(grads, opt_state, params=precision)
        new_precision = _symmetrize(optax.apply_updates(precision, updates))
        return new_precision, opt_state, loss

    precision = precision0
    history = [precision]
    for _ in range(max_steps):
        precision_new, opt_state, loss_value = step(precision, opt_state)
        history.append(precision_new)
        upd_norm = jnp.linalg.norm(precision_new - precision)
        precision = precision_new
        if use_tol:
            if upd_norm < tol:
                print(_)
                break

    precision_hat = _symmetrize(precision)

    return precision_hat, history



####################################################################################################



def _soft(x, t):
    """ softing out """
    

    ax = np.abs(x)
    return np.sign(x) * np.maximum(ax - t, 0.0)



####################################################################################################



def _solve_sm_l1_row(empirical_covariance_np, i, lam,
                    beta0 = None, max_inner = 100, tol = 1e-3):
    """ intermediate optimization for score matching """


    empirical_covariance = _symmetrize(empirical_covariance_np)
    n_spins = empirical_covariance.shape[0]
    beta = np.zeros(n_spins, dtype=empirical_covariance.dtype) if beta0 is None else beta0.astype(empirical_covariance.dtype, copy=True)

    empirical_covariance_jj = np.clip(np.diag(empirical_covariance), 1e-12, None)

    for _ in range(max_inner):
        beta_old = beta.copy()
        for j in range(n_spins):
            rhs = (1.0 if j == i else 0.0) - (empirical_covariance[j, :].dot(beta) - empirical_covariance[j, j] * beta[j])
            if j == i:
                beta[j] = rhs / empirical_covariance_jj[j]
            else:
                beta[j] = _soft(rhs, lam) / empirical_covariance_jj[j]
        if np.max(np.abs(beta - beta_old)) < tol:
            break

    return beta



####################################################################################################




def graphical_score_matching(samples, lam = 0.1, max_sweeps = 100,
                             inner_max = 100, tol = 1e-3):
    """ graphical score matching lasso with linear algebra elementary optimization """


    samples = np.asarray(samples)
    n_samples, n_spins = samples.shape

    # empirical covariance
    mean = samples.mean(axis=0)
    centered_samples = samples - mean
    empirical_covariance_np = (centered_samples.T @ centered_samples) / n_samples

    Theta = np.zeros((n_spins, n_spins), dtype=empirical_covariance_np.dtype)

    for _ in range(max_sweeps):
        Theta_prev = Theta.copy()
        for i in range(n_spins):
            beta0 = Theta[i, :]  # warm start
            beta = _solve_sm_l1_row(empirical_covariance_np, i, lam, beta0=beta0,
                                    max_inner=inner_max, tol=tol)
            Theta[i, :] = beta
            Theta[:, i] = beta
        if np.max(np.abs(Theta - Theta_prev)) < tol:
            break

    Theta = 0.5 * (Theta + Theta.T)

    return jnp.asarray(Theta)



####################################################################################################
####################################################################################################
#                                                                                                  #
# clime                                                                                            #
#                                                                                                  #
####################################################################################################
####################################################################################################



def _clime_column(empirical_covariance, j, lam=0.1):
    """ solves:   min ||beta||_1  s.t.  || S beta - e_j ||_inf <= lam
        via LP with variables beta = u - v, u, v >= 0 """


    empirical_covariance = np.asarray(empirical_covariance, dtype=float)
    n_spins = empirical_covariance.shape[0]

    prob = pulp.LpProblem(f"clime_col_{j}", pulp.LpMinimize)

    u = [pulp.LpVariable(f"u_{k}", lowBound=0.0) for k in range(n_spins)]
    v = [pulp.LpVariable(f"v_{k}", lowBound=0.0) for k in range(n_spins)]

    prob += pulp.lpSum([u[k] + v[k] for k in range(n_spins)])

    for i in range(n_spins):
        expr = pulp.lpSum(empirical_covariance[i, k] * (u[k] - v[k]) for k in range(n_spins)) - (1.0 if i == j else 0.0)
        prob += (expr <= lam)
        prob += (-expr <= lam)

    _ = prob.solve(pulp.PULP_CBC_CMD(msg=False))

    beta = np.array([pulp.value(u[k]) - pulp.value(v[k]) for k in range(n_spins)], dtype=float)

    return beta



####################################################################################################



def _symmetrize_minabs(matrix):
    """ symmetrizing choosing, for each couple (i, j), the element with smallest abs """


    A = np.asarray(matrix, dtype=float)
    d = A.shape[0]
    B = A.copy()
    for i in range(d):
        for j in range(i + 1, d):
            a, b = A[i, j], A[j, i]
            pick = a if abs(a) <= abs(b) else b
            B[i, j] = pick
            B[j, i] = pick

    return _symmetrize(B)



####################################################################################################



def clime(samples, lam=0.1):
    """ clime algorithm for inference """


    n_samples, n_spins = samples.shape
    mean = jnp.mean(samples, axis=0)
    centered_samples = samples - mean
    empirical_covariance = (centered_samples.T @ centered_samples) / n_samples
    empirical_covariance = np.asarray(empirical_covariance, dtype=float)

    Omega_cols = []
    for j in range(n_spins):
        beta_j = _clime_column(empirical_covariance, j, lam=lam)
        Omega_cols.append(beta_j)

    Omega = np.column_stack(Omega_cols)

    precision_hat = _symmetrize_minabs(Omega)

    return jnp.array(precision_hat, dtype=jnp.float32)




####################################################################################################
####################################################################################################
#                                                                                                  #
# boltzmann machine                                                                                #
#                                                                                                  #
####################################################################################################
####################################################################################################



def _naive_boltzmann_loss(precision, empirical_covariance, lam=0.1, tau=1e-9):
    """ graphical lasso loss function """

    smooth = jnp.linalg.norm(jnp.linalg.inv(precision) - empirical_covariance)**2
    reg = _l1_offdiag(precision, lam, tau)

    return smooth + reg



####################################################################################################



def naive_boltzmann_machine(samples, lam=0.1, lr=1e-2,
                              max_steps=1000, tol=1e-3,
                              tau=1e-9, eps=1e-9,
                              init_from_naive=True,
                              use_tol = True):
    """ projected algorithm for graphical lasso method """


    n_samples, n_spins = samples.shape

    mean = jnp.mean(samples, axis=0)
    centered_samples = samples - mean
    empirical_covariance = (centered_samples.T @ centered_samples) / n_samples

    if init_from_naive:
        precision0 = jnp.linalg.inv(empirical_covariance + eps * jnp.eye(n_spins))
    else:
        precision0 = jnp.eye(n_spins)

    precision = precision0

    def loss_fn(precision):
        return _naive_boltzmann_loss(precision, empirical_covariance, lam=lam, tau=tau)

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(precision)

    @jax.jit
    def step(precision, opt_state):
        loss, grads = jax.value_and_grad(loss_fn)(precision)
        updates, opt_state = optimizer.update(grads, opt_state, params=precision)
        precision_new = optax.apply_updates(precision, updates)
        precision_new = _project_to_pd(precision_new, eps=eps)
        upd_norm = jnp.linalg.norm(precision_new - precision)
        return precision_new, opt_state, loss, upd_norm

    history = [precision]
    for _ in range(max_steps):
        precision_new, opt_state, loss_value, upd = step(precision, opt_state)
        history.append(precision_new)
        if use_tol:
            if upd < tol:
                precision = precision_new
                break
        precision = precision_new

    precision_hat = _symmetrize(precision)

    return precision_hat, history



####################################################################################################



def compute_energy_distance(samples, evolved_samples):
    """ computing energy distance between two empirical measures """


    n = samples.shape[0]
    m = evolved_samples.shape[0]


    def pairwise_l2_norm(a, b):
        a_norm = jnp.sum(a**2, axis=1).reshape(-1, 1)
        b_norm = jnp.sum(b**2, axis=1).reshape(1, -1)
        sq_dists = a_norm + b_norm - 2 * a @ b.T
        return jnp.sqrt(jnp.maximum(sq_dists, 1e-10))


    d_xy = pairwise_l2_norm(samples, evolved_samples)
    d_xx = pairwise_l2_norm(samples, samples)
    d_yy = pairwise_l2_norm(evolved_samples, evolved_samples)


    ed = (
        2 * jnp.sum(d_xy) / (n * m)
        - jnp.sum(d_xx) / (n * n)
        - jnp.sum(d_yy) / (m * m)
    )

    return ed



####################################################################################################



def _jax_boltzmann_loss(precision, samples, lam=0.1, tau=1e-9):
    """ jax boltzmann loss function """

    new_samples = GMM_sampler.precision_sampler(precision, len(samples))
    smooth = compute_energy_distance(samples, new_samples)

    reg = _l1_offdiag(precision, lam, tau)

    return smooth + reg



####################################################################################################



def jax_boltzmann_machine(samples, lam=0.1, lr=1e-2,
                              max_steps=1000, tol=1e-3,
                              tau=1e-9, eps=1e-9,
                              init_from_naive=True,
                              use_tol = True):
    """ projected algorithm for jax boltzmann machine method """


    n_samples, n_spins = samples.shape

    mean = jnp.mean(samples, axis=0)
    centered_samples = samples - mean
    empirical_covariance = (centered_samples.T @ centered_samples) / n_samples

    if init_from_naive:
        precision0 = jnp.linalg.inv(empirical_covariance + eps * jnp.eye(n_spins))
    else:
        precision0 = jnp.eye(n_spins)

    precision = precision0

    def loss_fn(precision):
        return _jax_boltzmann_loss(precision, samples, lam=lam, tau=tau)

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(precision)

    @jax.jit
    def step(precision, opt_state):
        loss, grads = jax.value_and_grad(loss_fn)(precision)
        updates, opt_state = optimizer.update(grads, opt_state, params=precision)
        precision_new = optax.apply_updates(precision, updates)
        precision_new = _project_to_pd(precision_new, eps=eps)
        upd_norm = jnp.linalg.norm(precision_new - precision)
        return precision_new, opt_state, loss, upd_norm

    history = [precision]
    for _ in range(max_steps):
        precision_new, opt_state, loss_value, upd = step(precision, opt_state)
        history.append(precision_new)
        if use_tol:
            if upd < tol:
                precision = precision_new
                break
        precision = precision_new

    precision_hat = _symmetrize(precision)

    return precision_hat, history



####################################################################################################



def _diag_boltzmann_loss(precision, samples, lam=0.1, tau=1e-9):
    """ diagonal boltzmann machine loss function """


    new_samples = GMM_sampler.diag_generation(precision, len(samples))
    smooth = compute_energy_distance(samples, new_samples)

    reg = _l1_offdiag(precision, lam, tau)

    return smooth + reg



####################################################################################################



def diag_boltzmann_machine(samples, lam=0.1, lr=1e-2,
                              max_steps=1000, tol=1e-3,
                              tau=1e-9, eps=1e-9,
                              init_from_naive=True,
                              use_tol = True):
    """ projected algorithm for diagonal boltzmann machine method """


    n_samples, n_spins = samples.shape

    mean = jnp.mean(samples, axis=0)
    centered_samples = samples - mean
    empirical_covariance = (centered_samples.T @ centered_samples) / n_samples

    if init_from_naive:
        precision0 = jnp.linalg.inv(empirical_covariance + eps * jnp.eye(n_spins))
    else:
        precision0 = jnp.eye(n_spins)

    precision = precision0

    def loss_fn(precision):
        return _diag_boltzmann_loss(precision, samples, lam=lam, tau=tau)

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(precision)

    @jax.jit
    def step(precision, opt_state):
        loss, grads = jax.value_and_grad(loss_fn)(precision)
        updates, opt_state = optimizer.update(grads, opt_state, params=precision)
        precision_new = optax.apply_updates(precision, updates)
        precision_new = _project_to_pd(precision_new, eps=eps)
        upd_norm = jnp.linalg.norm(precision_new - precision)
        return precision_new, opt_state, loss, upd_norm

    history = [precision]
    for _ in range(max_steps):
        precision_new, opt_state, loss_value, upd = step(precision, opt_state)
        history.append(precision_new)
        if use_tol:
            if upd < tol:
                precision = precision_new
                break
        precision = precision_new

    precision_hat = _symmetrize(precision)

    return precision_hat, history










####################################################################################################
####################################################################################################
#                                                                                                  #
# reconstruction                                                                                   #
#                                                                                                  #
####################################################################################################
####################################################################################################



def mle_covariance(samples, mask, seed = 0, eps = 1e-12, lam = 1e-1, lr = 1e-2, steps = 500):
    """ simple optimization to reconstruct the mle of covariance given a mask on the precision matrix """


    n_samples, n_spins = samples.shape
    mean = jnp.mean(samples, axis = 0)
    centered_samples = samples - mean
    empirical_covariance = (centered_samples.T @ centered_samples) / n_samples

    E = (mask.astype(jnp.float32) > 0).astype(jnp.float32)
    E = jnp.maximum(E, E.T)
    E = E.at[jnp.diag_indices(n_spins)].set(1.0)

    w, V = jnp.linalg.eigh(empirical_covariance + 1e-3 * jnp.eye(n_spins))
    w_clamped = jnp.clip(w, a_min=1e-6, a_max=None)
    L0 = V @ jnp.diag(jnp.sqrt(w_clamped))
    L = L0

    def loss_fn(L):
        Lp = _project_to_pd(L, eps)
        Sigma = Lp @ Lp.T
        logdet = 2.0 * jnp.sum(jnp.log(jnp.diag(Lp)))
        constr = E * (Sigma - empirical_covariance)
        pen = jnp.sum(constr * constr)
        return -logdet + lam * pen, (Sigma, logdet, pen)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    for _ in range(steps):
        (val, aux), g = grad_fn(L)
        L = L - lr * g

    Lp = _project_to_pd(L, eps)
    Sigma_hat = Lp @ Lp.T
    Sigma_hat = 0.5 * (Sigma_hat + Sigma_hat.T)

    return Sigma_hat