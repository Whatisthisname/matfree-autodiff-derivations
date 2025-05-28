def _bidiag_vjp_bwd(
    matvec: MatVec,
    cache_and_params: tuple[BidiagCache_matvec, tuple],
    nabla: BidiagOutput,
) -> BidiagInput:
    cache, matvec_params = cache_and_params
    _, vecmat_fun = jax.vjp(
        lambda v, p: matvec(v, *p), cache.start_vector, matvec_params
    )

    def vecmat(v):
        return vecmat_fun(v)[0]

    w0_like = jax.eval_shape(matvec, cache.start_vector, *matvec_params)
    (n,) = np.shape(w0_like)

    # Unpack primal variables from cache. These are 0-indexed.
    betas = cache.primal.betas
    alphas = cache.primal.alphas
    rs = cache.primal.rs
    ls = cache.primal.ls
    c = cache.primal.c

    k = num_matvecs

    CarryState = typing.NamedTuple(
        "CarryState",
        [
            ("lambda_n_plus_one", ArrayLike),
            ("rho_n", ArrayLike),
            ("param_incremental_grads", ArrayLike),
        ],
    )

    def rewritten_gradable_fn(params, lam, r, l, rho):
        return -lam @ matvec(r, *params) - l @ matvec(rho, *params)

    def body_fun(i_in: int, carry: CarryState):
    # ------------------------------------------------------------
    # reverse index: i_in = 0 … k-2  ->  n = k-1 … 1
    # ------------------------------------------------------------
    n = k - i_in - 1

    lambda_n_plus_one = carry.lambda_n_plus_one   # λ_{n+1}
    rho_n             = carry.rho_n              # ρ_n

    # ---- 1. raw residual parts ---------------------------------
    u_n = (nabla.ls[:, n]
           + betas[n] * lambda_n_plus_one
           - matvec(rho_n, *matvec_params))       # ∇l_n + b_n λ_{n+1} - A ρ_n

    # ---- 2. orthogonal-complement correction -------------------
    proj = ls @ (ls.T @ u_n)     # P u_n  (same dim as u_n)
    Q_u  = u_n - proj            # Q u_n
    lambda_perp = -Q_u / alphas[n]     # λ_n^⊥  =  - Q u_n / a_n

    # ---- 3. scalar coefficients inside span{L} -----------------
    # c_off =  - (l_m^T u_n) / a_n   for all m
    coeffs = - (ls.T @ u_n) / alphas[n]           # shape (k,)
    # overwrite the n-th entry with c_n = t
    t = - (nabla.alphas[n] + rs[:, n].T @ rho_n)  # c_n
    coeffs = coeffs.at[n].set(t)

    # ---- 4. assemble λ_n and σ_n --------------------------------
    lambda_n = ls @ coeffs + lambda_perp          # full λ_n column
    sigma_n  = - (ls[:, n].T @ u_n) - alphas[n] * t

    # ---- (optional) sanity check --------------------------------
    drift = (nabla.ls[:, n]
             + alphas[n] * lambda_n
             + betas[n] * lambda_n_plus_one
             - matvec(rho_n, *matvec_params)
             + ls[:, n] * sigma_n)
    jax.debug.print("‖dl_n residual‖₂ = {}", jnp.linalg.norm(drift))

    # ---- 5. remainder of your reverse recursion ----------------
    w     = -nabla.betas[n - 1] - ls[:, n - 1].T @ lambda_n
    omega = (- rs[:, n].T
             @ (nabla.rs[:, n] - vecmat(lambda_n) + alphas[n] * rho_n)
             - betas[n - 1] * w)

    undivided_rho_prev = (-nabla.rs[:, n]
                          + vecmat(lambda_n)
                          - alphas[n] * rho_n
                          - omega * rs[:, n])

    rho_n_minus_one = jax.lax.cond(
        pred=jnp.allclose(betas[n - 1], 0.0),
        true_fun=lambda: jnp.zeros_like(undivided_rho_prev),
        false_fun=lambda: undivided_rho_prev / betas[n - 1],
    )

    new_param_grad_incr = jax.grad(rewritten_gradable_fn, argnums=0)(
        matvec_params,
        lambda_n,
        rs[:, n],
        ls[:, n],
        rho_n,
    )

    return CarryState(
        lambda_n_plus_one=lambda_n,
        rho_n=rho_n_minus_one,
        param_incremental_grads=jax.tree_util.tree_map(
            lambda running_sum, grad_component: running_sum + grad_component,
            carry.param_incremental_grads,
            new_param_grad_incr,
        ),
    )

# initialize param_grads to 0
init_param_grads = jax.tree.map(lambda x: x * 0, matvec_params)
if num_matvecs > 1:
    output: CarryState = jax.lax.fori_loop(
        lower=0,
        upper=k - 2 + 1,  # (+ 1 to include in iteration)
        body_fun=body_fun,
        init_val=CarryState(
            lambda_n_plus_one=jnp.zeros(n),
            rho_n=-nabla.res,
            param_incremental_grads=init_param_grads,
        ),
    )
else:
    output: CarryState = CarryState(
        lambda_n_plus_one=jnp.zeros(n),
        rho_n=-nabla.res,
        param_incremental_grads=init_param_grads,
    )

# last iteration steps:

if num_matvecs > 1:  # beta is defined from num_matvecs >= 2
    beta_times_next_lam = betas[0] * output.lambda_n_plus_one
else:
    beta_times_next_lam = 0 * output.lambda_n_plus_one

t = -nabla.alphas[0] - rs[:, 0].T @ output.rho_n

matvec_ = matvec(output.rho_n, *matvec_params)
u_n = nabla.ls[:, 0] + beta_times_next_lam - matvec_
sigma = -ls[:, 0].T @ u_n - alphas[0] * t

if reorthogonalize:
    lambda_1 = ls[:, 0] * t - (1 / alphas[0]) * (np.eye(k) - ls @ ls.T) @ u_n
else:
    lambda_1 = (-u_n - sigma * ls[:, 0]) / alphas[0]  # error?

jax.debug.print("lambda inner prod with ls[:, n]: {}", ls.T @ lambda_1)

v = nabla.rs[:, 0] - vecmat(lambda_1) + alphas[0] * output.rho_n

omega = -rs[:, 0].T @ v - c * nabla.c

kappa = -v - omega * rs[:, 0]

# use kappa and rhos and lambdas to compute grads.A and grads.start_vector
param_grads_out = jax.tree_util.tree_map(
    lambda running_sum, grad_component: running_sum + grad_component,
    output.param_incremental_grads,
    jax.grad(rewritten_gradable_fn, argnums=0)(
        matvec_params, lambda_1, rs[:, 0], ls[:, 0], output.rho_n
    ),
)

return (
    -c * kappa,
    *param_grads_out,
)
