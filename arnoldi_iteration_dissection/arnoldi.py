from typing import Callable, Union, NamedTuple
import jax.flatten_util
import jax.numpy as jnp
from jax import Array
import jax

jax.config.update("jax_enable_x64", True)


def arnoldi(matvecs_num: int, custom_vjp: bool, reorthogonalize: bool):
    def estimate(matvec, v0, *params):
        n = v0.shape[0]
        Q = jnp.zeros(shape=(n, matvecs_num))
        H = jnp.zeros(shape=(matvecs_num, matvecs_num))
        c = 1 / jnp.linalg.norm(v0)
        Q = Q.at[:, 0].set(v0 * c)

        def body_fun(i, stuff):
            Q, H, _ = stuff
            # jax.debug.print("Iteration {}: \nQ = {}", i, Q)

            _Aqi = matvec(Q[:, i], *params)
            # jax.debug.print("Aq_{} = {}", i, _Aqi)

            hs_i = Q.T @ _Aqi
            # jax.debug.print("hs_{} = {}", i, hs_i)

            t = _Aqi - Q @ hs_i
            hs_next_i = jnp.linalg.norm(t)

            # jax.debug.print("hs_next_{} = {}", i, hs_next_i)
            if reorthogonalize:
                q = t / hs_next_i
                Q = Q.at[:, i + 1].set(q - Q @ Q.T @ q)
            else:
                Q = Q.at[:, i + 1].set(t / hs_next_i)

            # jax.debug.print("i={}, {}", i, t)

            hs = hs_i.at[i + 1].set(hs_next_i)
            # jax.debug.print("hs_{} = {}", i, hs)
            H = H.at[:, i].set(hs)

            return (Q, H, t)

        (Q, H, t) = jax.lax.fori_loop(
            lower=0, upper=matvecs_num, body_fun=body_fun, init_val=(Q, H, v0)
        )

        return Q, H, t, c

    def _bidiag_vjp_fwd(matvec, v0, *matvec_params) -> tuple:
        matvec_convert, aux_args = jax.closure_convert(
            lambda u, *v: matvec(u, *v), v0, *matvec_params
        )

        (Q, H) = estimate(matvec_convert, v0, *matvec_params, *aux_args)
        cache = (Q, H, v0)

        return (Q, H), (cache, matvec_params)

    def _bidiag_vjp_bwd(
        matvec,
        cache,
        gradients,
    ) -> tuple:
        (Q, H, v0), matvec_params = cache
        nablaQ, nablaH = gradients

        _, vecmat_fun = jax.vjp(lambda v, p: matvec(v, *p), v0, matvec_params)

        def vecmat(v):
            return vecmat_fun(v)[0]

        w0_like = jax.eval_shape(matvec, v0, *matvec_params)
        (n,) = jnp.shape(w0_like)

        return v0, matvec_params

    if custom_vjp:
        arnoldi_func = jax.custom_vjp(
            estimate,
            nondiff_argnums=(0,),
        )
        arnoldi_func.defvjp(
            _bidiag_vjp_fwd,
            _bidiag_vjp_bwd,
        )
    else:
        arnoldi_func = estimate

    return jax.jit(arnoldi_func, static_argnums=(0))


class _DecompResult(NamedTuple):
    # If an algorithm returns a single Q, place it here.
    # If it returns multiple Qs, stack them
    # into a tuple and place them here.
    Q_tall: Union[Array, tuple[Array, ...]]

    # If an algorithm returns a materialized matrix,
    # place it here. If it returns a sparse representation
    # (e.g. two vectors representing diagonals), place it here
    J_small: Union[Array, tuple[Array, ...]]

    residual: Array
    init_length_inv: Array


def hessenberg(
    num_matvecs, /, *, reortho: str, custom_vjp: bool = True, reortho_vjp: str = "match"
):
    r"""Construct a **Hessenberg-factorisation** via the Arnoldi iteration.

    Uses pre-allocation, and full reorthogonalisation if `reortho` is set to `"full"`.
    It tends to be a good idea to use full reorthogonalisation.

    This algorithm works for **arbitrary matrices**.

    Setting `custom_vjp` to `True` implies using efficient, numerically stable
    gradients of the Arnoldi iteration according to what has been proposed by
    Krämer et al. (2024).
    These gradients are exact, so there is little reason not to use them.
    If you use this configuration,
    please consider citing Krämer et al. (2024; bibtex below).

    ??? note "BibTex for Krämer et al. (2024)"
        ```bibtex
        @article{kraemer2024gradients,
            title={Gradients of functions of large matrices},
            author={Kr\"amer, Nicholas and Moreno-Mu\~noz, Pablo and
            Roy, Hrittik and Hauberg, S{\o}ren},
            journal={arXiv preprint arXiv:2405.17277},
            year={2024}
        }
        ```
    """
    reortho_expected = ["none", "full"]
    if reortho not in reortho_expected:
        msg = f"Unexpected input for {reortho}: either of {reortho_expected} expected."
        raise TypeError(msg)

    def estimate(matvec, real_size, v, *params):
        matvec_convert, aux_args = jax.closure_convert(matvec, v, *params)
        return _estimate(matvec_convert, real_size, v, *params, *aux_args)

    def _estimate(matvec_convert: Callable, real_size, v, *params):
        return _hessenberg_forward(
            matvec_convert, real_size, num_matvecs, v, *params, reortho=reortho_vjp
        )

    def estimate_fwd(matvec_convert: Callable, real_size, v, *params):
        outputs = _estimate(matvec_convert, real_size, v, *params)
        return outputs, (outputs, params, real_size)

    def estimate_bwd(matvec_convert: Callable, cache, vjp_incoming):
        (Q, H, r, c), params, real_size = cache
        dQ, dH, dr, dc = vjp_incoming

        return _hessenberg_adjoint(
            matvec_convert,
            real_size,
            *params,
            Q=Q,
            H=H,
            r=r,
            c=c,
            dQ=dQ,
            dH=dH,
            dres=dr,
            dc=dc,
            reortho=reortho,
        )

    if custom_vjp:
        _estimate = jax.custom_vjp(_estimate, nondiff_argnums=(0,))
        _estimate.defvjp(estimate_fwd, estimate_bwd)  # type: ignore
    return estimate


def _hessenberg_forward(matvec, real_size, num_matvecs, v, *params, reortho: str):
    if num_matvecs < 0 or num_matvecs > len(v):
        msg = "fuck"  # error_num_matvecs(num_matvecs, maxval=len(v), minval=0)
        raise ValueError(msg)

    # Initialise the variables
    (n,), k = jnp.shape(v), num_matvecs
    Q = jnp.zeros((n, k), dtype=v.dtype)
    H = jnp.zeros((k, k), dtype=v.dtype)
    initlength = jnp.sqrt(v @ v)
    init = (Q, H, v, initlength)

    if num_matvecs == 0:
        return _DecompResult(
            Q_tall=Q, J_small=H, residual=v, init_length_inv=1 / initlength
        )

    # Fix the step function
    def forward_step(i, val):
        return _hessenberg_forward_step(*val, matvec, *params, idx=i, reortho=reortho)

    # Loop and return
    Q, H, v, _length = jax.lax.fori_loop(0, k, forward_step, init)
    return _DecompResult(
        Q_tall=Q, J_small=H, residual=v, init_length_inv=1 / initlength
    )


def _hessenberg_forward_step(Q, H, v, length, matvec, *params, idx, reortho: str):
    # Save
    v /= length
    Q = Q.at[:, idx].set(v)

    # Evaluate
    v = matvec(v, *params)

    # Orthonormalise
    h = Q.T @ v
    v = v - Q @ h

    # Re-orthonormalise
    if reortho != "none":
        v = v - Q @ (Q.T @ v)

    # Read the length
    length = jnp.sqrt(v @ v)

    # Save
    h = h.at[idx + 1].set(length)
    H = H.at[:, idx].set(h)

    return Q, H, v, length


def _hessenberg_adjoint(
    matvec, real_size, *params, Q, H, r, c, dQ, dH, dres, dc, reortho: str
):
    # Extract the matrix shapes from Q
    _, num_matvecs = jnp.shape(Q)
    n, m = real_size

    # Prepare a bunch of auxiliary matrices

    def lower(m):
        m_tril = jnp.tril(m)
        return m_tril - 0.5 * jnp.diag(jnp.diag(m_tril))

    e_1, e_K = jnp.eye(num_matvecs)[[0, -1], :]
    lower_mask = lower(jnp.ones((num_matvecs, num_matvecs)))

    # Initialise
    eta = dH @ e_K - Q.T @ dres
    lambda_k = dres + Q @ eta
    Lambda = jnp.zeros_like(Q)
    Gamma = jnp.zeros_like(dQ.T @ Q)
    dp = jax.tree.map(jnp.zeros_like, params)
    # jax.debug.print("last H \n{}", dH @ e_K)
    # jax.debug.print("Q @ last H \n{}", Q @ dH @ e_K)
    # jax.debug.print("lamk \n{}", Q @ Q.T @ dres)
    # jax.debug.print("lamk \n{}", lambda_k)

    # Prepare more  auxiliary matrices
    Pi_xi = dQ.T + jnp.linalg.outer(eta, r)
    Pi_gamma = -dc * c * jnp.linalg.outer(e_1, e_1) + H @ dH.T - (dQ.T @ Q)

    # Prepare reorthogonalisation:
    reortho_mask = jnp.tril(jnp.ones((num_matvecs, num_matvecs)), 1)

    # Loop over those values
    indices = jnp.arange(0, len(H), step=1)
    beta_minuses = jnp.concatenate([jnp.ones((1,)), jnp.diag(H, -1)])
    alphas = jnp.diag(H)
    beta_pluses = H - jnp.diag(jnp.diag(H)) - jnp.diag(jnp.diag(H, -1), -1)
    scan_over = {
        "beta_minus": beta_minuses,
        "alpha": alphas,
        "beta_plus": beta_pluses,
        "idx": indices,
        "lower_mask": lower_mask,
        "Pi_gamma": Pi_gamma,
        "Pi_xi": Pi_xi,
        "dH_k": dH.T,
        "reortho_mask_k": reortho_mask,
        "q": Q.T,
    }

    # Fix the step function
    def adjoint_step(x, y):
        output = _hessenberg_adjoint_step(
            *x, **y, matvec=matvec, params=params, Q=Q, reortho=reortho
        )
        return output, ()

    # Scan
    init = (lambda_k, Lambda, Gamma, dp)
    result, _ = jax.lax.scan(adjoint_step, init, xs=scan_over, reverse=True)
    (lambda_k, Lambda, Gamma, dp) = result

    jax.debug.print("Q.T:{}", Q.T)
    jax.debug.print("Lambda\n{}", Lambda)

    r1 = Q[4:, 0]
    l1 = Q[:4, 1]
    up_1 = Lambda[:4, 0]
    down_1 = Lambda[4:, 1]
    jax.debug.print("final_grad\n{}", jnp.outer(up_1, r1) + jnp.outer(l1, down_1))
    jax.debug.print("l_1\n{}", l1)

    jax.debug.print("final A augmented Grad:\n{}", Lambda @ Q.T)
    jax.debug.print("final A Grad:\n{}", Lambda @ Q.T + Q @ Lambda.T)
    # jax.debug.print("dQ:\n{}", dQ)

    # Solve for the input gradient
    dv = lambda_k * c

    return (None, None), dv, *dp


def _hessenberg_adjoint_step(
    # Running variables
    lambda_k,
    Lambda,
    Gamma,
    dp,
    *,
    # Matrix-vector product
    matvec,
    params,
    # Loop over: index
    idx,
    # Loop over: submatrices of H
    beta_minus,
    alpha,
    beta_plus,
    # Loop over: auxiliary variables for Gamma
    lower_mask,
    Pi_gamma,
    Pi_xi,
    q,
    # Loop over: reorthogonalisation
    dH_k,
    reortho_mask_k,
    # Other parameters
    Q,
    reortho: str,
):
    # Reorthogonalise
    if reortho == "full":
        # Get rid of the (I_ll o Sigma) term by multiplying with a mask
        Q_masked = reortho_mask_k[None, :] * Q
        rhs_masked = reortho_mask_k * dH_k
        # jax.debug.print("rhs, masked\n{}", rhs_masked)

        # Project x to Q^T x = y via
        # x = x - Q Q^\top x + Q Q^\top x = x - Q Q^\top x + Q y
        # (here, x = lambda_k and y = dH_k)
        lambda_k = lambda_k - Q_masked @ (Q_masked.T @ lambda_k) + Q_masked @ rhs_masked
        # jax.debug.print(" Q_masked @ rhs_masked: \n{}", Q_masked @ rhs_masked)

    # Transposed matvec and parameter-gradient in a single matvec
    _, vjp = jax.vjp(lambda u, v: matvec(u, *v), q, params)
    vecmat_lambda, dp_increment = vjp(lambda_k)

    # jax.debug.print("idx: {}", idx, ordered=True)

    dp = jax.tree.map(lambda g, h: g + h, dp, dp_increment)

    # Solve for (Gamma + Gamma.T) e_K
    tmp = lower_mask * (Pi_gamma - vecmat_lambda @ Q)
    Gamma = Gamma.at[idx, :].set(tmp)

    def test(i):
        pass
        # jax.debug.print("!!!Vecmat_Lambda: \n{}", vecmat_lambda)
        # jax.debug.print("Gamma:{}", Gamma)
        # jax.debug.print("Q:\n{}", Q)
        # jax.debug.print("Q @ Q.T:\n{}", Q @ Q.T)
        # ok = Q.T @ d_Q

    jax.lax.cond(
        idx == 0,
        true_fun=test,
        false_fun=lambda *p: None,
        operand=idx,
    )

    # Solve for the next lambda (backward substitution step)
    Lambda = Lambda.at[:, idx].set(lambda_k)
    xi = Pi_xi + (Gamma + Gamma.T)[idx, :] @ Q.T
    lambda_k = xi - (alpha * lambda_k - vecmat_lambda) - beta_plus @ Lambda.T
    lambda_k /= beta_minus
    return lambda_k, Lambda, Gamma, dp
