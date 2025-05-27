#!/usr/bin/env python
from functools import partial
import typing
import numpy as np
import dataclasses
import pytest
import jax  # type: ignore[import-not-found]
from jax.typing import ArrayLike  # type: ignore[import-not-found]
import jax.numpy as jnp  # type: ignore[import-not-found]

jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_debug_nans", True)


MatVec = typing.Callable[[ArrayLike], ArrayLike]


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass
class BidiagInput:
    A: ArrayLike
    """(n, m) matrix"""
    start_vector: ArrayLike
    r"""(m,) vector, a.k.a. $\tilde r$"""

    def tree_flatten(self):
        children = (self.A, self.start_vector)
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        A, start_vector = children
        return cls(A=A, start_vector=start_vector)


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass
class BidiagOutput:
    rs: ArrayLike
    """(m,k) float array"""
    ls: ArrayLike
    """(n,k) float array"""
    alphas: ArrayLike
    """(k,) float array"""
    betas: ArrayLike
    """(k-1,) float array"""
    c: float
    """(k-1,) float array"""
    res: ArrayLike
    """(m,) vector, beta_k * r_{k+1}"""

    def tree_flatten(self):
        children = (
            self.rs,
            self.ls,
            self.alphas,
            self.betas,
            self.c,
            self.res,
        )
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    @property
    def L(self) -> ArrayLike:
        """(n, k) float array"""
        return self.ls

    @property
    def B(self) -> ArrayLike:
        """(k, k) float array"""
        as_diag = jnp.diag(self.alphas)
        for i in range(len(self.betas)):
            as_diag = as_diag.at[i, i + 1].set(self.betas[i])
        return as_diag

    @property
    def R(self) -> ArrayLike:
        """(m, k) float array"""
        return self.rs

    @property
    def iterations_finished(self) -> int:
        """Accurate only up to floating point precision..."""

        # returns index of first zero element or the last element if no zero element is found
        def first_zero_or_len(arr):
            zeros = jnp.isclose(arr, 0.0, atol=1e-6)
            return jnp.where(jnp.any(zeros), jnp.argmax(zeros), len(arr) - 1)

        return first_zero_or_len(self.alphas)


@partial(jax.jit, static_argnames=("num_matvecs",))
def bidiagonalize(
    primals: tuple[ArrayLike, ArrayLike],
    num_matvecs: int,
) -> BidiagOutput:
    A, start_vector = primals

    c = 1 / jnp.linalg.norm(start_vector)

    size = num_matvecs + 1

    as_ = jnp.zeros((size))
    bs = jnp.zeros((size))
    rs = jnp.zeros((A.shape[1], size + 1))
    rs = rs.at[:, 1].set(start_vector * c)
    ls = jnp.zeros((A.shape[0], size))

    CarryState = typing.NamedTuple(
        "CarryState",
        [
            ("rs", ArrayLike),
            ("ls", ArrayLike),
            ("as_", ArrayLike),
            ("bs", ArrayLike),
        ],
    )

    def body_fun(i, carry: CarryState) -> CarryState:
        n = i + 1

        # Forward pass step
        if True:
            t = A @ carry.rs[:, n] - carry.bs[n - 1] * carry.ls[:, n - 1]

            new_alpha, new_l = jax.lax.cond(
                pred=jnp.allclose(t.T @ t, 0, atol=1e-6),  # | jnp.isnan(alpha_k),
                true_fun=lambda: (0.0, jnp.zeros_like(t)),
                false_fun=lambda: (jnp.linalg.norm(t), t / jnp.linalg.norm(t)),
            )

            as_ = carry.as_.at[n].set(new_alpha)
            ls = carry.ls.at[:, n].set(new_l)

            w = A.T @ ls[:, n] - as_[n] * carry.rs[:, n]
            # beta_k = jnp.linalg.norm(w)

            new_beta, new_r = jax.lax.cond(
                pred=jnp.allclose(w.T @ w, 0, atol=1e-6),  # | jnp.isnan(beta_k),
                true_fun=lambda: (0.0, jnp.zeros_like(w)),
                false_fun=lambda: (jnp.linalg.norm(w), w / jnp.linalg.norm(w)),
            )

            bs = carry.bs.at[n].set(new_beta)
            rs = carry.rs.at[:, n + 1].set(new_r)

        return CarryState(
            rs=rs,
            ls=ls,
            as_=as_,
            bs=bs,
        )

    # loop_out = body_fun(
    #     0,
    #     CarryState(
    #         rs=rs,
    #         ls=ls,
    #         as_=as_,
    #         bs=bs,
    #     ),
    # )

    # Run the loop
    loop_out = jax.lax.fori_loop(
        lower=0,
        upper=num_matvecs,
        body_fun=body_fun,
        init_val=CarryState(
            rs=rs,
            ls=ls,
            as_=as_,
            bs=bs,
        ),
    )

    k = num_matvecs

    # Create primal output
    primal_output = BidiagOutput(
        c=c,
        res=loop_out.bs[k] * loop_out.rs[:, k + 1],
        rs=loop_out.rs[:, 1:-1],
        ls=loop_out.ls[:, 1:],
        alphas=loop_out.as_[1:],
        betas=loop_out.bs[1:-1],
    )
    return primal_output


def bidiagonalize_primal(num_matvecs: int, reorthogonalize: bool = False):
    def bidiagonalize_matvec(
        matvec: MatVec,
        r1_tilde: ArrayLike,
        *matvec_params,
    ) -> BidiagOutput:
        _, vecmat_fun = jax.vjp(matvec, r1_tilde, *matvec_params)

        def vecmat(v):
            return vecmat_fun(v)[0]

        (ncols,) = np.shape(r1_tilde)
        w0_like = jax.eval_shape(matvec, r1_tilde, *matvec_params)
        (nrows,) = np.shape(w0_like)

        c = 1 / jnp.linalg.norm(r1_tilde)

        size = num_matvecs + 1

        as_ = jnp.zeros((size))
        bs = jnp.zeros((size))
        rs = jnp.zeros((ncols, size + 1))
        rs = rs.at[:, 1].set(r1_tilde * c)
        ls = jnp.zeros((nrows, size))

        CarryState = typing.NamedTuple(
            "CarryState",
            [
                ("rs", ArrayLike),
                ("ls", ArrayLike),
                ("as_", ArrayLike),
                ("bs", ArrayLike),
            ],
        )

        def body_fun(i, carry: CarryState) -> CarryState:
            n = i + 1

            # Forward pass step
            if True:
                t = (
                    matvec(carry.rs[:, n], *matvec_params)
                    - carry.bs[n - 1] * carry.ls[:, n - 1]
                )

                new_alpha, new_l = jax.lax.cond(
                    pred=jnp.allclose(t, 0, atol=1e-6),  # | jnp.isnan(alpha_k),
                    true_fun=lambda: (0.0, jnp.zeros_like(t)),
                    false_fun=lambda: (jnp.linalg.norm(t), t / jnp.linalg.norm(t)),
                )

                as_ = carry.as_.at[n].set(new_alpha)
                # ls = carry.ls.at[:, n].set(new_l)

                if reorthogonalize:
                    # _L = jax.lax.stop_gradient(carry.ls)
                    _L = carry.ls
                    ls = carry.ls.at[:, n].set(new_l - _L @ _L.T @ new_l)
                else:
                    ls = carry.ls.at[:, n].set(new_l)

                w = vecmat(ls[:, n]) - as_[n] * carry.rs[:, n]
                # beta_k = jnp.linalg.norm(w)

                new_beta, new_r = jax.lax.cond(
                    pred=jnp.allclose(w, 0, atol=1e-6),  # | jnp.isnan(beta_k),
                    true_fun=lambda: (0.0, jnp.zeros_like(w)),
                    false_fun=lambda: (jnp.linalg.norm(w), w / jnp.linalg.norm(w)),
                )

                bs = carry.bs.at[n].set(new_beta)
                rs = carry.rs.at[:, n + 1].set(new_r)

            return CarryState(
                rs=rs,
                ls=ls,
                as_=as_,
                bs=bs,
            )

        # Run the loop
        loop_out = jax.lax.fori_loop(
            lower=0,
            upper=num_matvecs,
            body_fun=body_fun,
            init_val=CarryState(
                rs=rs,
                ls=ls,
                as_=as_,
                bs=bs,
            ),
        )

        k = num_matvecs

        # Create primal output
        primal_output = BidiagOutput(
            c=c,
            res=loop_out.bs[k] * loop_out.rs[:, k + 1],
            rs=loop_out.rs[:, 1:-1],
            ls=loop_out.ls[:, 1:],
            alphas=loop_out.as_[1:],
            betas=loop_out.bs[1:-1],
        )
        return primal_output

    return bidiagonalize_matvec


@partial(jax.jit, static_argnames=("num_matvecs",))
def bidiagonalize_jvp(
    primals: tuple[ArrayLike, ArrayLike],
    tangents: tuple[ArrayLike, ArrayLike],
    num_matvecs: int,
) -> tuple[BidiagOutput, BidiagOutput]:
    A, start_vector = primals
    dA, d_start_vector = tangents

    c = 1 / jnp.linalg.norm(start_vector)

    size = num_matvecs + 1

    as_ = jnp.zeros((size))
    bs = jnp.zeros((size))
    rs = jnp.zeros((A.shape[1], size + 1))
    rs = rs.at[:, 1].set(start_vector * c)
    ls = jnp.zeros((A.shape[0], size))

    # Initialize tangent variables
    d_as = jnp.zeros((size))
    d_bs = jnp.zeros((size))
    d_rs = jnp.zeros((A.shape[1], size + 1))
    d_rs = d_rs.at[:, 1].set(
        (
            d_start_vector
            - start_vector
            * (start_vector.T @ d_start_vector)
            / (start_vector @ start_vector)
        )
        / jnp.linalg.norm(start_vector)
    )
    d_ls = jnp.zeros((A.shape[0], size))
    d_res = jnp.zeros((A.shape[0]))

    CarryState = typing.NamedTuple(
        "CarryState",
        [
            ("rs", ArrayLike),
            ("d_rs", ArrayLike),
            ("ls", ArrayLike),
            ("d_ls", ArrayLike),
            ("as_", ArrayLike),
            ("d_as", ArrayLike),
            ("bs", ArrayLike),
            ("d_bs", ArrayLike),
        ],
    )

    def body_fun(i, carry: CarryState) -> CarryState:
        n = i + 1

        # Forward pass step
        if True:
            t = A @ carry.rs[:, n] - carry.bs[n - 1] * carry.ls[:, n - 1]

            new_alpha, new_l = jax.lax.cond(
                pred=jnp.allclose(t.T @ t, 0, atol=1e-6),  # | jnp.isnan(alpha_k),
                true_fun=lambda: (0.0, jnp.zeros_like(t)),
                false_fun=lambda: (jnp.linalg.norm(t), t / jnp.linalg.norm(t)),
            )

            as_ = carry.as_.at[n].set(new_alpha)
            ls = carry.ls.at[:, n].set(new_l)

            w = A.T @ ls[:, n] - as_[n] * carry.rs[:, n]

            new_beta, new_r = jax.lax.cond(
                pred=jnp.allclose(w.T @ w, 0, atol=1e-6),  # | jnp.isnan(beta_k),
                true_fun=lambda: (0.0, jnp.zeros_like(w)),
                false_fun=lambda: (jnp.linalg.norm(w), w / jnp.linalg.norm(w)),
            )

            bs = carry.bs.at[n].set(new_beta)
            rs = carry.rs.at[:, n + 1].set(new_r)

        # Tangent map stuff
        if True:
            d_t = (
                dA @ rs[:, n]
                + A @ carry.d_rs[:, n]
                - carry.d_bs[n - 1] * ls[:, n - 1]
                - bs[n - 1] * carry.d_ls[:, n - 1]
            )
            d_alpha_n = ls[:, n].T @ d_t
            d_as = carry.d_as.at[n].set(d_alpha_n)

            d_l_n = jax.lax.cond(
                pred=jnp.allclose(new_alpha, 0, atol=1e-6) | jnp.isnan(new_alpha),
                true_fun=lambda: jnp.zeros_like(ls[:, 0]),
                false_fun=lambda: (d_t - d_alpha_n * ls[:, n]) / new_alpha,
            )
            d_ls = carry.d_ls.at[:, n].set(d_l_n)

            d_w = (
                dA.T @ ls[:, n]
                + A.T @ d_ls[:, n]
                - d_as[n] * rs[:, n]
                - as_[n] * carry.d_rs[:, n]
            )
            d_beta_n = rs[:, n + 1].T @ d_w
            d_bs = carry.d_bs.at[n].set(d_beta_n)

            d_r_np1 = jax.lax.cond(
                pred=jnp.allclose(new_beta, 0, atol=1e-6) | jnp.isnan(new_beta),
                true_fun=lambda: jnp.zeros_like(rs[:, 0]),
                false_fun=lambda: (d_w - d_beta_n * rs[:, n + 1]) / new_beta,
            )
            d_rs = carry.d_rs.at[:, n + 1].set(d_r_np1)

        return CarryState(
            rs=rs,
            d_rs=d_rs,
            ls=ls,
            d_ls=d_ls,
            as_=as_,
            d_as=d_as,
            bs=bs,
            d_bs=d_bs,
        )

    # Run the loop
    loop_out = jax.lax.fori_loop(
        lower=0,
        upper=num_matvecs,
        body_fun=body_fun,
        init_val=CarryState(
            rs=rs,
            d_rs=d_rs,
            ls=ls,
            d_ls=d_ls,
            as_=as_,
            d_as=d_as,
            bs=bs,
            d_bs=d_bs,
        ),
    )

    # Compute d_c
    d_c = -(start_vector @ d_start_vector) / (
        start_vector @ start_vector * jnp.linalg.norm(start_vector)
    )

    k = num_matvecs

    d_res = (
        A.T @ loop_out.d_ls[:, k]
        + dA.T @ loop_out.ls[:, k]
        - loop_out.as_[k] * loop_out.d_rs[:, k]
        - loop_out.d_as[k] * loop_out.rs[:, k]
    )

    # Create primal output
    primal_output = BidiagOutput(
        c=c,
        res=loop_out.bs[k] * loop_out.rs[:, k + 1],
        rs=loop_out.rs[:, 1:-1],
        ls=loop_out.ls[:, 1:],
        alphas=loop_out.as_[1:],
        betas=loop_out.bs[1:-1],
    )

    # Create tangent output
    tangent_output = BidiagOutput(
        c=d_c,
        res=d_res,
        rs=loop_out.d_rs[:, 1:-1],
        ls=loop_out.d_ls[:, 1:],
        alphas=loop_out.d_as[1:],
        betas=loop_out.d_bs[1:-1],
    )

    return primal_output, tangent_output


BidiagCache = typing.NamedTuple(
    "BidiagCache",
    [
        ("primal", BidiagOutput),
        ("A", ArrayLike),
        ("start_vector", ArrayLike),
    ],
)

BidiagCache_matvec = typing.NamedTuple(
    "BidiagCache_matvec",
    [
        ("primal", BidiagOutput),
        ("start_vector", ArrayLike),
    ],
)


def bidiagonalize_vjpable_matvec(
    num_matvecs: int,
    custom_vjp: bool = True,
    reorthogonalize: bool = False,
):
    primal_map = bidiagonalize_primal(num_matvecs, reorthogonalize)

    # @partial(jax.jit, static_argnums=(0,))
    def _bidiag_vjp_fwd(
        matvec: MatVec, v0: ArrayLike, *matvec_params
    ) -> tuple[BidiagOutput, tuple[BidiagCache_matvec, tuple]]:
        matvec_convert, aux_args = jax.closure_convert(
            lambda u, *v: matvec(u, *v), v0, *matvec_params
        )

        primal = primal_map(matvec_convert, v0, *matvec_params, *aux_args)
        cache = BidiagCache_matvec(
            primal=primal,
            start_vector=v0,
        )
        return primal, (cache, matvec_params)

    # @partial(jax.jit, static_argnums=(0,))
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
            # 'i_in' will go from 0 to k-2 (inclusive)
            n = k - i_in - 1
            # so 'n' will go from k-1 to 1 (inclusive)

            lambda_n_plus_one = carry.lambda_n_plus_one

            rho_n = carry.rho_n

            t = -nabla.alphas[n] - rs[:, n].T @ rho_n
            u_n = (
                nabla.ls[:, n]
                + betas[n] * lambda_n_plus_one
                - matvec(rho_n, *matvec_params)
            )
            sigma = -ls[:, n].T @ u_n - alphas[n] * t

            if reorthogonalize:
                lambda_n = (
                    ls[:, n] * t - (1 / alphas[n]) * (np.eye(k) - ls @ ls.T) @ u_n
                )
                jax.debug.print("lambda inner prod with ls[:, n]: {}", ls.T @ lambda_n)

            else:
                lams_undivided = -u_n - sigma * ls[:, n]

                lambda_n = jax.lax.cond(
                    pred=jnp.allclose(alphas[n], 0.0),
                    true_fun=lambda: 0 * lams_undivided,
                    false_fun=lambda: lams_undivided / alphas[n],
                )

            w = -nabla.betas[n - 1] - ls[:, n - 1].T @ lambda_n
            v = nabla.rs[:, n] - vecmat(lambda_n) + alphas[n] * rho_n
            omega = -rs[:, n].T @ v - betas[n - 1] * w
            undivided_rhos_ = -v - omega * rs[:, n]

            rho_n_minus_one = jax.lax.cond(
                pred=jnp.allclose(betas[n - 1], 0.0),
                true_fun=lambda: 0 * undivided_rhos_,
                false_fun=lambda: undivided_rhos_ / betas[n - 1],
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

    if custom_vjp:
        _bidiagonalize = jax.custom_vjp(
            primal_map,
            nondiff_argnums=(0,),
        )
        _bidiagonalize.defvjp(
            _bidiag_vjp_fwd,
            _bidiag_vjp_bwd,
        )
    else:
        _bidiagonalize = primal_map
    return _bidiagonalize


def bidiagonalize_vjpable(num_matvecs: int, custom_vjp: bool = True):
    @jax.jit
    def _bidiag_vjp_fwd(primals) -> tuple[BidiagOutput, BidiagCache]:
        primal = bidiagonalize(primals, num_matvecs)
        cache = BidiagCache(
            primal=primal,
            A=primals[0],
            start_vector=primals[1],
        )
        return primal, cache

    @jax.jit
    def _bidiag_vjp_bwd(
        cache: BidiagCache,
        nabla: BidiagOutput,
    ) -> BidiagInput:
        # Unpack primal variables from cache. These are 0-indexed.
        betas = cache.primal.betas
        alphas = cache.primal.alphas
        rs = cache.primal.rs
        ls = cache.primal.ls
        c = cache.primal.c
        A = cache.A
        (n, m) = cache.A.shape

        k = num_matvecs

        # Initialize adjoint variables.
        # We use 1-indexing for all of these to follow the math a bit more easily.
        lams = jnp.zeros((n, 1 + k + 1))  # lambdas
        rhos = jnp.zeros((m, 1 + k))  # rhos
        rhos = rhos.at[:, k].set(-nabla.res)
        sigmas = jnp.zeros(k + 1)
        omegas = jnp.zeros(k + 1)

        CarryState = typing.NamedTuple(
            "CarryState",
            [
                ("lams", ArrayLike),
                ("rhos", ArrayLike),
                ("sigmas", ArrayLike),
                ("omegas", ArrayLike),
            ],
        )

        def body_fun(i: int, carry: CarryState):
            # 'i' will go from 0 to k-2 (inclusive)
            # we subtract i from k to go from k to 2 (inclusive)
            ai = k - i
            """adjoint index, 1-based"""
            pi = k - i - 1
            """primal index, 0-based"""

            lams = carry.lams
            rhos = carry.rhos

            t = -nabla.alphas[pi] - rs[:, pi].T @ rhos[:, ai]
            sigmas_ = carry.sigmas.at[ai].set(
                -ls[:, pi].T
                @ (nabla.ls[:, pi] + betas[pi] * lams[:, ai + 1] - A @ rhos[:, ai])
                - alphas[pi] * t
            )

            lams_undivided = (
                -nabla.ls[:, pi]
                - betas[pi] * lams[:, ai + 1]
                + A @ rhos[:, ai]
                - sigmas_[ai] * ls[:, pi]
            )
            divided_lambda = jax.lax.cond(
                pred=jnp.allclose(alphas[pi], 0.0),
                true_fun=lambda: 0 * lams_undivided,
                false_fun=lambda: lams_undivided / alphas[pi],
            )
            lams_ = lams.at[:, ai].set(divided_lambda)

            w = -nabla.betas[pi - 1] - ls[:, pi - 1].T @ lams_[:, ai]
            omegas_ = carry.omegas.at[ai].set(
                -rs[:, pi].T
                @ (nabla.rs[:, pi] - A.T @ lams_[:, ai] + alphas[pi] * rhos[:, ai])
                - betas[pi - 1] * w
            )
            undivided_rhos_ = (
                -nabla.rs[:, pi]
                + A.T @ lams_[:, ai]
                - alphas[pi] * rhos[:, ai]
                - omegas_[ai] * rs[:, pi]
            )

            divided_rho = jax.lax.cond(
                pred=jnp.allclose(betas[pi - 1], 0.0),
                true_fun=lambda: 0 * undivided_rhos_,
                false_fun=lambda: undivided_rhos_ / betas[pi - 1],
            )
            rhos_ = rhos.at[:, ai - 1].set(divided_rho)

            return CarryState(
                lams=lams_,
                rhos=rhos_,
                sigmas=sigmas_,
                omegas=omegas_,
            )

        if num_matvecs > 1:
            output: CarryState = jax.lax.fori_loop(
                lower=0,
                upper=k - 2 + 1,  # (+ 1 to include in iteration)
                body_fun=body_fun,
                init_val=CarryState(
                    lams=lams,
                    rhos=rhos,
                    sigmas=sigmas,
                    omegas=omegas,
                ),
            )

            # output = body_fun(
            #     0,
            #     CarryState(
            #         lams=lams,
            #         rhos=rhos,
            #         sigmas=sigmas,
            #         omegas=omegas,
            #     ),
            # )
            # output = body_fun

        else:
            output: CarryState = CarryState(
                lams=lams,
                rhos=rhos,
                sigmas=sigmas,
                omegas=omegas,
            )

        # last iteration steps:
        ai = 1
        """adjoint index, 1-based"""
        pi = 0
        """primal index, 0-based"""

        if (
            num_matvecs > 1
        ):  # beta is defined from num_matvecs >= 2 # TODO this part is suspicious to me
            beta_times_next_lam = betas[pi] * output.lams[:, ai + 1]
        else:
            beta_times_next_lam = 0 * output.lams[:, ai + 1]

        t = -nabla.alphas[pi] - rs[:, pi].T @ output.rhos[:, ai]

        sigmas_ = output.sigmas.at[ai].set(
            -ls[:, pi].T
            @ (nabla.ls[:, pi] + beta_times_next_lam - A @ output.rhos[:, ai])
            - alphas[pi] * t
        )

        lams_ = output.lams.at[:, ai].set(
            (
                -nabla.ls[:, pi]
                - beta_times_next_lam
                + A @ output.rhos[:, ai]
                - sigmas_[ai] * ls[:, pi]
            )
            / alphas[pi]
        )

        omegas_ = output.omegas.at[ai].set(
            -rs[:, pi].T
            @ (nabla.rs[:, pi] - A.T @ lams_[:, ai] + alphas[pi] * output.rhos[:, ai])
            - c * nabla.c
        )

        kappa = (
            -nabla.rs[:, pi]
            + A.T @ lams_[:, ai]
            - alphas[pi] * output.rhos[:, ai]
            - omegas_[ai] * rs[:, pi]
        )

        # use kappa and rhos and lambdas to compute grads.A and grads.start_vector

        lambda_rs_outer_sum = jnp.einsum("ij, kj -> ik", lams_[:, 1:-1], rs)
        ls_rho_outer_sum = jnp.einsum("ij, kj -> ik", ls, output.rhos[:, 1:])

        gradients = BidiagInput(
            A=-lambda_rs_outer_sum - ls_rho_outer_sum,
            start_vector=-c * kappa,
        )

        return ((gradients.A, gradients.start_vector),)

    if custom_vjp:
        _bidiagonalize = jax.custom_vjp(
            lambda primals: bidiagonalize(primals=primals, num_matvecs=num_matvecs),
            nondiff_argnums=(),
        )
        _bidiagonalize.defvjp(
            _bidiag_vjp_fwd,
            _bidiag_vjp_bwd,
        )
    else:
        _bidiagonalize = lambda primals: bidiagonalize(
            primals=primals, num_matvecs=num_matvecs
        )
    return jax.jit(_bidiagonalize)


@pytest.mark.parametrize("seed", range(5))
def test_shapes(seed):
    (width_rng, height_rng, fill_rng, mask_rng, rand_choice_rng) = jax.random.split(
        jax.random.PRNGKey(seed), num=5
    )

    n = jax.random.randint(key=width_rng, minval=1, maxval=6, shape=())
    m = jax.random.randint(key=height_rng, minval=1, maxval=6, shape=())

    A = jax.random.normal(key=fill_rng, shape=(n, m))
    A = A.at[0, :].set(0)
    start_vector = jax.random.normal(key=rand_choice_rng, shape=(m,))

    num_matvecs = int(jax.random.randint(key=mask_rng, minval=2, maxval=10, shape=()))

    result, tangents = bidiagonalize_jvp(
        (A, start_vector), (A, start_vector), num_matvecs
    )
    assert result.c.shape == ()
    assert result.c.shape == tangents.c.shape
    assert result.alphas.shape == (num_matvecs,)
    assert result.alphas.shape == tangents.alphas.shape
    assert result.betas.shape == (num_matvecs - 1,)
    assert result.betas.shape == tangents.betas.shape
    assert result.res.shape == (m,)
    assert result.res.shape == tangents.res.shape
    assert result.rs.shape == (m, num_matvecs)
    assert result.rs.shape == tangents.rs.shape
    assert result.ls.shape == (n, num_matvecs)
    assert result.ls.shape == tangents.ls.shape


@pytest.mark.parametrize("seed", range(50))
def test_bidiag_properties(seed):
    (width_rng, height_rng, fill_rng, mask_rng, rand_choice_rng) = jax.random.split(
        jax.random.PRNGKey(seed), num=5
    )
    n = jax.random.randint(key=width_rng, minval=1, maxval=4 + 1, shape=())
    m = jax.random.randint(key=height_rng, minval=1, maxval=4 + 1, shape=())
    A = jax.random.normal(key=fill_rng, shape=(n, m))  # random tall-or-square matrix
    if jax.random.uniform(key=mask_rng) < 0.4:
        print("Setting first column to zero")
        A = A.at[:, 0].set(0)

    start_vector = jax.random.normal(key=rand_choice_rng, shape=(m,))
    print("A.shape", A.shape)

    result, _ = bidiagonalize_jvp(
        (A, start_vector),
        (A, start_vector),
        num_matvecs=min(int(n), int(m)),
    )

    result: BidiagOutput = result

    print("result alphas:", result.alphas)
    print("result.L.shape", result.L.shape)
    print("result.B.shape", result.B.shape)
    print("result.R.shape", result.R.shape)

    print("result.iterations_finished", result.iterations_finished)

    print("res", result.res)

    assert np.allclose(
        A.T @ result.L,
        result.R @ result.B.T
        + np.outer(
            result.res,
            np.eye(1, len(result.alphas), k=len(result.alphas) - 1),
        ),
        atol=1e-5,
    ), "A.T L != R B.T + residual"

    assert np.allclose(A @ result.R, result.L @ result.B, atol=1e-5), "AR != LB"

    k = result.iterations_finished

    assert np.allclose(result.L[:, :k].T @ result.L[:, :k], np.eye(k), atol=1e-3), (
        f"L^T L is not identity, {result.L[:, :k].T @ result.L[:, :k]}"
    )

    assert np.allclose(result.R[:, :k].T @ result.R[:, :k], np.eye(k), atol=1e-3), (
        f"R^T R is not identity, {result.R[:, :k].T @ result.R[:, :k]}"
    )


@pytest.mark.parametrize("seed", range(30))
def test_bidiag_jvp_with_autodiff(seed):
    (width_rng, height_rng, fill_rng, mask_rng, rand_choice_rng) = jax.random.split(
        jax.random.PRNGKey(seed), num=5
    )
    n = jax.random.randint(key=width_rng, minval=1, maxval=6, shape=())
    m = jax.random.randint(key=height_rng, minval=1, maxval=6, shape=())
    A = jax.random.normal(key=fill_rng, shape=(n, m))
    d_A = jax.random.normal(key=height_rng, shape=(n, m))

    if jax.random.uniform(key=mask_rng) < 0.4:
        print("Setting first column to zero")
        A = A.at[:, 0].set(0)

    print("Rank:", jnp.linalg.matrix_rank(A))

    start_vector = jax.random.normal(key=rand_choice_rng, shape=(m,))
    d_start_vector = jax.random.normal(key=height_rng, shape=(m,))

    iterations = int(min(A.shape[0], A.shape[1]))

    _, my_tangent = bidiagonalize_jvp(
        primals=(np.array(A), np.array(start_vector)),
        tangents=(np.array(d_A), np.array(d_start_vector)),
        num_matvecs=iterations,
    )

    _, jax_tangent = jax.jvp(
        fun=lambda p: bidiagonalize(primals=p, num_matvecs=iterations),
        primals=((np.array(A), np.array(start_vector)),),
        tangents=((np.array(d_A), np.array(d_start_vector)),),
        has_aux=False,
    )

    assert jnp.allclose(jax_tangent.c, my_tangent.c, atol=1e-6), (
        f"c differs: {jax_tangent.c} vs {my_tangent.c}"
    )
    assert jnp.allclose(jax_tangent.rs, my_tangent.rs, atol=1e-6), (
        f"rs differs: {jax_tangent.rs} vs {my_tangent.rs}"
    )
    assert jnp.allclose(jax_tangent.ls, my_tangent.ls, atol=1e-6), (
        f"ls differs: {jax_tangent.ls} vs {my_tangent.ls}"
    )
    assert jnp.allclose(jax_tangent.alphas, my_tangent.alphas, atol=1e-6), (
        f"alphas differs: {jax_tangent.alphas} vs {my_tangent.alphas}"
    )
    assert jnp.allclose(jax_tangent.betas, my_tangent.betas, atol=1e-6), (
        f"betas differs: {jax_tangent.betas} vs {my_tangent.betas}"
    )
    assert jnp.allclose(jax_tangent.res, my_tangent.res, atol=1e-6), (
        f"res differs: {jax_tangent.res} vs {my_tangent.res}"
    )


@pytest.mark.parametrize("seed", range(10))
def test_bidiag_agrees_with_jvp(seed):
    (width_rng, height_rng, fill_rng, mask_rng, rand_choice_rng) = jax.random.split(
        jax.random.PRNGKey(seed), num=5
    )
    n = jax.random.randint(key=width_rng, minval=1, maxval=6, shape=())
    m = jax.random.randint(key=height_rng, minval=1, maxval=6, shape=())
    A = jax.random.normal(key=fill_rng, shape=(n, m))
    d_A = jax.random.normal(key=mask_rng, shape=(n, m))

    start_vector = jax.random.normal(key=rand_choice_rng, shape=(m,))
    d_start_vector = jax.random.normal(key=height_rng, shape=(m,))

    iterations = min(A.shape[0], A.shape[1])

    def matvec(vec, *params):
        return A @ vec

    vjp_output = bidiagonalize_primal(num_matvecs=iterations)(
        matvec, np.array(start_vector)
    )

    # Get output from bidiagonalize
    bidiag_output = bidiagonalize(
        primals=(np.array(A), np.array(start_vector)),
        num_matvecs=iterations,
    )

    assert np.allclose(bidiag_output.c, vjp_output.c, atol=1e-6), "c values differ"
    assert np.allclose(bidiag_output.rs, vjp_output.rs, atol=1e-6), "rs values differ"
    assert np.allclose(bidiag_output.ls, vjp_output.ls, atol=1e-6), "ls values differ"
    assert np.allclose(bidiag_output.alphas, vjp_output.alphas, atol=1e-6), (
        "alphas values differ"
    )
    assert np.allclose(bidiag_output.betas, vjp_output.betas, atol=1e-6), (
        "betas values differ"
    )
    assert np.allclose(bidiag_output.res, vjp_output.res, atol=1e-6), (
        "res values differ"
    )

    # Get output from bidiagonalize_jvp
    jvp_output, _ = bidiagonalize_jvp(
        primals=(np.array(A), np.array(start_vector)),
        tangents=(np.array(d_A), np.array(d_start_vector)),
        num_matvecs=iterations,
    )

    # Compare all fields
    assert np.allclose(bidiag_output.c, jvp_output.c, atol=1e-6), "c values differ"
    assert np.allclose(bidiag_output.rs, jvp_output.rs, atol=1e-6), "rs values differ"
    assert np.allclose(bidiag_output.ls, jvp_output.ls, atol=1e-6), "ls values differ"
    assert np.allclose(bidiag_output.alphas, jvp_output.alphas, atol=1e-6), (
        "alphas values differ"
    )
    assert np.allclose(bidiag_output.betas, jvp_output.betas, atol=1e-6), (
        "betas values differ"
    )
    assert np.allclose(bidiag_output.res, jvp_output.res, atol=1e-6), (
        "res values differ"
    )
    assert bidiag_output.iterations_finished == jvp_output.iterations_finished, (
        "iterations_finished values differ"
    )


@pytest.mark.parametrize("seed", range(10))
def test_bidiag_vjp_agrees_with_jax(seed):
    (width_rng, height_rng, fill_rng, mask_rng, rand_choice_rng) = jax.random.split(
        jax.random.PRNGKey(seed), num=5
    )
    n = jax.random.randint(key=width_rng, minval=2, maxval=6, shape=())
    m = jax.random.randint(key=height_rng, minval=2, maxval=6, shape=())
    n, m = 5, 5
    A = jax.random.normal(key=fill_rng, shape=(n, m))
    start_vector = jax.random.normal(key=rand_choice_rng, shape=(m,))

    print(f"Matrix A shape: {A.shape}")
    print(f"A: {A}")

    num_matvecs = int(jax.random.randint(key=fill_rng, minval=1, maxval=8, shape=()))
    num_matvecs = min(A.shape[0], A.shape[1])
    print("num_matvecs", num_matvecs)

    # Create a random cotangent vector with the same structure as BidiagOutput
    cotangent = BidiagOutput(
        c=jax.random.normal(key=mask_rng, shape=()),
        res=jax.random.normal(key=height_rng, shape=(m,)),
        rs=jax.random.normal(key=fill_rng, shape=(m, num_matvecs)),
        ls=jax.random.normal(key=rand_choice_rng, shape=(n, num_matvecs)),
        alphas=jax.random.normal(key=width_rng, shape=(num_matvecs,)),
        betas=jax.random.normal(key=mask_rng, shape=(num_matvecs - 1,)),
    )

    _, jax_vjp = jax.vjp(
        lambda p: bidiagonalize(p, int(num_matvecs)), (A, start_vector)
    )

    jax_grads = jax_vjp(cotangent)

    # _, custom_vjp = jax.vjp(
    #     bidiagonalize_vjpable(int(num_matvecs), custom_vjp=True), (A, start_vector)
    # )
    # custom_grads = custom_vjp(cotangent)

    def matvec(v, A):
        return A @ v

    bidiag = bidiagonalize_vjpable_matvec(
        int(num_matvecs),
        custom_vjp=True,
        reorthogonalize=False,  # True
    )

    fwd, custom_vjp_matvec_fun = jax.vjp(
        lambda vec, *params: bidiag(matvec, vec, *params),
        start_vector,
        A,
    )
    dvec, Agrad = custom_vjp_matvec_fun(cotangent)
    # assert False
    # print("A grad:", jax_grads[0][0])
    # print("custom A grad", custom_grads[0][0])
    # print("start vec grad:", jax_grads[0][1])
    # print("custom start vec grad:", custom_grads[0][1])

    print("Agrad", Agrad)
    print("jax_grads[0][0]", jax_grads[0][0])

    # # Compare the gradients
    # assert jnp.allclose(jax_grads[0][0], custom_grads[0][0], atol=1e-6), (
    #     "A gradients differ"
    # )
    # assert jnp.allclose(jax_grads[0][1], custom_grads[0][1], atol=1e-6), (
    #     "start_vector gradients differ"
    # )

    assert jnp.allclose(dvec, jax_grads[0][1], atol=1e-6), (
        "start_vector gradients differ"
    )
    assert jnp.allclose(Agrad, jax_grads[0][0], atol=1e-6), "A gradients differ"


if __name__ == "__main__":
    A = jnp.array([[1.0, 1.0], [0.0, 0.0]])

    vec = jnp.array([3.0, 1.1313424])

    num_matvecs = 1

    (width_rng, height_rng, fill_rng, mask_rng, rand_choice_rng) = jax.random.split(
        jax.random.PRNGKey(2), num=5
    )
    n = 1
    m = 1
    A = jax.random.normal(key=fill_rng, shape=(n, m))
    vec = jax.random.normal(key=rand_choice_rng, shape=(m,))

    # Create a random cotangent vector with the same structure as BidiagOutput
    cotangent = BidiagOutput(
        c=jax.random.normal(key=mask_rng, shape=()),
        res=jax.random.normal(key=height_rng, shape=(m,)),
        rs=jax.random.normal(key=fill_rng, shape=(m, num_matvecs)),
        ls=jax.random.normal(key=rand_choice_rng, shape=(n, num_matvecs)),
        alphas=jax.random.normal(key=width_rng, shape=(num_matvecs,)),
        betas=jax.random.normal(key=mask_rng, shape=(num_matvecs - 1,)),
    )

    def matvec(v, A):
        return A @ v

    # print(jax.jacfwd(matvec, argnums=1)(vec, (A,)))

    bidiag = bidiagonalize_vjpable_matvec(num_matvecs=num_matvecs, custom_vjp=True)
    primal, custom_vjp_matvec_fun = jax.vjp(
        lambda vec, *params: bidiag(matvec, vec, *params),
        vec,
        A,
    )

    (dvec, rest) = custom_vjp_matvec_fun(cotangent)
