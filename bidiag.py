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
jnp.printoptions(precision=4)


MatVec = typing.Callable[[ArrayLike], ArrayLike]


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass
class BidiagInput:
    A: ArrayLike
    """(n, m) matrix"""
    v: ArrayLike
    r"""(m,) vector, a.k.a. $\tilde r$"""

    def tree_flatten(self):
        children = (self.A, self.v)
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        A, v = children
        return cls(A=A, v=v)


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


def __bidiagonalize_matvec(num_matvecs: int, reorthogonalize: bool = False):
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
    v, A = primals
    d_v, dA = tangents

    c = 1 / jnp.linalg.norm(v)

    size = num_matvecs + 1

    as_ = jnp.zeros((size))
    bs = jnp.zeros((size))
    rs = jnp.zeros((A.shape[1], size + 1))
    rs = rs.at[:, 1].set(v * c)
    ls = jnp.zeros((A.shape[0], size))

    # Initialize tangent variables
    d_as = jnp.zeros((size))
    d_bs = jnp.zeros((size))
    d_rs = jnp.zeros((A.shape[1], size + 1))
    d_rs = d_rs.at[:, 1].set((d_v - v * (v.T @ d_v) / (v @ v)) / jnp.linalg.norm(v))
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
    d_c = -(v @ d_v) / (v @ v * jnp.linalg.norm(v))

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
        ("v", ArrayLike),
    ],
)

BidiagCache_matvec = typing.NamedTuple(
    "BidiagCache_matvec",
    [
        ("primal", BidiagOutput),
        ("v", ArrayLike),
    ],
)


def bidiagonalize(
    num_matvecs: int,
    custom_vjp: bool = True,
    reorthogonalize: bool = False,
):
    primal_map = __bidiagonalize_matvec(num_matvecs, reorthogonalize)

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
            v=v0,
        )
        return primal, (cache, matvec_params)

    # @partial(jax.jit, static_argnums=(0,))
    def _bidiag_vjp_bwd(
        matvec: MatVec,
        cache_and_params: tuple[BidiagCache_matvec, tuple],
        nabla: BidiagOutput,
    ) -> BidiagInput:
        cache, matvec_params = cache_and_params
        _, vecmat_fun = jax.vjp(lambda v, p: matvec(v, *p), cache.v, matvec_params)

        def vecmat(v):
            return vecmat_fun(v)[0]

        w0_like = jax.eval_shape(matvec, cache.v, *matvec_params)
        (n,) = np.shape(w0_like)
        (m,) = np.shape(cache.v)

        # Unpack primal variables from cache. These are 0-indexed.
        betas = cache.primal.betas
        alphas = cache.primal.alphas
        rs = cache.primal.rs
        ls = cache.primal.ls
        c = cache.primal.c

        # jax.debug.print(
        #     "bdval {}",
        #     rs * nabla.alphas[num_matvecs - 1],
        # )
        # jax.debug.print("alphas\n{}", alphas)

        # jax.debug.print("rs\n{}", rs)
        # jax.debug.print("ls\n{}", ls)
        # jax.debug.print(  #! LAMBDA _ K
        #     "bdval {}",
        #     nabla.res
        #     + rs[:, num_matvecs - 1] * nabla.alphas[num_matvecs - 1]
        #     - rs @ rs.T @ nabla.res,
        # )
        other_lambda_K = (
            nabla.res
            + rs[:, num_matvecs - 1] * nabla.alphas[num_matvecs - 1]
            - rs @ rs.T @ nabla.res
        )

        other_Lambda = jnp.zeros((m, num_matvecs))
        other_Lambda = other_Lambda.at[:, num_matvecs - 1].set(other_lambda_K)

        k = num_matvecs

        CarryState = typing.NamedTuple(
            "CarryState",
            [
                ("lambda_n_plus_one", ArrayLike),
                ("rho_n", ArrayLike),
                ("param_incremental_grads", ArrayLike),
                ("Lambda", ArrayLike),
                ("Rho", ArrayLike),
                ("other_Lambda", ArrayLike),
            ],
        )

        # jax.debug.print("L grad: \n{}", nabla.ls)

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

            lams_undivided = -u_n - sigma * ls[:, n]

            lambda_n = jax.lax.cond(
                pred=jnp.allclose(alphas[n], 0.0),
                true_fun=lambda: 0 * lams_undivided,
                false_fun=lambda: lams_undivided / alphas[n],
            )

            if reorthogonalize:
                jax.debug.print(
                    "constraint before: \n{}",
                    ls.T @ u_n
                    + alphas[n] * ls.T @ lambda_n
                    + sigma * jnp.eye(len(ls))[n],
                    ordered=True,
                )

                lambda_n = (
                    lambda_n
                    - ls @ ls.T @ lambda_n
                    - (1 / alphas[n]) * ls @ (ls.T @ u_n + sigma * jnp.eye(len(ls))[n])
                )

                jax.debug.print(
                    "constraint after : \n{}\n",
                    ls.T @ u_n
                    + alphas[n] * ls.T @ lambda_n
                    + sigma * jnp.eye(len(ls))[n],
                    ordered=True,
                )

            else:
                pass
                # jax.debug.print(
                #     "constraint: \n{}", ls.T @ u_n + alphas[n] * ls.T @ lambda_n
                # )

            Lambda = carry.Lambda.at[:, n].set(lambda_n)

            w = -nabla.betas[n - 1] - ls[:, n - 1].T @ lambda_n
            v = nabla.rs[:, n] - vecmat(lambda_n) + alphas[n] * rho_n
            omega = -rs[:, n].T @ v - betas[n - 1] * w
            undivided_rhos_ = -v - omega * rs[:, n]

            def test(i):
                lambda_k = carry.other_Lambda[:, num_matvecs - 1]
                # jax.debug.print("lambda_k:{}", lambda_k)
                # jax.debug.print(
                #     "!!! BIDIAG VecmatLambda: \n{}", matvec(lambda_k, *matvec_params)
                # )

            jax.lax.cond(
                i_in == 0,
                true_fun=test,
                false_fun=lambda *p: None,
                operand=i_in,
            )

            rho_n_minus_one = jax.lax.cond(
                pred=jnp.allclose(betas[n - 1], 0.0),
                true_fun=lambda: 0 * undivided_rhos_,
                false_fun=lambda: undivided_rhos_ / betas[n - 1],
            )

            Rho = carry.Rho.at[:, n - 1].set(rho_n_minus_one)

            if True:
                pass
                # jax.debug.print(
                #     "da[{0}] = {1}",
                #     n,
                #     nabla.alphas[n] + ls[:, n].T @ lambda_n + rs[:, n].T @ rho_n,
                #     ordered=True,
                # )

                # jax.debug.print(
                #     "‖dl[{0}]‖₂ = {1}",
                #     n,
                #     jnp.linalg.norm(
                #         nabla.ls[:, n]
                #         + alphas[n] * lambda_n
                #         + betas[n] * lambda_n_plus_one
                #         - matvec(rho_n, *matvec_params)
                #         + ls[:, n] * sigma
                #     ),
                #     ordered=True,
                # )

                # jax.debug.print(
                #     "db[{0}] = {1}",
                #     n,
                #     nabla.betas[n - 1]
                #     + ls[:, n - 1].T @ lambda_n
                #     + rs[:, n].T @ rho_n_minus_one,
                #     ordered=True,
                # )

                # jax.debug.print(
                #     "‖dr[{0}]‖₂ = {1}",
                #     n,
                #     jnp.linalg.norm(
                #         nabla.rs[:, n]
                #         - vecmat(lambda_n)
                #         + alphas[n] * rho_n
                #         + betas[n - 1] * rho_n_minus_one
                #         + rs[:, n] * omega
                #     ),
                #     ordered=True,
                # )

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
                Lambda=Lambda,
                Rho=Rho,
                other_Lambda=carry.other_Lambda,
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
                    Lambda=jnp.zeros((n, k)),
                    Rho=jnp.zeros((m, k)).at[:, k - 1].set(-nabla.res),
                    other_Lambda=other_Lambda,
                ),
            )
        else:
            output: CarryState = CarryState(
                lambda_n_plus_one=jnp.zeros(n),
                rho_n=-nabla.res,
                param_incremental_grads=init_param_grads,
                Lambda=jnp.zeros((n, k)),
                Rho=jnp.zeros((m, k)).at[:, k - 1].set(-nabla.res),
                other_Lambda=other_Lambda,
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
            lambda_1_undivided = -u_n - sigma * ls[:, 0]
            lambda_1 = jax.lax.cond(
                pred=jnp.allclose(alphas[0], 0.0),
                true_fun=lambda: 0 * lambda_1_undivided,
                false_fun=lambda: lambda_1_undivided / alphas[0],
            )

        Lambda = output.Lambda.at[:, 0].set(lambda_1)
        ltl = ls.T @ Lambda
        rtr = rs.T @ output.Rho

        jax.debug.print("Lambda: \n{}", Lambda)
        jax.debug.print("Rho   : \n{}", output.Rho)

        v = nabla.rs[:, 0] - vecmat(lambda_1) + alphas[0] * output.rho_n

        omega = -rs[:, 0].T @ v - c * nabla.c

        kappa = -v - omega * rs[:, 0]

        R_grad = -jnp.sum(
            jax.vmap(
                lambda lambda_i, r_i: jnp.outer(lambda_i, r_i),
            )(Lambda.T, rs.T),
            axis=0,
        )
        L_grad = -jnp.sum(
            jax.vmap(
                lambda l_i, rho_i: jnp.outer(l_i, rho_i),
            )(ls.T, output.Rho.T),
            axis=0,
        )

        jax.debug.print(
            "Heyyy,, \n{}",
            -jnp.outer(ls[:, 0], output.Rho[:, 0]) - jnp.outer(Lambda[:, 0], rs[:, 0]),
        )

        jax.debug.print("ls: \n{}", ls)
        jax.debug.print("rs: \n{}", rs)
        jax.debug.print("Partial R grad: \n{}", R_grad)
        jax.debug.print("Partial L grad: \n{}", L_grad)
        # A_grad = R_grad + L_grad

        # jax.debug.print("KKK: {}", jnp.outer(Lambda.T[0], rs.T[0]))

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
