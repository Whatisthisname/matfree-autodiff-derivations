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
    """(m,) vector,  beta_k * r_{k+1}"""
    iterations_finished: int

    def tree_flatten(self):
        children = (
            self.rs,
            self.ls,
            self.alphas,
            self.betas,
            self.c,
            self.res,
            self.iterations_finished,
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


@partial(jax.jit, static_argnames=("n_total_iterations",))
def bidiagonalize(
    primals: tuple[ArrayLike, ArrayLike],
    n_total_iterations: int,
) -> BidiagOutput:
    """hello"""
    A, start_vector = primals

    c = 1 / jnp.linalg.norm(start_vector)

    size = n_total_iterations + 1

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
            ("n", int),
        ],
    )

    def body_fun(carry: CarryState) -> CarryState:
        n = carry.n + 1

        # Forward pass step
        if True:
            t = A @ carry.rs[:, n] - carry.bs[n - 1] * carry.ls[:, n - 1]

            alpha_k = jnp.linalg.norm(t)
            new_alpha, new_l = jax.lax.cond(
                pred=jnp.allclose(alpha_k, 0, atol=1e-6),  # | jnp.isnan(alpha_k),
                true_fun=lambda: (0.0, jnp.zeros_like(t)),
                false_fun=lambda: (alpha_k, t / alpha_k),
            )

            as_ = carry.as_.at[n].set(new_alpha)
            ls = carry.ls.at[:, n].set(new_l)

            w = A.T @ ls[:, n] - as_[n] * carry.rs[:, n]
            beta_k = jnp.linalg.norm(w)

            new_beta, new_r = jax.lax.cond(
                pred=jnp.allclose(beta_k, 0, atol=1e-6),  # | jnp.isnan(beta_k),
                true_fun=lambda: (0.0, jnp.zeros_like(w)),
                false_fun=lambda: (beta_k, w / beta_k),
            )

            bs = carry.bs.at[n].set(new_beta)
            rs = carry.rs.at[:, n + 1].set(new_r)

        return CarryState(
            rs=rs,
            ls=ls,
            as_=as_,
            bs=bs,
            n=carry.n + 1,
        )

    def cond_fun(carry: CarryState):
        reach_max_iter = carry.n >= n_total_iterations
        reach_zero_beta = jnp.logical_and(
            carry.n >= 2,
            jnp.allclose(carry.bs[carry.n - 1], 0, atol=1e-6),
        )
        reach_zero_alpha = jnp.logical_and(
            carry.n >= 1,
            jnp.allclose(carry.as_[carry.n], 0, atol=1e-6),
        )
        should_continue = jnp.logical_and(
            jnp.logical_not(reach_max_iter),
            jnp.logical_not(
                jnp.logical_or(
                    reach_zero_beta,
                    reach_zero_alpha,
                )
            ),
        )

        return should_continue

    # Run the loop
    loop_out = jax.lax.while_loop(
        cond_fun=cond_fun,
        body_fun=body_fun,
        init_val=CarryState(
            rs=rs,
            ls=ls,
            as_=as_,
            bs=bs,
            n=0,
        ),
    )

    k = loop_out.n

    # Create primal output
    primal_output = BidiagOutput(
        c=c,
        res=loop_out.bs[k] * loop_out.rs[:, k + 1],
        rs=loop_out.rs[:, 1:-1],
        ls=loop_out.ls[:, 1:],
        alphas=loop_out.as_[1:],
        betas=loop_out.bs[1:-1],
        iterations_finished=k,
    )
    return primal_output


@partial(jax.jit, static_argnames=("n_total_iterations",))
def bidiagonalize_jvp(
    primals: tuple[ArrayLike, ArrayLike],
    tangents: tuple[ArrayLike, ArrayLike],
    n_total_iterations: int,
) -> tuple[BidiagOutput, BidiagOutput]:
    A, start_vector = primals
    dA, d_start_vector = tangents

    c = 1 / jnp.linalg.norm(start_vector)

    size = n_total_iterations + 1

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
            ("n", int),
        ],
    )

    def body_fun(carry: CarryState) -> CarryState:
        n = carry.n + 1
        jax.debug.print("Iteration {n}", n=n)

        # Forward pass step
        if True:
            t = A @ carry.rs[:, n] - carry.bs[n - 1] * carry.ls[:, n - 1]

            alpha_k = jnp.linalg.norm(t)
            new_alpha, new_l = jax.lax.cond(
                pred=jnp.allclose(alpha_k, 0, atol=1e-6),  # | jnp.isnan(alpha_k),
                true_fun=lambda: (0.0, jnp.zeros_like(t)),
                false_fun=lambda: (alpha_k, t / alpha_k),
            )

            as_ = carry.as_.at[n].set(new_alpha)
            ls = carry.ls.at[:, n].set(new_l)

            w = A.T @ ls[:, n] - as_[n] * carry.rs[:, n]
            beta_k = jnp.linalg.norm(w)

            new_beta, new_r = jax.lax.cond(
                pred=jnp.allclose(beta_k, 0, atol=1e-6),  # | jnp.isnan(beta_k),
                true_fun=lambda: (0.0, jnp.zeros_like(w)),
                false_fun=lambda: (beta_k, w / beta_k),
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
                pred=jnp.allclose(alpha_k, 0, atol=1e-6) | jnp.isnan(alpha_k),
                true_fun=lambda: jnp.zeros_like(t),
                false_fun=lambda: (d_t - d_alpha_n * ls[:, n]) / alpha_k,
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
                pred=jnp.allclose(beta_k, 0, atol=1e-6) | jnp.isnan(beta_k),
                true_fun=lambda: jnp.zeros_like(w),
                false_fun=lambda: (d_w - d_beta_n * rs[:, n + 1]) / beta_k,
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
            n=carry.n + 1,
        )

    def cond_fun(carry: CarryState):
        reach_max_iter = carry.n >= n_total_iterations
        reach_zero_beta = jnp.logical_and(
            carry.n >= 2,
            jnp.allclose(carry.bs[carry.n - 1], 0, atol=1e-6),
        )
        reach_zero_alpha = jnp.logical_and(
            carry.n >= 1,
            jnp.allclose(carry.as_[carry.n], 0, atol=1e-6),
        )
        should_continue = jnp.logical_and(
            jnp.logical_not(reach_max_iter),
            jnp.logical_not(
                jnp.logical_or(
                    reach_zero_beta,
                    reach_zero_alpha,
                )
            ),
        )

        return should_continue

    # Run the loop
    loop_out = jax.lax.while_loop(
        cond_fun=cond_fun,
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
            n=0,
        ),
    )

    # Compute d_c
    d_c = -(start_vector @ d_start_vector) / (
        start_vector @ start_vector * jnp.linalg.norm(start_vector)
    )
    k = loop_out.n

    d_res = (
        A.T @ loop_out.d_ls[:, k]
        + dA.T @ loop_out.ls[:, k]
        - loop_out.as_[k] * loop_out.d_rs[:, k]
        - loop_out.d_as[k] * loop_out.rs[:, k],
    )

    # Create primal output
    primal_output = BidiagOutput(
        c=c,
        res=loop_out.bs[k] * loop_out.rs[:, k + 1],
        rs=loop_out.rs[:, 1:-1],
        ls=loop_out.ls[:, 1:],
        alphas=loop_out.as_[1:],
        betas=loop_out.bs[1:-1],
        iterations_finished=k,
    )

    # Create tangent output
    tangent_output = BidiagOutput(
        c=d_c,
        res=d_res,
        rs=loop_out.d_rs[:, 1:-1],
        ls=loop_out.d_ls[:, 1:],
        alphas=loop_out.d_as[1:],
        betas=loop_out.d_bs[1:-1],
        iterations_finished=k,
    )

    return primal_output, tangent_output


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass
class BidiagCache:
    primal: BidiagOutput
    A: ArrayLike
    start_vector: ArrayLike


def vjp_forward(primals, n_total_iterations) -> tuple[BidiagOutput, BidiagCache]:
    primal = bidiagonalize(primals, n_total_iterations)
    return primal, primal  # todo make cache


def vjp_backward(
    cache: BidiagCache,
    grads: BidiagOutput,
) -> tuple[BidiagOutput, BidiagOutput]:
    pri = cache.primal
    k = pri.iterations_finished
    (n, m) = cache.A.shape

    # initialize adjoint variables
    l1s = jnp.zeros((n, 1 + k + 1))
    l2s = jnp.zeros((m, 1 + k))
    l2s = l2s.at[:, k].set(-grads.res)
    l3s = jnp.zeros_like(pri.alphas)
    l4s = jnp.zeros_like(pri.betas)

    CarryState = typing.NamedTuple(
        "CarryState",
        [
            ("l1s", ArrayLike),
            ("l2s", ArrayLike),
            ("l3s", ArrayLike),
            ("l4s", ArrayLike),
        ],
    )

    def body_fun(i: int, carry: CarryState):
        # 'i' will go from 0 to k-2.
        # we subtract i from k to go from k to 2 (inclusive)

        t = -grads.alphas[:, k - i] - pri.rs[:, k - i].T @ carry.l2s[:, k - i]
        l3s = carry.l3s.at[:, k - i].set(
            pri.ls[:, k - i].T
            @ (
                cache.A @ carry.l2s[:, k - i]
                - grads.ls[:, k - i]
                - carry.l1s[:, k - i + 1] * pri.betas[k - i]
            )
            - pri.alphas[k - i] * t
        )
        l1s = carry.l1s.at[:, k - i].set(
            (
                -grads.ls[:, k - i]
                - carry.l3s[:, k - i] * pri.ls[:, k - i]
                + cache.A @ carry.l2s[:, k - i]
                - carry.l1s[: k - i + 1] * pri.betas[k - i]
            )
            / pri.alphas[k - i]
        )

        w = -grads.betas[:, k - i - 1] - pri.ls[:, k - i - 1].T @ l1s[:, k - i]
        l4s = carry.l4s.at[:, k - i].set(
            -pri.rs[:, k - i].T
            @ (
                grads.rs[:, k - i]
                + cache.A * carry.l1s[:, k - i]
                + pri.alphas[k - i] * carry.l2s[:, k - i]
            )
            - pri.betas[k - i - 1] * w
        )
        l2s = carry.l2s.at[:, k - i - 1].set(
            (
                -grads.rs[:, k - i]
                - cache.A.T @ carry.l1s[:, k - i]
                - pri.alphas[k - i] * carry.l2s[:, k - i]
                - l4s[:, k - i] * pri.rs[:, k - i]
            )
            / pri.betas[k - i - 1]
        )

        return CarryState(
            l1s=l1s,
            l2s=l2s,
            l3s=l3s,
            l4s=l4s,
        )

    output: CarryState = jax.lax.fori_loop(
        lower=0,
        upper=k - 2 + 1,  # (+ 1 to include in iteration)
        fun=body_fun,
        init_val=CarryState(
            l1s=l1s,
            l2s=l2s,
            l3s=l3s,
            l4s=l4s,
        ),
    )

    output.l4s = output.l4s.at[:, 1].set(
        -pri.rs[:, 1].T
        @ (
            cache.A.T @ output.l1s[:, 1]
            + grads.rs[:, 1]
            + pri.alphas[1] * output.l2s[:, 1]
            + pri.c * grads.c
        )
    )

    l5 = (
        grads.rs[:, 1]
        + cache.A.T @ output.l1s[:, 1]
        + pri.alphas[1] * output.l2s[:, 1]
        + output.l4s[:, 1] * pri.rs[:, 1]
    )

    # TODO: convert these lambdas into grads.A and grads.start_vector, by having a look at the differentiated constraints

    return output


@pytest.mark.parametrize("seed", range(30))
def test_bidiag_jvp_numeric(seed):
    (width_rng, height_rng, fill_rng, mask_rng, rand_choice_rng) = jax.random.split(
        jax.random.PRNGKey(seed), num=5
    )
    n = jax.random.randint(key=width_rng, minval=2, maxval=6, shape=())
    m = jax.random.randint(key=height_rng, minval=2, maxval=6, shape=())
    A = jax.random.normal(key=fill_rng, shape=(n, m))
    d_A = jax.random.normal(key=mask_rng, shape=(n, m))

    print("Rank:", jnp.linalg.matrix_rank(A))

    start_vector = jax.random.normal(key=rand_choice_rng, shape=(m,))
    d_start_vector = jax.random.normal(key=height_rng, shape=(m,))

    iterations = min(A.shape[0], A.shape[1])

    result, tangents = bidiagonalize_jvp(
        primals=(np.array(A), np.array(start_vector)),
        tangents=(np.array(d_A), np.array(d_start_vector)),
        n_total_iterations=iterations,
    )

    h = 0.000001
    result_wiggled, _ = bidiagonalize_jvp(
        primals=(np.array(A + d_A * h), np.array(start_vector + d_start_vector * h)),
        tangents=(np.array(d_A), np.array(d_start_vector)),  # doesn't matter
        n_total_iterations=iterations,
    )

    print(f"-- Field: {'c'}".ljust(20), sep="", end="")
    assert np.allclose((result_wiggled.c - result.c) / h, tangents.c, atol=1e-2), (
        f"c: {(result.c - result_wiggled.c) / h}, {tangents.c}"
    )
    print(" (OK)")

    for idx in range(0, result.iterations_finished):
        for field in ["rs", "alphas", "ls", "betas"]:
            if field == "betas" and idx == result.iterations_finished - 1:
                continue
            print(f"-- Field: {field}[{idx}]".ljust(20), sep="", end="")
            try:
                result_field = getattr(result, field)
                wiggled_field = getattr(result_wiggled, field)
                tangent_field = getattr(tangents, field)

                if field == "rs" or field == "ls":
                    aprox = (wiggled_field[:, idx] - result_field[:, idx]) / h
                    exact = tangent_field[:, idx]
                else:
                    aprox = (wiggled_field[idx] - result_field[idx]) / h
                    exact = tangent_field[idx]

                assert np.allclose(aprox, exact, atol=1e-2), (
                    f"\nApprox: {aprox}, \nExact: {exact}"
                )
                print(" (OK)")
            except IndexError:
                print(" (IndexError)")
                continue


@pytest.mark.parametrize("seed", range(50))
def test_bidiag(seed):
    (width_rng, height_rng, fill_rng, mask_rng, rand_choice_rng) = jax.random.split(
        jax.random.PRNGKey(seed), num=5
    )
    n = jax.random.randint(key=width_rng, minval=2, maxval=4 + 1, shape=())
    m = jax.random.randint(key=height_rng, minval=2, maxval=4 + 1, shape=())
    A = jax.random.normal(key=fill_rng, shape=(n, m))  # random tall-or-square matrix
    # if jax.random.uniform(key=mask_rng) < 0.4:
    #     print("Setting first column to zero")
    #     A = A.at[:, 0].set(0)

    start_vector = jax.random.normal(key=rand_choice_rng, shape=(m,))
    print("A.shape", A.shape)

    result, _ = bidiagonalize_jvp(
        (A, start_vector),
        (A, start_vector),
        n_total_iterations=min(int(n), int(m)),
    )

    result: BidiagOutput = result

    print("result.L.shape", result.L.shape)
    print("result.B.shape", result.B.shape)
    print("result.R.shape", result.R.shape)

    print("result.iterations_finished", result.iterations_finished)

    print(
        (
            np.outer(
                result.res,
                np.eye(1, result.iterations_finished, k=result.iterations_finished - 1),
            )
        ).shape
    )

    assert np.allclose(
        A.T @ result.L,
        result.R @ result.B.T
        + np.outer(
            result.res,
            np.eye(1, result.iterations_finished, k=result.iterations_finished - 1),
        ),
        atol=1e-5,
    ), "A.T L != R B.T + residual"

    assert np.allclose(A @ result.R, result.L @ result.B, atol=1e-5), "AR != LB"

    assert np.allclose(
        result.L.T @ result.L, np.eye(result.iterations_finished), atol=1e-3
    ), f"L^T L is not identity, {result.L.T @ result.L}"

    assert np.allclose(
        result.R.T @ result.R, np.eye(result.iterations_finished), atol=1e-3
    ), f"R^T R is not identity, {result.R.T @ result.R}"


@pytest.mark.parametrize("seed", range(30))
def test_bidiag_jvp_with_autodiff(seed):
    (width_rng, height_rng, fill_rng, mask_rng, rand_choice_rng) = jax.random.split(
        jax.random.PRNGKey(seed), num=5
    )
    n = jax.random.randint(key=width_rng, minval=2, maxval=6, shape=())
    m = jax.random.randint(key=height_rng, minval=2, maxval=6, shape=())
    # n, m = (3, 3)
    A = jax.random.normal(key=fill_rng, shape=(n, m))
    d_A = jax.random.normal(key=mask_rng, shape=(n, m))

    print("Rank:", jnp.linalg.matrix_rank(A))

    start_vector = jax.random.normal(key=rand_choice_rng, shape=(m,))
    # start_vector = jnp.array([1.0, 0.0, 0.0])
    d_start_vector = jax.random.normal(key=height_rng, shape=(m,))

    iterations = int(min(A.shape[0], A.shape[1]))
    # todo clean up
    result, tangents = bidiagonalize_jvp(
        primals=(np.array(A), np.array(start_vector)),
        tangents=(np.array(d_A), np.array(d_start_vector)),
        n_total_iterations=iterations,
    )

    (primal_result, jax_tangent, my_tangent) = jax.jvp(
        fun=lambda p, t: bidiagonalize_jvp(primals=p, tangents=t, n_total_iterations=3),
        primals=(
            (np.array(A), np.array(start_vector)),
            (np.array(d_A), np.array(d_start_vector)),
        ),
        tangents=(
            (np.array(d_A), np.array(d_start_vector)),
            (np.array(d_A), np.array(d_start_vector)),
        ),
        has_aux=True,
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
    assert jnp.allclose(
        jax_tangent.r_beta_residual, my_tangent.r_beta_residual, atol=1e-6
    ), (
        f"r_beta_residual differs: {jax_tangent.r_beta_residual} vs {my_tangent.r_beta_residual}"
    )

    # # if __name__ == "__main__":
    # seed = 1
    # (width_rng, height_rng, fill_rng, mask_rng, rand_choice_rng) = jax.random.split(
    #     jax.random.PRNGKey(seed), num=5
    # )
    # n = jax.random.randint(key=width_rng, minval=2, maxval=6, shape=())
    # m = jax.random.randint(key=height_rng, minval=2, maxval=6, shape=())
    # n, m = (3, 3)
    # A = jax.random.normal(key=fill_rng, shape=(n, m))
    # d_A = jax.random.normal(key=mask_rng, shape=(n, m))

    # output = bidiagonalize()

    # def loss(input: BidiagOutput) -> float:
    #     return input.alphas[0] + input.alphas[1]

    # grads = jax.grad(
    #     fun=loss,
    # )


if __name__ == "__main__":
    seed = 1
    (width_rng, height_rng, fill_rng, mask_rng, rand_choice_rng) = jax.random.split(
        jax.random.PRNGKey(seed), num=5
    )
    n = jax.random.randint(key=width_rng, minval=2, maxval=6, shape=())
    m = jax.random.randint(key=height_rng, minval=2, maxval=6, shape=())
    A = jax.random.normal(key=fill_rng, shape=(n, m))
    d_A = jax.random.normal(key=mask_rng, shape=(n, m))

    print("Rank:", jnp.linalg.matrix_rank(A))

    start_vector = jax.random.normal(key=rand_choice_rng, shape=(m,))
    d_start_vector = jax.random.normal(key=height_rng, shape=(m,))

    iterations = min(A.shape[0], A.shape[1])
    print(A)
    k = bidiagonalize((A, start_vector), 5)
    _, f = jax.vjp(lambda a: bidiagonalize(a, 5), (A, start_vector))
    f(k)
