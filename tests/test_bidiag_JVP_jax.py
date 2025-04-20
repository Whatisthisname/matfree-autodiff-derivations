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
    r_beta_residual: ArrayLike
    """(m,) vector,  beta_k * r_{k+1}"""
    iterations_finished: int

    def tree_flatten(self):
        children = (
            self.rs,
            self.ls,
            self.alphas,
            self.betas,
            self.c,
            self.r_beta_residual,
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
def bidiagonalize_jvp_jax(
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
    d_ls = jnp.zeros((A.shape[0], size))

    # Compute initial tangent for r1
    d_rs = d_rs.at[:, 1].set(
        (
            d_start_vector
            - start_vector
            * (start_vector.T @ d_start_vector)
            / (start_vector @ start_vector)
        )
        / jnp.linalg.norm(start_vector)
    )

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
        # Forward pass step
        # jax.debug.print("rs[:,n]: {rs}", rs=carry.rs[:, carry.n])
        # jax.debug.print("bs[n-1]: {bs}", bs=carry.bs[carry.n - 1])
        # jax.debug.print("ls[:,n-1]: {ls}", ls=carry.ls[:, carry.n - 1])
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

        # jax.debug.print(
        #     "max iterations: {n_total_iterations}, carry.n: {n}, reach_max_iter: {reach_max_iter}, reach_zero_beta: {reach_zero_beta}, reach_zero_alpha: {reach_zero_alpha}, should_continue: {should_continue}",
        #     n_total_iterations=n_total_iterations,
        #     n=carry.n,
        #     reach_max_iter=reach_max_iter,
        #     reach_zero_beta=reach_zero_beta,
        #     reach_zero_alpha=reach_zero_alpha,
        #     should_continue=should_continue,
        # )
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

    # Create primal output
    primal_output = BidiagOutput(
        c=c,
        r_beta_residual=loop_out.bs[k] * loop_out.rs[:, k + 1],
        rs=loop_out.rs[:, 1:-1],
        ls=loop_out.ls[:, 1:],
        alphas=loop_out.as_[1:],
        betas=loop_out.bs[1:-1],
        iterations_finished=k,
    )

    # Create tangent output
    tangent_output = BidiagOutput(
        c=d_c,
        r_beta_residual=A.T @ loop_out.d_ls[:, k]
        + dA.T @ loop_out.ls[:, k]
        - loop_out.as_[k] * loop_out.d_rs[:, k]
        - loop_out.d_as[k] * loop_out.rs[:, k],
        rs=loop_out.d_rs[:, 1:],
        ls=loop_out.d_ls[:, 1:],
        alphas=loop_out.d_as[1:],
        betas=loop_out.d_bs[1:],
        iterations_finished=k - 1,
    )

    return primal_output, tangent_output


@pytest.mark.parametrize("seed", range(30))
def test_bidiag_jvp(seed):
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

    result, tangents = bidiagonalize_jvp_jax(
        primals=(np.array(A), np.array(start_vector)),
        tangents=(np.array(d_A), np.array(d_start_vector)),
        n_total_iterations=iterations,
    )

    h = 0.000001
    result_wiggled, _ = bidiagonalize_jvp_jax(
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
def test_bidiag_tall_matrix(seed):
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

    result, _ = bidiagonalize_jvp_jax(
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
        np.eye(1, result.iterations_finished, k=result.iterations_finished - 1),
    )

    assert np.allclose(
        A.T @ result.L,
        result.R @ result.B.T
        + np.outer(
            result.r_beta_residual,
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


if __name__ == "__main__":
    (width_rng, height_rng, fill_rng, mask_rng, rand_choice_rng) = jax.random.split(
        jax.random.PRNGKey(1), num=5
    )
    n = 1500
    m = 2500
    A = jax.random.normal(key=fill_rng, shape=(n, m))  # random tall-or-square matrix
    start_vector = jax.random.normal(key=rand_choice_rng, shape=(m,))

    result, _ = bidiagonalize_jvp_jax(
        (A, start_vector),
        (A, start_vector),
        n_total_iterations=min(int(n), int(m)),
    )
    print(result.iterations_finished)
