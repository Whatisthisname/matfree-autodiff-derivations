#!/usr/bin/env python

from functools import partial
import typing
import numpy as np
import dataclasses
import pytest
import jax  # type: ignore[import-not-found]
from jax.typing import ArrayLike  # type: ignore[import-not-found]
import jax.numpy as jnp  # type: ignore[import-not-found]


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
    residual: ArrayLike
    """(m,) vector,  beta_k * r_{k+1}"""
    iterations_finished: int

    def tree_flatten(self):
        children = (
            self.rs,
            self.ls,
            self.alphas,
            self.betas,
            self.c,
            self.residual,
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
        return self.ls[:, 1:]

    @property
    def B(self) -> ArrayLike:
        """(k, k) float array"""
        as_diag = jnp.diag(self.alphas[1:])
        for i in range(len(self.betas[1:])):
            as_diag = as_diag.at[i, i + 1].set(self.betas[1:][i])
        return as_diag

    @property
    def R(self) -> ArrayLike:
        """(m, k) float array"""
        return self.rs[:, 1:]


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass
class BidiagonalizeStep:
    A: ArrayLike
    """(n, m) float array"""
    rs: ArrayLike
    """(m, k) float array"""
    ls: ArrayLike
    """(n, k) float array"""
    as_: ArrayLike
    """(k,) float array"""
    bs: ArrayLike
    """(k-1,) float array"""
    residual: ArrayLike
    """(m,) vector,  beta_k * r_{k+1}"""
    iterations_finished: int
    has_exhausted_space: bool
    max_steps: int

    def tree_flatten(self):
        children = (
            self.A,
            self.rs,
            self.ls,
            self.as_,
            self.bs,
            self.residual,
            self.iterations_finished,
            self.has_exhausted_space,
            self.max_steps,
        )
        return (children, None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


def _bidiagonalize_step_body(in_: BidiagonalizeStep) -> BidiagonalizeStep:
    n = in_.iterations_finished + 1

    jax.debug.print(
        "\n\nn: {n}",
        n=n,
    )

    t = in_.A @ in_.rs[:, n] - in_.bs[n - 1] * in_.ls[:, n - 1]
    alpha_k = jnp.linalg.norm(t)

    new_alpha, new_l = jax.lax.cond(
        pred=jnp.allclose(alpha_k, 0, atol=1e-6) | jnp.isnan(alpha_k),
        true_fun=lambda: (0.0, jnp.zeros_like(t)),
        false_fun=lambda: (alpha_k, t / alpha_k),
    )

    in_.as_ = in_.as_.at[n].set(new_alpha)
    in_.ls = in_.ls.at[:, n].set(new_l)

    jax.debug.print(
        "as: {as_}",
        as_=in_.as_[1:].round(3),
    )

    w = in_.A.T @ in_.ls[:, n] - in_.as_[n] * in_.rs[:, n]  # shape (m,)
    in_.residual = w
    beta_k = jnp.linalg.norm(w)

    new_beta, new_r = jax.lax.cond(
        pred=jnp.allclose(beta_k, 0, atol=1e-6) | jnp.isnan(beta_k),
        true_fun=lambda: (0.0, jnp.zeros_like(w)),
        false_fun=lambda: (beta_k, w / beta_k),
    )
    in_.bs = in_.bs.at[n].set(new_beta)
    in_.rs = in_.rs.at[:, n + 1].set(new_r)
    in_.iterations_finished += 1

    jax.debug.print(
        "bs: {bs}",
        bs=in_.bs[1:].round(3),
    )

    # Check if we have exhausted space
    in_.has_exhausted_space = jnp.logical_or(
        jnp.allclose(beta_k, 0, atol=1e-6) | jnp.isnan(beta_k),
        jnp.allclose(alpha_k, 0, atol=1e-6) | jnp.isnan(alpha_k),
    )

    return in_


# @partial(jax.jit, static_argnames=("max_iterations",))
def bidiagonalize_jvp_jax(
    primals: tuple[ArrayLike, ArrayLike],
    tangents: tuple[ArrayLike, ArrayLike],
    max_iterations: int,
) -> tuple[BidiagOutput, BidiagOutput]:
    A, start_vector = primals
    dA, d_start_vector = tangents
    jax.debug.print(
        "A: \n{A}",
        A=A.round(3),
    )

    c = 1 / jnp.linalg.norm(start_vector)
    as_ = jnp.zeros((max_iterations + 1))
    bs = jnp.zeros((max_iterations + 1))
    rs = jnp.zeros((A.shape[1], max_iterations + 1))
    rs = rs.at[:, 1].set(start_vector * c)
    ls = jnp.zeros((A.shape[0], max_iterations + 1))

    output: BidiagonalizeStep = jax.lax.while_loop(
        cond_fun=lambda input: jnp.logical_and(
            input.iterations_finished < max_iterations,
            jnp.logical_not(input.has_exhausted_space),
        ),
        body_fun=_bidiagonalize_step_body,
        init_val=BidiagonalizeStep(
            A=A,
            rs=rs,
            ls=ls,
            as_=as_,
            bs=bs,
            iterations_finished=0,
            residual=jnp.zeros(A.shape[1]),
            has_exhausted_space=False,
            max_steps=max_iterations,
        ),
    )

    primal_output = BidiagOutput(
        c=c,
        residual=output.residual,
        rs=output.rs,
        ls=output.ls,
        alphas=output.as_,
        betas=output.bs,
        iterations_finished=output.iterations_finished,
    )

    return primal_output, primal_output

    # d_as = [0] * len(as_)
    # d_bs = [0] * len(bs)
    # d_rs = [zero_n_vec.copy() * 0 for _ in range(len(rs))]
    # d_rs[1] = (
    #     d_start_vector
    #     - (start_vector * (start_vector.T @ d_start_vector))
    #     / (start_vector @ start_vector)
    # ) / np.linalg.norm(start_vector)
    # d_ls = [zero_m_vec.copy() * 0 for _ in range(len(ls))]

    # # d_rs[1] = d_start_vector, known
    # # d_ls[0] = doesn't matter because bs_[0] = 0
    # # d_bs[0] = 0

    # # In each iteration, assume we already know d_rs[n], d_ls[n-1], d_bs[n-1]
    # for n in range(1, len(as_[1:]) + 1):
    #     d_a_n = ls[n].T @ (dA @ rs[n] + A @ d_rs[n] - d_ls[n - 1] * bs[n - 1])
    #     d_as[n] = d_a_n
    #     d_l_n = (
    #         dA @ rs[n]
    #         + A @ d_rs[n]
    #         - bs[n - 1] * d_ls[n - 1]
    #         - d_as[n] * ls[n]
    #         - d_bs[n - 1] * ls[n - 1]
    #     ) / as_[n]
    #     # there will not be divide by zero here, else the loop would have stopped
    #     d_ls[n] = d_l_n.copy()

    #     # There might not be a beta here, so we have to check.
    #     if n > len(bs[1:]) or np.allclose(bs[n], 0, atol=1e-7) or np.isnan(bs[n]):
    #         break

    #     d_b_n = rs[n + 1].T @ (dA.T @ ls[n] + A.T @ d_ls[n] - d_rs[n] * as_[n])
    #     d_bs[n] = d_b_n
    #     d_r_np1 = (
    #         dA.T @ ls[n]
    #         + A.T @ d_ls[n]
    #         - d_rs[n] * as_[n]
    #         - rs[n] * d_as[n]
    #         - rs[n + 1] * d_bs[n]
    #     ) / bs[n]
    #     # there will not be divide by zero here because we checked above.
    #     d_rs[n + 1] = d_r_np1

    # # print("d_as")
    # # print(np.array(d_as[1:]).round(4))
    # # print("d_bs")
    # # print(np.array(d_bs[1:]).round(4))

    # d_c = -(start_vector @ d_start_vector) / (
    #     start_vector @ start_vector * np.linalg.norm(start_vector)
    # )

    # dL = np.array(d_ls[1:]).T
    # dR = np.array(d_rs[1:]).T

    # if len(as_[1:]) == len(bs[1:]):
    #     dB = np.concatenate((np.diag(d_as[1:]), np.zeros((len(as_[1:]), 1))), axis=1)
    #     for i in range(len(d_bs[1:])):
    #         dB[i, i + 1] = d_bs[1:][i]
    # else:
    #     dB = np.diag(d_as[1:]) + np.diag(d_bs[1:], k=1)

    # tangent_output = BidiagOutput(
    #     rs=d_rs, ls=d_ls, L=dL, B=dB, R=dR, alphas=d_as, betas=d_bs, c=d_c
    # )

    # return primal_output, tangent_output


# @pytest.mark.parametrize("seed", range(50))
# def test_bidiag_jvp(seed):
#     np.random.seed(seed)
#     n = np.random.randint(2, 6)
#     m = np.random.randint(2, 6)
#     A = np.random.randn(n, m)
#     d_A = np.random.randn(n, m)

#     print("Rank:", np.linalg.matrix_rank(A))

#     start_vector = 2 * np.eye(m, 1).flatten()
#     start_vector = np.random.randn(m)
#     d_start_vector = np.random.randn(m)

#     result, tangents = bidiagonalize_jvp(
#         primals=(A, start_vector),
#         tangents=(d_A, d_start_vector),
#         iterations=20,
#     )

#     h = 0.000001
#     result_wiggled, _ = bidiagonalize_jvp(
#         primals=(A + d_A * h, start_vector + d_start_vector * h),
#         tangents=(d_A, d_start_vector),  # doesn't matter
#         iterations=20,
#     )

#     print(f"-- Field: {'c'}".ljust(20), sep="", end="")
#     assert np.allclose((result_wiggled.c - result.c) / h, tangents.c, atol=1e-2)
#     print(" (OK)")

#     for idx in range(1, len(result.rs)):
#         for field in ["rs", "alphas", "ls", "betas"]:
#             print(f"-- Field: {field}[{idx}]".ljust(20), sep="", end="")
#             try:
#                 aprox = (
#                     result_wiggled.__getattribute__(field)[idx]
#                     - result.__getattribute__(field)[idx]
#                 ) / h

#                 exact = tangents.__getattribute__(field)[idx]
#                 assert np.allclose(aprox, exact, atol=1e-2), (
#                     f"\nApprox: {aprox}, \nExact: {exact}"
#                 )
#                 print(" (OK)")
#             except IndexError:
#                 print(" (IndexError)")
#                 continue


# @pytest.mark.parametrize("seed", range(50))
# def test_bidiag_tall_matrix(seed):
#     np.random.seed(seed)
#     n = np.random.randint(low=2, high=8 + 1)
#     m = np.random.randint(low=2, high=n + 1)
#     A = np.random.randn(n, m)  # random tall-or-square matrix
#     # if np.random.rand() < 0.4:
#     # A[:, 0] = 0

#     start_vector = 2 * np.eye(1, m).flatten()
#     print("A.shape", A.shape)

#     result, _ = bidiagonalize_jvp((A, start_vector), (A, start_vector), iterations=20)

#     # print(A)
#     print()
#     # print(result.L @ result.B @ result.R.T)

#     assert np.allclose(result.L @ result.B @ result.R.T, A, atol=1e-3), "A != LBR^T"

#     # Inspect reduced iteration count properties:
#     r = np.random.randint(1, min(m, n))

#     L = result.L[:, :r]
#     B = result.B[:r, :r]
#     R = result.R[:, :r]

#     assert np.allclose(L.T @ L, np.eye(r), atol=1e-5), "L^TL is not identity"
#     assert np.allclose(R.T @ R, np.eye(r), atol=1e-5), "R^TR is not identity"
#     assert np.allclose(A @ R, L @ B, atol=1e-5), "AR != LB"
#     assert np.allclose(
#         A.T @ L,
#         R @ B.T + np.outer(result.betas[r] * result.rs[r + 1], np.eye(1, r, k=r - 1)),
#         atol=1e-5,
#     ), "A.T L != R B.T + extra"
#     assert np.allclose(L.T @ A @ R, B, atol=1e-5), "L^TAR != B"


@pytest.mark.parametrize("seed", range(1))
def test_bidiag_wide_matrix(seed: int):
    (width_rng, height_rng, fill_rng, mask_rng, rand_choice_rng) = jax.random.split(
        jax.random.PRNGKey(seed), num=5
    )
    m = jax.random.randint(key=width_rng, minval=2, maxval=8 + 1, shape=())
    n = jax.random.randint(key=height_rng, minval=2, maxval=m + 1, shape=())
    A = jax.random.normal(key=fill_rng, shape=(n, m))
    if jax.random.uniform(key=mask_rng) < 0.4:
        A.at[:, 0].set(0)

    start_vector = jax.random.normal(key=rand_choice_rng, shape=(m,))

    from test_bidiag_JVP_numpy import bidiagonalize_jvp

    result_jax, _ = bidiagonalize_jvp_jax(
        (np.array(A), np.array(start_vector)),
        (np.array(A), np.array(start_vector)),
        max_iterations=4,
    )
    print("result_jax", result_jax.iterations_finished)
    result, _ = bidiagonalize_jvp(
        primals=(np.array(A), np.array(start_vector)),
        tangents=(np.array(A), np.array(start_vector)),
        iterations=20,
    )

    assert np.allclose(result.L @ result.B @ result.R.T, A, atol=1e-3), "A != LBR^T"

    r = jax.random.randint(key=rand_choice_rng, minval=1, maxval=min(m, n), shape=())

    L = result.L[:, :r]
    B = result.B[:r, :r]
    R = result.R[:, :r]

    assert np.allclose(L.T @ L, np.eye(r), atol=1e-5), "L^TL is not identity"
    assert np.allclose(R.T @ R, np.eye(r), atol=1e-5), "R^TR is not identity"
    assert np.allclose(A @ R, L @ B, atol=1e-5), "AR != LB"
    assert np.allclose(
        A.T @ L,
        R @ B.T + np.outer(result.betas[r] * result.rs[r + 1], np.eye(1, r, k=r - 1)),
        atol=1e-5,
    ), "A.T L != R B.T + extra"


if __name__ == "__main__":
    # Run the test with a specific seed
    test_bidiag_wide_matrix(1)
