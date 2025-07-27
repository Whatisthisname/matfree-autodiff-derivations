#!/usr/bin/env python
from functools import partial
import typing
import pytest
import jax  # type: ignore[import-not-found]
from jax.typing import ArrayLike  # type: ignore[import-not-found]
import jax.numpy as jnp  # type: ignore[import-not-found]
from jax.numpy import array as ø
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import bidiag

jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_debug_nans", True)


def test_shape_match(bidiag_input):
    A = bidiag_input["A"]
    start_vector = bidiag_input["v"]
    num_matvecs = bidiag_input["num_matvecs"]
    m, n = bidiag_input["m"], bidiag_input["n"]

    result = bidiag.bidiagonalize(num_matvecs, custom_vjp=False, reorthogonalize=True)(
        lambda v: A @ v, start_vector
    )
    assert result.c.shape == ()
    assert result.alphas.shape == (num_matvecs,)
    assert result.betas.shape == (num_matvecs - 1,)
    assert result.res.shape == (m,)
    assert result.rs.shape == (m, num_matvecs)
    assert result.ls.shape == (n, num_matvecs)


def test_primal_constraints(bidiag_input):
    A = bidiag_input["A"]
    v = bidiag_input["v"]
    num_matvecs = bidiag_input["num_matvecs"]

    # result, _ = bidiagonalize_jvp(
    #     (A, start_vector),
    #     (A, start_vector),
    #     num_matvecs=num_matvecs,
    # )
    func = bidiag.bidiagonalize(
        num_matvecs=num_matvecs, custom_vjp=True, reorthogonalize=True
    )
    result = func(lambda v: A @ v, v)

    result: bidiag.BidiagOutput = result

    assert jnp.allclose(
        A.T @ result.L,
        result.R @ result.B.T
        + jnp.outer(
            result.res,
            jnp.eye(1, len(result.alphas), k=len(result.alphas) - 1),
        ),
        atol=1e-5,
    ), "A.T L != R B.T + residual"

    assert jnp.allclose(A @ result.R, result.L @ result.B, atol=1e-5), "AR != LB"

    k = result.iterations_finished

    assert jnp.allclose(
        result.L[:, :k].T @ result.L[:, :k], jnp.eye(k), atol=1e-3
    ), f"L^T L is not identity, {result.L[:, :k].T @ result.L[:, :k]}"

    assert jnp.allclose(
        result.R[:, :k].T @ result.R[:, :k], jnp.eye(k), atol=1e-3
    ), f"R^T R is not identity, {result.R[:, :k].T @ result.R[:, :k]}"


@pytest.fixture(params=range(1, 11))
def seed(request) -> int:
    return request.param


@pytest.fixture
def bidiag_input(seed):
    # Split RNG keys for different random operations
    (width_rng, height_rng, fill_rng, mask_rng, rand_choice_rng) = jax.random.split(
        jax.random.PRNGKey(seed), num=5
    )

    # Generate random matrix dimensions between 1 and 6
    n = jax.random.randint(key=width_rng, minval=1, maxval=6, shape=()) + 8
    m = jax.random.randint(key=height_rng, minval=1, maxval=6, shape=()) + 8

    # Generate random matrix A and its tangent d_A
    A = jax.random.normal(key=fill_rng, shape=(n, m))
    dA = jax.random.normal(key=height_rng, shape=(n, m))

    # Randomly zero out first column with 40% probability
    if jax.random.uniform(key=mask_rng) < 0.4:
        print("Setting first column to zero")
        A = A.at[:, 0].set(0)

    print("Matrix dimensions:", A.shape)
    print("Matrix rank:", jnp.linalg.matrix_rank(A))

    # Generate random start vector and its tangent
    v = jax.random.normal(key=rand_choice_rng, shape=(m,))
    dv = jax.random.normal(key=height_rng, shape=(m,))

    # Number of iterations is min dimension
    num_matvecs = int(min(n, m))

    return {
        "A": A,
        "d_A": dA,
        "v": v,
        "dv": dv,
        "num_matvecs": num_matvecs,
        "n": n,
        "m": m,
    }


def _compare_bidiag_outputs(
    output1: bidiag.BidiagOutput,
    output2: bidiag.BidiagOutput,
    atol: float = 1e-6,
):
    names = ("rs", "ls", "alphas", "betas", "c", "res")
    names = bidiag.BidiagOutput(*names)

    def compare(leaf1, leaf2, name):
        assert jnp.allclose(
            leaf1, leaf2, atol=atol
        ), f"{name} differ: error magnitude: {jnp.linalg.norm(leaf1-leaf2)}"

    jax.tree.map(compare, output1, output2, names)


def _compare_bidiag_gradients(
    vgrad1: ArrayLike,
    vgrad2: ArrayLike,
    paramgrad1: ArrayLike,
    paramgrad2: ArrayLike,
    atol: float = 1e-6,
):
    """Compare two BidiagOutput objects for approximate equality."""
    assert jnp.allclose(vgrad1, vgrad2, atol=atol), "v gradients differ"
    assert jnp.allclose(paramgrad1, paramgrad2, atol=atol), "parameter gradients differ"


def _dont_test_jvp(bidiag_input):
    A = bidiag_input["A"]
    d_A = bidiag_input["d_A"]
    start_vector = bidiag_input["v"]
    d_start_vector = bidiag_input["dv"]
    num_matvecs = bidiag_input["num_matvecs"]

    print("Matrix dimensions:", A.shape)
    print("Matrix rank:", jnp.linalg.matrix_rank(A))

    primals = ø(start_vector), ø(A)
    tangents = ø(d_start_vector), ø(d_A)

    _, my_tangent = bidiag.bidiagonalize_jvp(
        primals=primals,
        tangents=tangents,
        num_matvecs=num_matvecs,
    )

    func = bidiag.__bidiagonalize_matvec(num_matvecs=num_matvecs, reorthogonalize=False)

    primal, jax_tangent = jax.jvp(
        fun=lambda v, param: func(lambda v_, A: A @ v_, v, param),
        primals=primals,
        tangents=tangents,
    )
    print(jax.tree.structure(my_tangent))
    print(jax.tree.structure(jax_tangent))

    _compare_bidiag_outputs(my_tangent, jax_tangent)


@pytest.fixture
def bidiag_cotangent(bidiag_input, seed):
    # Use a new RNG key for cotangent generation
    (cotangent_rng,) = jax.random.split(jax.random.PRNGKey(seed + 1000), num=1)

    # Random dimensions
    n = bidiag_input["n"]
    m = bidiag_input["m"]
    num_matvecs = bidiag_input["num_matvecs"]

    # Random cotangent vector with the same structure as BidiagOutput
    return bidiag.BidiagOutput(
        c=jax.random.normal(key=cotangent_rng, shape=()),
        res=jax.random.normal(key=cotangent_rng, shape=(m,)),
        rs=jax.random.normal(key=cotangent_rng, shape=(m, num_matvecs)),
        ls=jax.random.normal(key=cotangent_rng, shape=(n, num_matvecs)),
        alphas=jax.random.normal(key=cotangent_rng, shape=(num_matvecs,)),
        betas=jax.random.normal(key=cotangent_rng, shape=(num_matvecs - 1,)),
    )


def test_vjp(bidiag_input, bidiag_cotangent):
    A = bidiag_input["A"]
    v = bidiag_input["v"]
    num_matvecs = bidiag_input["num_matvecs"]

    print(f"Matrix A shape: {A.shape}")
    print(f"v shape: {v.shape}")
    print(f"A: {A}")
    print("num_matvecs", num_matvecs)

    # --- matvec bidiag with autodiff
    def matvec(v, A):
        return A @ v

    bidiag_func = bidiag.bidiagonalize(
        int(num_matvecs),
        custom_vjp=False,  # False!
        reorthogonalize=True,
    )
    _, jax_vjp = jax.vjp(lambda inp, *params: bidiag_func(matvec, inp, *params), v, A)

    (dv_matvec_autodiff, dA_matvec_autodiff) = jax_vjp(bidiag_cotangent)

    # --- matvec bidiag with custom vjp
    bidiag_func = bidiag.bidiagonalize(
        int(num_matvecs),
        custom_vjp=True,  # True!
        reorthogonalize=True,
    )
    _, custom_vjp_matvec_fun = jax.vjp(
        lambda vec, *params: bidiag_func(matvec, vec, *params),
        v,
        A,
    )
    (dv_matvec_custom, dA_matvec_custom) = custom_vjp_matvec_fun(bidiag_cotangent)

    # --- comparing
    _compare_bidiag_gradients(
        dv_matvec_custom, dv_matvec_autodiff, dA_matvec_custom, dA_matvec_autodiff
    )

    # assert False
