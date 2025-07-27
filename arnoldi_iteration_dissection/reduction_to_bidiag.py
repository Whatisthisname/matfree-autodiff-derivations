import jax
import jax.numpy as jnp
from arnoldi import arnoldi, hessenberg
from matfree import decomp

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from bidiag import bidiagonalize, BidiagOutput


n = 4
m = 2

A = jax.random.normal(key=jax.random.PRNGKey(0), shape=(n, m))

A_aug = jnp.block(
    [
        [jnp.zeros((n, n)), A],
        [A.T, jnp.zeros((m, m))],
    ]
)


def matvec(v, *params):
    (A,) = params
    return A @ v


# def matvec_sym(v, *params):
#     (A,) = params
#     # A = jnp.tril(A)
#     return (A @ v + A.T @ v) / 2.0


def matvec_sym(v0, *params):
    (A,) = params
    upper, lower = jnp.split(v0, [n])
    print(len(upper), len(lower))
    return jnp.concat((A @ lower, A.T @ upper))


v = jax.random.normal(jax.random.PRNGKey(0), shape=(m))
v_aug = jnp.concat([jnp.zeros(n), v])

arnoldi_func = hessenberg(4, reortho="full")
arnoldi_result = arnoldi_func(matvec_sym, (n, m), v_aug, A)
# bd = decomp.bidiag(2)
bd = bidiagonalize(3, reorthogonalize=False)
result: BidiagOutput = bd(matvec, v, A)


def bidiag_loss(alphas, betas, ls, rs, res, c):
    flattened = jax.flatten_util.ravel_pytree((alphas, betas, ls, rs, res, c))[0]
    return jnp.sum(jax.random.normal(jax.random.PRNGKey(0), len(flattened)) * flattened)


def padded_arnoldi_loss(result: decomp._DecompResult):
    rlrlrl = result.Q_tall
    rs = rlrlrl[n:, 0::2]
    ls = rlrlrl[:n, 1::2]
    ababab = jnp.diag(result.J_small, k=1)
    alphas = ababab[::2]
    betas = ababab[1::2]
    res = result.residual[n:]
    return bidiag_loss(alphas, betas, ls, rs, res, result.init_length_inv)


def bidiag_materialized_loss(result: BidiagOutput):
    ls, rs = result.ls, result.rs
    res, c = result.res, result.c
    alphas = result.alphas
    betas = result.betas
    return bidiag_loss(alphas, betas, ls, rs, res, c)


arnoldi_loss_val, arnoldi_grad = jax.value_and_grad(padded_arnoldi_loss)(arnoldi_result)

_, vjpfun = jax.vjp(lambda v, A: arnoldi_func(matvec_sym, (n, m), v, A), v_aug, A)
(v_aug_grad, A_aug_grad) = vjpfun(arnoldi_grad)

bidiag_loss_val, bidiag_grad = jax.value_and_grad(bidiag_materialized_loss)(result)

_, vjpfun = jax.vjp(lambda v, A: bd(matvec, v, A), v, A)
v_grad, A_grad = vjpfun(bidiag_grad)

print(v_aug_grad)
print(v_grad)
assert jnp.allclose(v_aug_grad[n:], v_grad)

assert jnp.allclose(A_grad, A_aug_grad)
