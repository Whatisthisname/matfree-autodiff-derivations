import jax
import jax.numpy as jnp
from arnoldi import arnoldi, hessenberg
from matfree import decomp

import os
import sys

jnp.printoptions(precision=2)

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from bidiag import bidiagonalize, BidiagOutput


n = 4
m = 2

matvecs = 1

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


def matvec_sym(v0, *params):
    (A,) = params
    upper, lower = jnp.split(v0, [n])
    return jnp.concat((A @ lower, A.T @ upper))


v = jax.random.normal(jax.random.PRNGKey(0), shape=(m))
v_aug = jnp.concat([jnp.zeros(n), v])

arnoldi_func = hessenberg(2 * matvecs, reortho="full")
arnoldi_result = arnoldi_func(matvec_sym, (n, m), v_aug, A)
bd = bidiagonalize(matvecs, reorthogonalize=False)
bd_result: BidiagOutput = bd(matvec, v, A)


def bidiag_loss(alphas, betas, ls, rs, res, c):
    flattened = jax.flatten_util.ravel_pytree((alphas, betas, ls, rs, res, c))[0]
    return jnp.sum(jax.random.normal(jax.random.PRNGKey(0), len(flattened)) * flattened)


def extract_arnoldi_results(result: decomp._DecompResult) -> tuple:
    rlrlrl = result.Q_tall
    ls = rlrlrl[:n, 1::2]
    rs = rlrlrl[n:, 0::2]
    ababab = jnp.diag(result.J_small, k=1)
    alphas = ababab[::2]
    betas = ababab[1::2]
    res = result.residual[n:]
    return alphas, betas, ls, rs, res, result.init_length_inv


def padded_arnoldi_loss(result: decomp._DecompResult):
    return bidiag_loss(*extract_arnoldi_results(result))


def extract_bidiag_results(result: decomp._DecompResult) -> tuple:
    ls, rs = result.ls, result.rs
    res, c = result.res, result.c
    alphas = result.alphas
    betas = result.betas
    return alphas, betas, ls, rs, res, c


def bidiag_materialized_loss(result: BidiagOutput):
    return bidiag_loss(*extract_bidiag_results(result))


print()
print("Arnoldi VJP:")

arnoldi_loss_val, arnoldi_grad = jax.value_and_grad(padded_arnoldi_loss)(arnoldi_result)

_, vjpfun = jax.vjp(lambda v, A: arnoldi_func(matvec_sym, (n, m), v, A), v_aug, A)
(v_aug_grad, A_aug_grad) = vjpfun(arnoldi_grad)

# print("v_aug_grad grad")
# print(v_aug_grad)
# print("Arnoldi A_aug grad")
# print(A_aug_grad)

print("Bidiag VJP:")
bidiag_loss_val, bidiag_grad = jax.value_and_grad(bidiag_materialized_loss)(bd_result)

_, vjpfun = jax.vjp(lambda v, A: bd(matvec, v, A), v, A)
v_grad, A_grad = vjpfun(bidiag_grad)

# print("v grad")
# print(v_grad)
print("Bidiag A grad")
print(A_grad)

# print(
#     "bidiag loss value and arnoldi loss value:",
#     bidiag_loss_val,
#     arnoldi_loss_val,
#     sep="\n",
# )

stuff1 = extract_bidiag_results(bd_result)
stuff2 = extract_arnoldi_results(arnoldi_result)
names = ("rs", "ls", "alphas", "betas", "res", "c")


def compare(leaf1, leaf2, name):
    assert jnp.allclose(
        leaf1, leaf2, atol=1e-15
    ), f"{name} differ: error magnitude: {jnp.linalg.norm(leaf1-leaf2)}"


jax.tree.map(compare, stuff1, stuff2, names)


def measure(leaf1, leaf2, name):
    return jnp.linalg.norm(leaf1 - leaf2).item()


print("errors:\n", jax.tree.map(measure, stuff1, stuff2, names))
