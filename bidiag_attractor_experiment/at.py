import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from arnoldi_bidiag import arnoldi_bidiag


def rmult_inv(term, to_inv):
    return jnp.linalg.solve(to_inv.T, term.T).T


def eigvectors(dim):
    return jnp.eye(dim)


def eigvalues(dim):
    return jnp.diag(jnp.array([10] + (dim - 2) * [1] + [10]))


def run() -> list:
    dim = 10
    eigenvectors = eigvectors(dim)
    eigenvalues = eigvalues(dim)
    A = rmult_inv(eigenvectors @ eigenvalues, eigenvectors)

    bidiag = arnoldi_bidiag.arnoldi_bidiagonalization(num_matvecs=dim, output_size=dim)

    results = []
    for k in range(10):
        key = jax.random.PRNGKey(k)
        v0 = jax.random.ball(key, d=dim)
        result: arnoldi_bidiag.DecompResult = bidiag(lambda v, p: A @ v, v0, ())
        results.append({"start_vec": v0, "output": result})
    return results
