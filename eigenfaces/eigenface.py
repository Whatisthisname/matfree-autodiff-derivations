from sklearn.datasets import fetch_lfw_people
import sys

sys.path.append("..")
from pendulum.train_tools import bidiag_module


def matvec(v, mat):
    return mat @ v


baked = bidiag_module.bidiagonalize_vjpable_matvec(num_matvecs=100, custom_vjp=True)


def bidiag_func(vec, mat):
    return baked(matvec, vec, mat)


# bidiag_output = bidiag_func(start_vec, Phi)
# B = jnp.diag(bidiag_output.alphas) + jnp.diag(bidiag_output.betas, 1)
# L = bidiag_output.ls
# R_T = bidiag_output.rs.T
