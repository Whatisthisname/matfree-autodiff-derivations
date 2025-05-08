from tests.test_bidiag_JVP_and_VJP_jax import BidiagOutput, bidiagonalize_jvp
from test_bidiag_JVP_numpy import bidiagonalize_jvp as bidiagonalize_jvp_npy

import jax
import numpy as np

# set jax precision to 64
jax.config.update("jax_enable_x64", True)


def test_bidiagonalize_jvp_jax_vs_numpy():
    # Use a fixed seed for reproducibility
    key = jax.random.PRNGKey(42)
    primal_key, tangent_key = jax.random.split(key)
    A = jax.random.normal(primal_key, (3, 3))
    dA = jax.random.normal(tangent_key, (3, 3))

    v = jax.random.normal(primal_key, (3,))
    dv = jax.random.normal(tangent_key, (3,))

    n_total_iterations = min(A.shape[0], A.shape[1]) - 1

    (jaxresult_fwd, jaxresult_tan) = bidiagonalize_jvp(
        (A, v), (dA, dv), n_total_iterations=n_total_iterations
    )

    jaxresult_fwd: BidiagOutput = jaxresult_fwd
    jaxresult_tan: BidiagOutput = jaxresult_tan

    np_result_fwd, np_result_tan = bidiagonalize_jvp_npy(
        (A, v), (dA, dv), iterations=n_total_iterations
    )

    ### tangent map
    print("\nNumPy Results:")
    print("d_alphas:", np.array(np_result_tan.alphas)[1:])
    print("d_betas:", np.array(np_result_tan.betas)[1:])
    print("d_rs:\n", np.array(np_result_tan.rs).T[1:])
    print("d_ls:\n", np.array(np_result_tan.ls).T[1:])
    print("d_c:", np_result_tan.c)
    print("\nJAX Results:")
    print("d_alphas:", jaxresult_tan.alphas)
    print("d_betas:", jaxresult_tan.betas)
    print("d_rs:\n", jaxresult_tan.rs)
    print("d_ls:\n", jaxresult_tan.ls)
    print("r_beta_residual:", jaxresult_tan.res)
    print("d_c:", jaxresult_tan.c)
    # # ## forward pass
    # print("\nNumPy Results:")
    # print("alphas:", np.array(np_result_fwd.alphas)[1:])
    # print("betas:", np.array(np_result_fwd.betas)[1:])
    # print("rs:\n", np.array(np_result_fwd.rs).T[:, 1:])
    # print("ls:\n", np.array(np_result_fwd.ls).T[:, 1:])
    # print("c:", np_result_fwd.c)

    # print("\nJAX Results:")
    # print("alphas:", jaxresult_fwd.alphas)
    # print("betas:", jaxresult_fwd.betas)
    # print("rs:\n", jaxresult_fwd.rs)
    # print("ls:\n", jaxresult_fwd.ls)
    # print("r_beta_residual:", jaxresult_fwd.r_beta_residual)
    # print("c:", jaxresult_fwd.c)


if __name__ == "__main__":
    test_bidiagonalize_jvp_jax_vs_numpy()
