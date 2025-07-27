from functools import partial
import jax
import jax.flatten_util
import jax.numpy as jnp

# Import matfree modules for matrix-free linear algebra and stochastic trace estimation
from matfree import decomp, funm, stochtrace

jax.config.update("jax_enable_x64", True)


def minimal_jitted_matfree_stochtrace_example():
    """Minimal example to test if matfree's stochastic trace estimation works with JIT."""
    import matfree.stochtrace

    @jax.jit
    def jitted_matfree_test(key):
        # Create a simple matrix-vector product function
        def simple_matvec(v):
            return 2.0 * v  # Simple diagonal matrix

        # Example vector for shape and dtype
        f0_example = jnp.array([1.0, 2.0, 3.0])

        # Use matfree's stochastic trace estimation
        sample_fun = matfree.stochtrace.sampler_normal(f0_example, num=5)
        integrand = matfree.stochtrace.integrand_trace()
        estimator = matfree.stochtrace.estimator(integrand, sampler=sample_fun)

        # Generate a new key
        _, new_key = jax.random.split(key)

        # Try to estimate the trace
        result = estimator(simple_matvec, new_key)
        return result

    # Test the function
    key = jax.random.PRNGKey(0)
    try:
        result = jitted_matfree_test(key)
        print("JIT compilation succeeded!")
        print(f"Result: {result}")
        return True
    except Exception as e:
        print("JIT compilation failed!")
        print(f"Error: {e}")
        return False


def tridiag_extract_inner_from_regularized(
    params,
    key,
    input_data,
    model,
    alpha: float,
    num_matvecs: int,
) -> float:
    def model_forward_pass(params):
        return model.apply(params, input_data, training=True, dropout_key=key)

    # Linearize testfunc at x0: returns the function value and a JVP (Jacobian-vector product) function
    f0, jvp = jax.linearize(model_forward_pass, params)

    # Compute the VJP (vector-Jacobian product) function for testfunc at x0
    _f0, vjp = jax.vjp(model_forward_pass, params)

    # Create the matrix-vector product function for J J^T + 0.1 * I
    def matvec(gradient, /):
        r"""Matrix-vector product with $J J^\top + \alpha I$ for a given vector."""
        vjp_eval = vjp(
            gradient.reshape(1, gradient.shape[0])
        )  # Apply VJP to fx: returns a function to compute J^T v
        matvec_eval = jvp(*vjp_eval)  # Apply JVP to the result: computes J J^T v

        # Add alpha * unflatted for the regularization term (alpha * I)
        flattened_again, _ = jax.flatten_util.ravel_pytree(
            jax.tree.map(lambda x, y: x + alpha * y, matvec_eval, gradient)
        )
        print(flattened_again.shape)
        return flattened_again

    tridiag_sym = decomp.tridiag_sym(num_matvecs)
    decompresult = tridiag_sym(matvec, vec=jax.random.normal(key=key, shape=10))
    logdet = -jnp.linalg.slogdet(decompresult.J_small).logabsdet

    return logdet


def compute_log_determinant_of_regularized_metric_tensor(
    params,
    key,
    input_data,
    model,
    alpha: float,
    num_matvecs: int,
) -> float:
    def model_forward_pass(params):
        return model.apply(params, input_data, training=True, dropout_key=key)

    # Linearize testfunc at x0: returns the function value and a JVP (Jacobian-vector product) function
    f0, jvp = jax.linearize(model_forward_pass, params)
    # Compute the VJP (vector-Jacobian product) function for testfunc at x0
    _f0, vjp = jax.vjp(model_forward_pass, params)

    # Create the matrix-vector product function for J J^T + 0.1 * I
    def matvec(fx, /):
        r"""Matrix-vector product with $J J^\top + \alpha I$ for a given vector fx."""
        vjp_eval = vjp(fx)  # Apply VJP to fx: returns a function to compute J^T v
        matvec_eval = jvp(*vjp_eval)  # Apply JVP to the result: computes J J^T v
        # Add alpha * fx for the regularization term (alpha * I)
        return jax.tree.map(lambda x, y: x + alpha * y, matvec_eval, fx)

    # Create a tridiagonalization function for symmetric matrices (Lanczos algorithm)
    tridiag_sym = decomp.tridiag_sym(num_matvecs)
    # Create the integrand for the log-determinant using the tridiagonalization
    integrand = funm.integrand_funm_sym_logdet(tridiag_sym)
    # Create a normal random vector sampler for stochastic trace estimation
    sample_fun = stochtrace.sampler_normal(f0, num=10)
    # Build the stochastic trace estimator
    estimator = stochtrace.estimator(integrand, sampler=sample_fun)
    # Generate a random key
    _, new_key = jax.random.split(key)
    # Estimate log determinant
    logdet = estimator(matvec, new_key)
    return logdet


if __name__ == "__main__":
    from regularization_experiments import Net

    key = jax.random.PRNGKey(0)
    rngs = {"params": key, "dropout": key}
    input = jax.random.normal(key=key, shape=(1, 28, 28, 1))

    model = Net(tiny=True)
    params = model.init(rngs, input)

    leaves, _ = jax.tree.flatten(
        (jax.tree.map(lambda x: jnp.prod(jnp.array(jnp.shape(x))), params))
    )
    print("parameter_count:", jnp.sum(jnp.stack(leaves)))

    x_batch = jax.random.normal(key=key, shape=(32, 28, 28, 1))
    x_single = jax.random.normal(key=key, shape=(1, 28, 28, 1))

    # For a single image
    logdet = compute_log_determinant_of_regularized_metric_tensor(
        params=params,
        key=key,
        input_data=x_single,
        model=model,
        alpha=0.5,
        num_matvecs=3,
    )

    keys = jax.random.split(key, x_batch.shape[0])
    # For a batch of images
    batched_logdet = jax.vmap(
        lambda x_single, key: compute_log_determinant_of_regularized_metric_tensor(
            model=model,
            params=params,
            input_data=x_single.reshape(1, *x_single.shape),
            key=key,
            num_matvecs=3,
            alpha=0.5,
        ),
        in_axes=(0),
    )(x_batch, keys)

    print("result logdet:", logdet)
    print("result batched:", batched_logdet)

    # # THIS WORKS:
    grad_fun = jax.jit(
        jax.grad(compute_log_determinant_of_regularized_metric_tensor, argnums=0),
        static_argnums=(3, 4, 5),
    )
    # # THIS DOES NOT WORK:
    # grad_fun = jax.grad(
    #     lambda params: jax.jit(
    #         compute_log_determinant_of_regularized_metric_tensor(
    #             model=model,
    #             params=params,
    #             input_data=x_single.reshape(1, *x_single.shape),
    #             key=key,
    #             num_matvecs=3,
    #             alpha=0.5,
    #         )
    #         static_argnums=(3, 4, 5),
    #     ),
    # )

    gradient = grad_fun(
        params,
        key,
        input,
        model,
        0.5,
        3,
    )

    print(gradient)

    # # For reference,

    # f0_flat, unravel_func_f = jax.flatten_util.ravel_pytree(f0)

    # def make_matvec_flat(alpha):
    #     """Create a flattened matrix-vector-product function."""

    #     def fun(f_flat):
    #         """Evaluate a flattened matrix-vector product."""
    #         f_unravelled = unravel_func_f(f_flat)
    #         vjp_eval = vjp(f_unravelled)
    #         matvec_eval = jvp(*vjp_eval)
    #         f_eval, _unravel_func = jax.flatten_util.ravel_pytree(matvec_eval)
    #         return f_eval + alpha * f_flat

    #     return fun

    # matvec_flat = make_matvec_flat(alpha=0.1)
    # M = jax.jacfwd(matvec_flat)(f0_flat)
    # print(jnp.linalg.slogdet(M))
