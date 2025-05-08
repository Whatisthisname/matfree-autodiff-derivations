import time
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from tests.test_bidiag_JVP_and_VJP_jax import (
    bidiagonalize_vjpable,
    BidiagOutput,
    bidiagonalize,
)


def benchmark_bidiag_gradients(
    matvec_nums,
    matrix_sizes,
    repeats=10,
):
    jax_times = np.zeros((len(matrix_sizes), len(matvec_nums)))
    custom_times = np.zeros((len(matrix_sizes), len(matvec_nums)))

    total_iters = len(matrix_sizes) * len(matvec_nums) * repeats
    pbar = tqdm(total=total_iters, desc="Running benchmarks")

    for i, size in enumerate(matrix_sizes):
        for j, matvec_num in enumerate(matvec_nums):
            # Create and compile JAX VJP function
            key = jax.random.PRNGKey(0)
            A = jax.random.normal(key, shape=(size, size))
            start_vector = jax.random.normal(key, shape=(size,))
            cotangent = BidiagOutput(
                c=jax.random.normal(key, shape=()),
                res=jax.random.normal(key, shape=(size,)),
                rs=jax.random.normal(key, shape=(size, matvec_num)),
                ls=jax.random.normal(key, shape=(size, matvec_num)),
                alphas=jax.random.normal(key, shape=(matvec_num,)),
                betas=jax.random.normal(key, shape=(matvec_num - 1,)),
            )

            _, jax_vjp_fn = jax.vjp(
                lambda p: bidiagonalize(p, matvec_num), (A, start_vector)
            )

            # Compile
            jax_vjp_fn = jax.jit(jax_vjp_fn)
            jax_vjp_fn(cotangent)

            # Create and compile custom VJP function
            _, custom_vjp_fn = jax.vjp(
                bidiagonalize_vjpable(matvec_num, custom_vjp=True),
                (A, start_vector),
            )
            # Compile
            custom_vjp_fn(cotangent)

            # Time repeats with different random inputs
            for r in range(repeats):
                key = jax.random.PRNGKey(r)
                A = jax.random.normal(key, shape=(size, size))
                start_vector = jax.random.normal(key, shape=(size,))
                cotangent = BidiagOutput(
                    c=jax.random.normal(key, shape=()),
                    res=jax.random.normal(key, shape=(size,)),
                    rs=jax.random.normal(key, shape=(size, matvec_num)),
                    ls=jax.random.normal(key, shape=(size, matvec_num)),
                    alphas=jax.random.normal(key, shape=(matvec_num,)),
                    betas=jax.random.normal(key, shape=(matvec_num - 1,)),
                )

                # Time JAX VJP
                start_time = time.time()
                jax_vjp_fn(cotangent)
                jax_times[i, j] += time.time() - start_time

                # Time custom VJP
                start_time = time.time()
                custom_vjp_fn(cotangent)
                custom_times[i, j] += time.time() - start_time

                pbar.update(1)

            jax_times[i, j] /= repeats
            custom_times[i, j] /= repeats

    pbar.close()
    return matrix_sizes, matvec_nums, jax_times, custom_times


def plot_benchmark_results(matrix_sizes, matvec_nums, jax_times, custom_times):
    import matplotlib.pyplot as plt

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot scaling with matvec_nums (using first matrix size)
    ax1.plot(matvec_nums, jax_times[0], label="JAX VJP", linestyle="--")
    ax1.plot(matvec_nums, custom_times[0], label="Custom VJP", linestyle="-")
    ax1.set_xlabel("Number of iterations")
    ax1.set_ylabel("Gradient runtime (seconds)")
    ax1.set_yscale("log")
    ax1.set_title(f"Scaling with iterations\n(n={matrix_sizes[0]})")
    ax1.legend()
    ax1.grid(True)

    # Plot scaling with matrix size (using first matvec_num)
    ax2.plot(matrix_sizes, jax_times[:, 0], label="JAX VJP", linestyle="--")
    ax2.plot(matrix_sizes, custom_times[:, 0], label="Custom VJP", linestyle="-")
    ax2.set_xlabel("Matrix dimension n")
    ax2.set_ylabel("Gradient runtime (seconds)")
    ax2.set_yscale("log")
    ax2.set_title(f"Scaling with matrix size\n(iterations={matvec_nums[0]})")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


matrix_sizes, matvec_nums, jax_times, custom_times = benchmark_bidiag_gradients(
    matvec_nums=[100],
    matrix_sizes=np.linspace(100, 5000, 10).astype(int),
)
plot_benchmark_results(matrix_sizes, matvec_nums, jax_times, custom_times)
