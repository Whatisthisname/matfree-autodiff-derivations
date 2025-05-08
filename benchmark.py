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
    matvec_nums=[2, 4, 8, 16],
    matrix_sizes=[16, 32, 64, 128],
    repeats=3,
):
    jax_times = np.zeros((len(matrix_sizes), len(matvec_nums)))
    custom_times = np.zeros((len(matrix_sizes), len(matvec_nums)))

    total_iters = len(matrix_sizes) * len(matvec_nums) * repeats
    pbar = tqdm(total=total_iters, desc="Running benchmarks")

    for i, size in enumerate(matrix_sizes):
        for j, matvec_num in enumerate(matvec_nums):
            for _ in range(repeats):
                key = jax.random.PRNGKey(i * 1000 + j)
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

                # JAX VJP
                _, jax_vjp_fn = jax.vjp(
                    lambda p: bidiagonalize(p, matvec_num), (A, start_vector)
                )
                start_time = time.time()
                jax_vjp_fn(cotangent)
                jax_times[i, j] += time.time() - start_time

                # Custom VJP
                _, custom_vjp_fn = jax.vjp(
                    bidiagonalize_vjpable(matvec_num, custom_vjp=True),
                    (A, start_vector),
                )
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

    for i, size in enumerate(matrix_sizes):
        plt.plot(matvec_nums, jax_times[i], label=f"JAX VJP (n={size})", linestyle="--")
        plt.plot(
            matvec_nums, custom_times[i], label=f"Custom VJP (n={size})", linestyle="-"
        )

    plt.xlabel("matvec_num (iterations)")
    plt.ylabel("Gradient runtime (seconds)")
    plt.title("Gradient VJP Runtime Scaling")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


matrix_sizes, matvec_nums, jax_times, custom_times = benchmark_bidiag_gradients()
plot_benchmark_results(matrix_sizes, matvec_nums, jax_times, custom_times)
