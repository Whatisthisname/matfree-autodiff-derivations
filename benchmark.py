import time
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from tests.test_bidiag_JVP_and_VJP_jax import (
    bidiagonalize_vjpable_matvec,
    bidiagonalize_vjpable,
    BidiagOutput,
    bidiagonalize_matvec,
)


def benchmark_bidiag_gradients(
    matvec_nums,
    matrix_sizes,
    repeats=5,
):
    # Store all runtimes for each configuration
    jax_all_times = np.zeros((len(matrix_sizes), len(matvec_nums), repeats))
    custom_all_times = np.zeros((len(matrix_sizes), len(matvec_nums), repeats))

    total_iters = len(matrix_sizes) * len(matvec_nums) * repeats
    pbar = tqdm(total=total_iters, desc="Running benchmarks")

    for j, matvec_num in enumerate(matvec_nums):
        for i, size in enumerate(matrix_sizes):
            pbar.set_description(f"dim:{size}, depth:{matvec_num}")
            # Create and compile JAX VJP function
            key = jax.random.PRNGKey(0)
            start_vector = jax.random.normal(key, shape=(size,))
            cotangent = BidiagOutput(
                c=jax.random.normal(key, shape=()),
                res=jax.random.normal(key, shape=(size,)),
                rs=jax.random.normal(key, shape=(size, matvec_num)),
                ls=jax.random.normal(key, shape=(size, matvec_num)),
                alphas=jax.random.normal(key, shape=(matvec_num,)),
                betas=jax.random.normal(key, shape=(matvec_num - 1,)),
            )

            # minimalist function that does very little work.
            def matvec(v, s):
                return s * v

            bidiag_custom = bidiagonalize_vjpable_matvec(matvec_num, custom_vjp=True)
            # Create and compile custom VJP function
            _, jax_vjp_fn = jax.vjp(
                lambda vec, *params: bidiag_custom(matvec, vec, *params),
                start_vector,
                jnp.array(0.0),
            )

            # Compile
            jax_vjp_fn = jax.jit(jax_vjp_fn)
            (dv, dA) = jax_vjp_fn(cotangent)
            dv.block_until_ready()
            dA.block_until_ready()

            func = bidiagonalize_vjpable_matvec(matvec_num, custom_vjp=False)
            # Create and compile custom VJP function
            _, custom_vjp_fn = jax.vjp(
                lambda vec, *params: func(matvec, vec, *params),
                start_vector,
                jnp.array(0.0),
            )
            # Compile
            custom_vjp_fn = jax.jit(custom_vjp_fn)
            (dv, dA) = custom_vjp_fn(cotangent)
            dA.block_until_ready()
            dv.block_until_ready()

            # Time repeats with different random inputs
            for r in range(repeats):
                key = jax.random.PRNGKey(r)
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
                (dv, dA) = jax_vjp_fn(cotangent)
                dA.block_until_ready()
                dv.block_until_ready()
                jax_all_times[i, j, r] = time.time() - start_time

                # Time custom VJP
                start_time = time.time()
                (dv, dA) = custom_vjp_fn(cotangent)
                dA.block_until_ready()
                dv.block_until_ready()
                custom_all_times[i, j, r] = time.time() - start_time

                pbar.update(1)

    # Calculate means and standard deviations
    jax_times = np.mean(jax_all_times, axis=2)
    custom_times = np.mean(custom_all_times, axis=2)
    jax_stds = np.std(jax_all_times, axis=2)
    custom_stds = np.std(custom_all_times, axis=2)

    pbar.close()
    return matrix_sizes, matvec_nums, jax_times, custom_times, jax_stds, custom_stds


def plot_benchmark_results(
    matrix_sizes, matvec_nums, jax_times, custom_times, jax_stds, custom_stds
):
    import matplotlib.pyplot as plt

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot scaling with matvec_nums for all matrix sizes
    colors = plt.cm.hsv(np.linspace(0, 0.8, len(matrix_sizes)))
    for i, dim in enumerate(matrix_sizes):
        ax1.errorbar(
            matvec_nums,
            jax_times[i],
            yerr=jax_stds[i],
            # label=f"JAX VJP (dim={dim})",
            linestyle="--",
            color=colors[i],
            capsize=5,
        )
        ax1.errorbar(
            matvec_nums,
            custom_times[i],
            yerr=custom_stds[i],
            label=f"Custom VJP (dim={dim})",
            linestyle="-",
            color=colors[i],
            capsize=5,
        )
    ax1.set_xlabel("Number of iterations")
    ax1.set_ylabel("Gradient runtime (seconds)")
    # ax1.set_yscale("log")
    ax1.set_title("Scaling with iterations")
    ax1.legend(fontsize=8)
    ax1.grid(True)

    colors = plt.cm.hsv(np.linspace(0, 0.8, len(matvec_nums)))
    # Plot scaling with matrix size for all matvec_nums
    for i, matvec_num in enumerate(matvec_nums):
        ax2.errorbar(
            matrix_sizes,
            jax_times[:, i],
            yerr=jax_stds[:, i],
            # label=f"JAX VJP (k={matvec_num})",
            linestyle="--",
            color=colors[i],
            capsize=5,
        )
        ax2.errorbar(
            matrix_sizes,
            custom_times[:, i],
            yerr=custom_stds[:, i],
            label=f"Custom VJP (k={matvec_num})",
            linestyle="-",
            color=colors[i],
            capsize=5,
        )
    ax2.set_xlabel("Vector size n")
    ax2.set_ylabel("Gradient runtime (seconds)")
    # ax2.set_yscale("log")
    ax2.set_title("Scaling with matrix size")
    ax1.legend(fontsize=8)
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


matrix_sizes, matvec_nums, jax_times, custom_times, jax_stds, custom_stds = (
    benchmark_bidiag_gradients(
        matvec_nums=np.linspace(1, 3000, 10, endpoint=True).astype(int),
        matrix_sizes=np.linspace(10000, 10000, 1, endpoint=True).astype(int),
    )
)
plot_benchmark_results(
    matrix_sizes, matvec_nums, jax_times, custom_times, jax_stds, custom_stds
)
