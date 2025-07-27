from collections import namedtuple
import dataclasses
from functools import partial
import jax
import jax.numpy as jnp
import os
import optax
import matplotlib.pyplot as plt
from regression_exp import (
    parameter_penalty,
    plot_parameterized_function,
)


def data_and_params_to_pred(data, *params):
    t1, t2, t3, t4 = params
    return t1 * (data - t2) * (data - t3) * (data - t4) + data * (t3 - 5) ** 2


def show_clustered_states_and_penalty():
    # sample intercepts and weights:
    seed = 3
    n = 500

    data = jnp.linspace(-1, 1, 2)

    t1 = jax.random.uniform(
        jax.random.PRNGKey(seed), shape=(n), minval=-2.0, maxval=2.0
    )
    t2 = jax.random.uniform(
        jax.random.PRNGKey(seed), shape=(n), minval=-2.0, maxval=2.0
    )
    t3 = jax.random.uniform(
        jax.random.PRNGKey(seed), shape=(n), minval=-2.0, maxval=2.0
    )
    t4 = jax.random.uniform(
        jax.random.PRNGKey(seed), shape=(n), minval=-2.0, maxval=2.0
    )
    zipped = jnp.stack((t1, t2, t3, t4), axis=1)
    # zipped = jnp.stack((intercepts, weights, sqr_weights), axis=1)

    penalties = jax.vmap(
        lambda params: parameter_penalty(data, data_and_params_to_pred, *params)
    )(zipped)

    n_quants = 5

    penalty_quantiles = jnp.quantile(penalties, jnp.linspace(0, 1, n_quants))

    samples = 5
    fig, axs = plt.subplots(samples, n_quants)

    for col, quant in zip(axs.T, penalty_quantiles):
        closest_idx = jnp.argsort(jnp.abs(quant - penalties))[:samples]

        col[0].set_title(f"Penalty ≈ {quant:.2f}")
        for ax, idx in zip(col, closest_idx):
            plot_parameterized_function(ax, data_and_params_to_pred, *zipped[idx])

    plt.show()


@jax.jit
def loss(params, data, lam) -> float:
    x, y = data
    y_hat = data_and_params_to_pred(x, *params)
    mse = jnp.mean((y_hat - y) ** 2)
    penalty = parameter_penalty(x, data_and_params_to_pred, *params)
    return mse + penalty * lam


@jax.jit
def loss2(params, data, lam) -> float:
    x, y = data
    penalty = parameter_penalty(x, data_and_params_to_pred, *params)
    return (penalty - lam) ** 2


Minimized = namedtuple(
    "Minimized", ["final_params", "final_penalty", "all_losses", "all_params"]
)


@jax.jit
def minimize(params, penalty, data) -> Minimized:
    x, y = data
    steps = 50000
    optim = optax.adam(0.03)
    opt_state = optim.init(params)

    grad_fun = jax.value_and_grad(loss, argnums=0)

    def body_fun(carry, _):
        params, opt_state = carry
        value, grad = grad_fun(params, data=(x, y), lam=penalty)

        updates, opt_state = optim.update(grad, opt_state)
        params = optax.apply_updates(params, updates)
        return (params, opt_state), (params, value)

    (final_params, opt_state), (all_params, all_losses) = jax.lax.scan(
        body_fun, init=(params, opt_state), length=steps
    )
    final_penalty = parameter_penalty(x, data_and_params_to_pred, *final_params)

    return Minimized(final_params, final_penalty, all_losses, all_params)


def show_loss_landscape_for_2_params(which: tuple[int, int]):
    x = jnp.linspace(-1.0, 1.0, 2)
    y = 1.0 * x - 2.0

    # Create a grid of parameter values for the first two parameters
    param1_range = jnp.linspace(-10.0, 10.0, 100)
    param2_range = jnp.linspace(-10.0, 10.0, 100)
    param1_grid, param2_grid = jnp.meshgrid(param1_range, param2_range)

    # Fixed values for the other parameters (t3, t4)
    fixed_t3 = 0.0
    fixed_t4 = 0.0
    lam = 0.0

    params = jnp.array([0.5, 0.5, 0.5, 0.5])

    minimized: Minimized = minimize(params, penalty=lam, data=(x, y))

    params = jnp.array([0.0, 0.0, 0.0, 0.0])

    def put(which, first, second, params):
        params = params.at[which[0]].set(first)
        params = params.at[which[1]].set(second)
        return params

    # Compute loss for each combination of param1 and param2
    def compute_loss_grid(param1, param2):
        return jax.vmap(
            jax.vmap(
                lambda p1, p2: loss2(
                    params=put(which, p1, p2, params), data=(x, y), lam=lam
                )
            )
        )(param1, param2)

    loss_grid = compute_loss_grid(param1_grid, param2_grid)

    variance = jnp.var(loss_grid)

    # Create the contour plot
    plt.figure(figsize=(10, 8))
    contour = plt.contourf(param1_grid, param2_grid, loss_grid, levels=20)
    plt.colorbar(contour, label="Loss")
    plt.xlabel("Parameter 1 (t1)")
    plt.ylabel("Parameter 2 (t2)")
    plt.title(
        f"Loss Landscape w.r.t. Parameters 1 and 2 (λ={lam}), Variance = {variance:.2f}"
    )
    plt.grid(True, alpha=0.3)

    # Plot the optimization trajectory
    trajectory_params = minimized.all_params
    param1_trajectory = trajectory_params[:, which[0]]  # First parameter trajectory
    param2_trajectory = trajectory_params[:, which[1]]  # Second parameter trajectory

    # Plot the trajectory with arrows showing direction
    plt.plot(
        param1_trajectory,
        param2_trajectory,
        "r-",
        linewidth=2,
        alpha=0.8,
        label="Optimization trajectory",
        marker="x",
    )

    # Mark start and end points
    plt.plot(
        param1_trajectory[0], param2_trajectory[0], "go", markersize=10, label="Start"
    )
    plt.plot(
        param1_trajectory[-1], param2_trajectory[-1], "ro", markersize=10, label="End"
    )

    plt.legend()
    plt.show()


def show_minimized_losses_for_different_lambda():
    x = jnp.linspace(-1.0, 1.0, 3)
    y = 1.0 * x**3 + 2.0

    n_experiments = 9
    n_repetitions = 7

    fig, axs = plt.subplots(n_repetitions, n_experiments, sharex=True, sharey=True)

    for i in range(n_experiments):
        min, max = 0, 20
        penalty = 10 * 0.65 ** ((i / (n_experiments - 1) * (max - min)) + min)

        losses = jnp.zeros(n_repetitions)
        penalties = jnp.zeros(n_repetitions)
        for j in range(n_repetitions):
            params = 5 * jax.random.normal(
                jax.random.PRNGKey(i * n_repetitions + j), shape=(4)
            )
            minimized: Minimized = minimize(params=params, penalty=penalty, data=(x, y))

            losses = losses.at[j].set(minimized.all_losses[-1])
            penalties = penalties.at[j].set(minimized.final_penalty)

            plot_parameterized_function(
                axs[j, i], data_and_params_to_pred, *minimized.final_params
            )
            axs[j, i].scatter(x, y, marker="x", color="red")
            # axs[j, i].plot(
            #     jnp.linspace(0, 1, len(minimized.all_losses)), minimized.all_losses
            # )

        newline = "\n"
        axs[0, i].set_title(
            rf"$\lambda={penalty:.2f}${newline}Avg loss: {jnp.mean(losses):.3}{newline}Avg penalty: {jnp.mean(penalties):.2f}"
        )

    plt.show()


if __name__ == "__main__":
    # show_loss_landscape_for_2_params(which=(0, 1))
    show_minimized_losses_for_different_lambda()
    # show_clustered_states_and_penalty()
