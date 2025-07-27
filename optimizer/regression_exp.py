from functools import partial
import jax
import jax.numpy as jnp
import os
import optax
import matplotlib.pyplot as plt


def data_and_params_to_pred(data, *params):
    t1, t2, t3, t4 = params
    return t1 + (data - t2)  # * (data - t3) * (data - t4)
    # return (data - intercept) * (weights - intercept) * sqr_weights


def parameter_penalty(input, predictor, *params):
    jac = jax.jacrev(lambda params: predictor(input, *params))(params)
    jac = jnp.stack(jac).T
    res: jnp.linalg.SlogdetResult = jnp.linalg.slogdet(
        jac @ jac.T + jnp.eye(len(input)) * 0.01
    )
    return res.logabsdet


def plot_parameterized_function(ax, predictor, *params):
    """Predictor takes data and *params and produces outputs"""
    xmin, xmax = -2, 2
    ymin, ymax = -10, 10
    more_data = jnp.linspace(xmin, xmax, 50)

    dense_preds = predictor(more_data, *params)

    ax.plot(more_data, dense_preds, label="function")
    ax.hlines(0.0, xmin=xmin, xmax=xmax, linestyle="dotted")
    ax.vlines(0.0, ymin=ymin, ymax=ymax, linestyle="dotted")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    return ax


if __name__ == "__main__":
    ax = plt.subplot()

    data = jnp.linspace(-1, 1, 2)

    plot_parameterized_function(ax, data_and_params_to_pred, *(1.0, 1.0, 1.0, 1.0))
    plt.show()
