from functools import partial
import jax
import jax.numpy as jnp
import os
import optax
import matplotlib.pyplot as plt


jax.config.update("jax_enable_x64", True)

from robot_arm_exp import (
    configuration_penalty,
    configuration_penalty_hessian,
    robot_arm_position,
)


ROBOT_ARM_DOF = 3
SAMPLES = 100000


def get_robot_arm(angles):
    """Draw a robot arm configuration from global angles.

    Args:
        angles: Array of DIM angles in global coordinates

    Returns:
        Tuple of (x_points, y_points) arrays for plotting
    """
    # Start at origin
    x_points = [0.0]
    y_points = [0.0]

    # Length of each segment (same as in exp.py)
    segment_length = 1.2 / ROBOT_ARM_DOF

    # Add points for each segment
    for angle in angles:
        # Get previous endpoint
        prev_x = x_points[-1]
        prev_y = y_points[-1]

        # Calculate new endpoint using angle
        x = prev_x + segment_length * jnp.cos(angle)
        y = prev_y + segment_length * jnp.sin(angle)

        x_points.append(x)
        y_points.append(y)

    return jnp.array(x_points), jnp.array(y_points)


def plot_robot_arm(ax, angles, color):
    """Plot a robot arm configuration on the given axis.

    Args:
        ax: Matplotlib axis to plot on
        angles: Array of angles for the robot arm
        color: Color for the arm segments
    """
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    x_points, y_points = get_robot_arm(angles)
    # Plot arm segments
    ax.plot(x_points, y_points, linewidth=2, marker="o", markersize=4, color=color)
    # Add arrow at tip
    tip_x, tip_y = x_points[-1], y_points[-1]
    direction_x = x_points[-1] - x_points[-2]
    direction_y = y_points[-1] - y_points[-2]
    ax.arrow(
        tip_x - 0.1 * direction_x,
        tip_y - 0.1 * direction_y,
        0.1 * direction_x,
        0.1 * direction_y,
        head_width=0.05,
        head_length=0.1,
        fc=color,
        ec=color,
    )


def run_clustered_states_experiment():
    """Run the robot arm clustered states experiment."""
    # Sample 100 x DIM random angles uniformly between 0 and 2π
    key = jax.random.PRNGKey(1)
    angles = jax.random.uniform(
        key=key, shape=(SAMPLES, ROBOT_ARM_DOF), minval=0.0, maxval=2.0 * jnp.pi
    )
    LENGTHS = jnp.array([1.0] * ROBOT_ARM_DOF) * (1.2 / ROBOT_ARM_DOF)
    # Each angle maps to a robot arm state.

    # Compute configuration penalty for each set of angles
    penalties = jax.vmap(lambda angle: configuration_penalty(angle, LENGTHS))(angles)

    # Find indices where penalties are NaN
    nan_indices = jnp.isnan(penalties)
    print(nan_indices.shape)

    print("nan percentage:", jnp.sum(nan_indices) / SAMPLES)

    # Remove angles where penalties are NaN
    angles = angles[~nan_indices]
    penalties = penalties[~nan_indices]

    angles -= angles[:, 0:1] + 0.1 * jax.random.uniform(
        key=jax.random.PRNGKey(0), shape=(len(angles), 1), minval=-1, maxval=1
    )
    # rotate all arms so that base segment points right.

    PLOTS = 5

    # Set up the figure and subplots
    penalty_quantiles = jnp.quantile(penalties, jnp.linspace(0, 1, PLOTS))
    print(penalty_quantiles)
    fig, axs = plt.subplots(1, PLOTS, figsize=(15, 5))

    # For each cluster, in order of increasing penalty
    for plot_idx, penalty_quantile in enumerate(penalty_quantiles):
        # Get some arms with penalties closest to cluster center
        sample_indices = jnp.argsort(jnp.abs(penalties - penalty_quantile))[:3]

        # Plot each sampled arm
        for i, idx in enumerate(sample_indices):
            plot_robot_arm(axs[plot_idx], angles[idx], f"C{i}")

        # Set title and limits for each subplot
        axs[plot_idx].set_title(f"Penalty ≈ {penalty_quantile:3f}")
        axs[plot_idx].set_xlim([-1.5, 1.5])
        axs[plot_idx].set_ylim([-1.5, 1.5])
        axs[plot_idx].set_aspect("equal")

    # Adjust layout and display
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_clustered_states_experiment()
