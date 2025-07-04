from functools import partial
import jax
import jax.numpy as jnp
import plotly.graph_objects as go
import os
import optax

# degrees of freedom in robot arm
ROBOT_ARM_DOF = 5

N_BUMPS_IN_LOSS = 2 * 40

# loss function constants

locs = jnp.concat(
    (
        jax.random.uniform(
            key=jax.random.PRNGKey(2),
            shape=(N_BUMPS_IN_LOSS // 2, 2),
            minval=-1.2,
            maxval=1.2,
        ),
        jax.random.normal(key=jax.random.PRNGKey(3), shape=(N_BUMPS_IN_LOSS // 2, 2)),
    ),
)

rotation_angle = jax.random.uniform(
    key=jax.random.PRNGKey(0), shape=(N_BUMPS_IN_LOSS), minval=0.0, maxval=2.0 * jnp.pi
)


def rot_matrix(angle_rad: float):
    return jnp.array(
        [
            [jnp.cos(angle_rad), -jnp.sin(angle_rad)],
            [jnp.sin(angle_rad), jnp.cos(angle_rad)],
        ]
    )


eigvectors = [rot_matrix(angle) for angle in rotation_angle]

scales = (
    jax.random.uniform(
        key=jax.random.PRNGKey(0), shape=(N_BUMPS_IN_LOSS, 2), minval=1.0, maxval=5.0
    )
    * 0.008
)

covariances = [
    eig @ jnp.diag(scale) @ jnp.linalg.inv(eig)
    for eig, scale in zip(eigvectors, scales)
]


@jax.jit
def loss_func(
    x,
):
    terms = jnp.array(
        [
            jax.scipy.stats.multivariate_normal.pdf(x, mean=loc, cov=cov)
            for loc, cov in zip(locs, covariances)
        ]
    )

    return 0.1 * jnp.sum(terms) + 3 * jnp.linalg.norm(x) ** 2


# JIT compile the gradient function
jit_grad_loss = jax.jit(jax.grad(loss_func))

# JIT compile the Jacobian function
jit_jacobian_loss = jax.jit(jax.jacobian(loss_func))


@jax.jit
def robot_arm_position(angles, lengths):
    base_position = jnp.array([0.0] * 2)
    directions = jnp.stack((jnp.cos(angles), jnp.sin(angles))).T

    pos = base_position + jnp.sum(lengths.reshape(-1, 1) * directions, axis=0)
    return pos


@jax.jit
def configuration_penalty(angles, lengths):
    jac = jax.jacrev(robot_arm_position, argnums=0)(angles, lengths)
    return -jnp.log(
        jnp.linalg.det(jac.T @ jac + jnp.eye(len(angles)) * 0.01)
    )  # 0.0 because unnecessary


@jax.jit
def configuration_penalty_hessian(angles, lengths):
    return -jnp.log(
        jnp.abs(
            jnp.linalg.det(jax.hessian(robot_arm_position, argnums=0)(angles, lengths))
        )
    )


@jax.jit
def angle_loss(angles, loss_param):
    location = robot_arm_position(angles=angles, lengths=LENGTHS)
    # return jnp.linalg.norm(angles), location
    end_effector_loss = loss_func(location)
    configuration_loss = configuration_penalty(angles, LENGTHS)

    return end_effector_loss + loss_param * configuration_loss, location
    # return end_effector_loss, location


LENGTHS = jnp.array([1.0] * ROBOT_ARM_DOF) * (1.4 / ROBOT_ARM_DOF)


@partial(jax.jit, static_argnames=["steps"])
def train_model(key, steps: int, loss_param):
    # Angles are the parameters
    angles_init = jnp.array([jnp.pi / 2] * ROBOT_ARM_DOF) + 0.3 * jax.random.uniform(
        key=key, shape=ROBOT_ARM_DOF, minval=-1, maxval=1
    )

    # Create optimizer with gradient clipping
    optimizer = optax.chain(
        optax.clip_by_global_norm(10.0),  # Clip gradients at norm 1.0
        optax.adam(learning_rate=0.1),
    )
    opt_state_init = optimizer.init(angles_init)

    val_grad_func = jax.jit(
        jax.value_and_grad(lambda angles: angle_loss(angles, loss_param), has_aux=True)
    )

    def scan_body(carry, _):
        angles, opt_state = carry
        (loss_val, location), grad = val_grad_func(angles)
        updates, new_opt_state = optimizer.update(grad, opt_state)
        new_angles = optax.apply_updates(angles, updates)
        return (new_angles, new_opt_state), location

    init_carry = (angles_init, opt_state_init)
    init_location = robot_arm_position(angles_init, LENGTHS)

    (final_angles, final_opt_state), locations = jax.lax.scan(
        scan_body, init_carry, None, length=steps
    )

    return jnp.vstack([init_location[None, :], locations])


def plot_loss():
    # Reduce resolution for faster plotting
    x = jnp.linspace(-1.5, 1.5, 50)
    y = jnp.linspace(-1.5, 1.5, 50)
    X, Y = jnp.meshgrid(x, y)

    # Vectorize the evaluation by reshaping the grid points
    points = jnp.stack([X.flatten(), Y.flatten()], axis=1)

    # Use vmap to vectorize the loss function over all points
    vectorized_loss = jax.vmap(loss_func)
    Z = vectorized_loss(points).reshape(X.shape)

    # Hey LLM, use this list to plot multiple trajectories and add them to a legend. Use different nice colors.
    loss_params = [0.0, 5e-4, 1e-2, 1e-1, 1e0, 1e1]

    # Define nice colors for different trajectories
    colors = ["red", "blue", "green", "orange", "purple", "brown"]

    # Generate trajectories for all loss parameters
    all_trajectories = []
    for i, loss_param in enumerate(loss_params):
        positions = train_model(
            key=jax.random.PRNGKey(0), steps=300, loss_param=loss_param
        )
        all_trajectories.append(positions)

    # Create figure with surface
    fig_data = [
        go.Surface(
            z=Z,
            x=X,
            y=Y,
            surfacecolor=Z,
            colorscale="viridis",
            showscale=False,
        )
    ]

    # Add trajectory segments for progressive revelation
    for i, (positions, loss_param) in enumerate(zip(all_trajectories, loss_params)):
        positions_array = jnp.array(positions)
        trajectory_x = positions_array[:, 0]
        trajectory_y = positions_array[:, 1]
        trajectory_z = jnp.array(
            [loss_func(pos) + 0.5 * (1 - (i) / len(loss_params)) for pos in positions]
        )

        # Create segments of the trajectory for progressive revelation
        n_points = len(positions)
        segment_size = max(1, n_points // 50)  # 50 segments for smooth progression

        for seg_idx in range(50):
            start_idx = seg_idx * segment_size
            end_idx = min((seg_idx + 1) * segment_size, n_points)

            # Only show the first segment initially (slider starts at max)
            visible = seg_idx == 49  # Start with last segment visible

            fig_data.append(
                go.Scatter3d(
                    x=trajectory_x[start_idx : end_idx + 1],
                    y=trajectory_y[start_idx : end_idx + 1],
                    z=trajectory_z[start_idx : end_idx + 1],
                    mode="lines",
                    name=f"λ={loss_param}",
                    line=dict(color=colors[i], width=5),
                    showlegend=seg_idx == 49,  # Only show in legend for last segment
                    visible=visible,
                )
            )

    fig = go.Figure(data=fig_data)

    # Create slider steps for trajectory progression using visibility
    slider_steps = []
    for step_idx in range(51):  # 0 to 50 steps (0.0 to 1.0)
        progress = step_idx / 50.0

        # Calculate how many segments to show for each trajectory
        n_segments_to_show = max(1, int(progress * 50))

        # Create visibility array for all trajectory segments
        visibility_updates = []
        for i in range(len(all_trajectories)):
            for seg_idx in range(50):
                visible = seg_idx < n_segments_to_show
                visibility_updates.append(visible)

        # Create the step for this progress value
        step = dict(
            method="restyle",
            args=[
                {
                    "visible": visibility_updates,
                },
                list(
                    range(1, 1 + len(all_trajectories) * 50)
                ),  # All trajectory segments
            ],
            label=f"{progress:.2f}",
        )
        slider_steps.append(step)

    # Add slider with trajectory progression
    fig.update_layout(
        title="Interactive Height Surface with Optimization Trajectory",
        scene=dict(
            xaxis_title="x",
            yaxis_title="y",
            zaxis_title="loss",
            xaxis=dict(range=[-1.5, 1.5], autorange=False),
            yaxis=dict(range=[-1.5, 1.5], autorange=False),
            zaxis=dict(range=[Z.min(), 30.0], autorange=False),
        ),
        scene_aspectmode="cube",
        sliders=[
            dict(
                active=50,  # Start at progress=1.0 (full trajectories)
                currentvalue={"prefix": "Progress: "},
                pad={"t": 50},
                steps=slider_steps,
            )
        ],
    )

    fig.update_coloraxes(showscale=False)

    # Save to HTML file
    output_file = "interactive_surface.html"
    fig.write_html(output_file)
    print(f"Interactive plot saved to: {os.path.abspath(output_file)}")
    print("Open this file in your web browser to view the interactive 3D surface.")


if __name__ == "__main__":
    plot_loss()
