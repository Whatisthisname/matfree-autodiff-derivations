import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
import optax
import equinox as eqx
import koopman_model
from tests import test_bidiag_JVP_and_VJP_jax as bidiag_module
from arnoldi_bidiag import arnoldi_bidiag
# from matfree.decomp import bidiag as matfree_bidiag


def bidiagonalize(Phi: ArrayLike, bidiag_func, key):
    v0 = jax.random.normal(key=key, shape=Phi[0, :].shape)
    # v0 = Phi[0, :]
    # bidiag_output: bidiag_module.BidiagOutput = bidiag_func(v0, Phi)
    # B = jnp.diag(bidiag_output.alphas) + jnp.diag(bidiag_output.betas, 1)
    # L = bidiag_output.ls
    # R_T = bidiag_output.rs.T

    output: arnoldi_bidiag.DecompResult = bidiag_func(v0, Phi)
    B = jnp.diag(output.as_) + jnp.diag(output.bs_, 1)
    L = output.ls
    R_T = output.rs.T

    # def matvec(v, mat):
    #     return mat @ v
    # matves = 100
    # result = matfree_bidiag(matves, materialize=True)(matvec, start_vec, Phi)
    # mL, mR = result.Q_tall
    # mB = result.J_small

    # Phi = jax.lax.stop_gradient(Phi)
    # L, B, R_T = jnp.linalg.svd(Phi, full_matrices=False)
    # B = jnp.diag(B)

    return L, B, R_T


# Loss function
def compute_loss(model, trajec, bidiag_func, key):
    # Encode trajectory to latent space
    Traj1 = trajec[:-1, :]
    Traj2 = trajec[1:, :]
    all_encoded = jax.vmap(model.encode)(trajec)
    A = all_encoded[:-1, :]  # shape (T-1, k)
    B = all_encoded[1:, :]  # shape (T-1, k)

    koop = jnp.linalg.lstsq(A, B)[0].T

    # Compute bidiagonal decomposition
    L, Bi, R_T = bidiagonalize(koop, bidiag_func, key)
    # Minimize squared sum of Bi diagonal

    extra_loss = jnp.sum((jnp.abs(jnp.diag(Bi)) - 1) ** 2)

    # Bi_inv = jnp.linalg.inv(Bi)

    # # Compute Koopman matrix: A = Phi_next R B^{-1} L^T
    # A_koop = B @ R_T.T @ Bi_inv @ L.T  # (k, k)

    # Predict one step forward from current encodings
    # phi_t = jax.vmap(model.encode)(x_t)  # (T-1, k)
    # B_hat = A_koop @ A  # (k, T-1)
    B_hat = (koop @ A.T).T  # (T-1, k)

    # The koop should have determinant less than one

    # jax.debug.print("{}", jnp.mean(Phi2 - Phi2_hat) ** 2)

    # Compute MSE loss in physical space with reconstruction error
    dynamics_loss = jnp.mean((B - B_hat) ** 2)
    reconstruct_loss = jnp.mean((jax.vmap(model.decode)(B) - Traj2) ** 2)
    loss = dynamics_loss + reconstruct_loss + extra_loss
    return loss, (koop, loss)


# Training step
@eqx.filter_jit
def make_step(model, opt_state, optimizer, trajec, bidiag_func, key):
    key, subkey = jax.random.split(key)
    loss_fn = lambda model: compute_loss(model, trajec, bidiag_func, subkey)
    grads, (koopman_op, loss) = eqx.filter_grad(loss_fn, has_aux=True)(model)
    updates, opt_state = optimizer.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, koopman_op, loss, key


# Training loop
def train_koopman_model(
    trajec, latent_dim=64, steps=1000, lr=1e-3, seed=0, matvecs: int = 10
) -> tuple[koopman_model.KoopmanModel, ArrayLike]:
    """Returns trained model and koopman operator."""

    def matvec(v, mat):
        return mat @ v

    baked = arnoldi_bidiag.arnoldi_bidiagonalization(
        num_matvecs=matvecs,
        output_size=latent_dim,
        custom_vjp=True,
        reortho="full",
    )

    def bidiag_func(vec, mat):
        return baked(matvec, vec, mat)

    key = jax.random.PRNGKey(seed)
    model = koopman_model.KoopmanModel(
        input_dim=trajec.shape[1], latent_dim=latent_dim, key=key
    )
    key, _ = jax.random.split(key)  # Split key after model initialization
    optimizer = optax.chain(optax.adam(lr), optax.clip(jnp.array(0.01)))

    opt_state = optimizer.init(model)

    for step in range(steps):
        model, opt_state, koopman_op, loss, key = make_step(
            model=model,
            opt_state=opt_state,
            optimizer=optimizer,
            trajec=trajec,
            bidiag_func=bidiag_func,
            key=key,
        )
        if step % 100 == 0:
            print(f"Step {step} | Loss: {loss:.5f}")

    return model, koopman_op
