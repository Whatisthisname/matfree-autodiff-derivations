import jax
import jax.numpy as jnp
import flax.linen as nn
import matfree.decomp
import optax
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from functools import partial

from learn_matfree import (
    compute_log_determinant_of_regularized_metric_tensor,
    tridiag_extract_inner_from_regularized,
)

jax.config.update("jax_enable_x64", True)


class Net(nn.Module):
    tiny: bool = False

    def setup(self):
        if self.tiny:
            # Tiny network with ~2000 params
            self.conv1 = nn.Conv(features=3, kernel_size=(3, 3))
            self.conv2 = nn.Conv(features=3, kernel_size=(3, 3))
            self.fc1 = nn.Dense(features=3)
            self.fc2 = nn.Dense(features=10)
        else:
            # Original larger network
            self.conv1 = nn.Conv(features=32, kernel_size=(3, 3))
            self.conv2 = nn.Conv(features=64, kernel_size=(3, 3))
            self.fc1 = nn.Dense(features=128)
            self.fc2 = nn.Dense(features=10)
        self.dropout = nn.Dropout(rate=0.5)

    def __call__(self, x, training=True, dropout_key=None):
        x = self.conv1(x)
        x = nn.relu(x)
        x = self.conv2(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))  # flatten
        x = self.fc1(x)
        x = nn.relu(x)
        x = self.dropout(x, deterministic=not training, rng=dropout_key)
        x = self.fc2(x)
        return x


@partial(jax.jit, static_argnums=(0, 1, 2))
def train_step(
    model: Net,
    tx: optax.GradientTransformationExtraArgs,
    regularization_strength: float,
    params: jax.Array,
    opt_state: jax.Array,
    batch_images: jax.Array,
    batch_labels: jax.Array,
    key: jax.Array,
):
    """JIT-compiled training step that combines forward pass, loss, gradients, and updates"""

    def loss_fn(params):
        logits = model.apply(params, batch_images, training=False, dropout_key=key)
        one_hot = jax.nn.one_hot(batch_labels, 10)
        loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))

        return loss, logits

    def extra_loss_fn(params):
        # result = jnp.array(
        #     [
        #         compute_log_determinant_of_regularized_metric_tensor(
        #             model=model,
        #             params=params,
        #             input_data=batch_images[i].reshape(1, *batch_images[0].shape),
        #             key=key,
        #             num_matvecs=5,
        #             alpha=0.5,
        #         )
        #         for i in range(batch_images.shape[0])
        #     ]
        # )
        # loss += jnp.mean(result) * regularization_strength

        result = jnp.array(
            [
                tridiag_extract_inner_from_regularized(
                    model=model,
                    params=params,
                    input_data=batch_images[i].reshape(1, *batch_images[0].shape),
                    key=key,
                    num_matvecs=5,
                    alpha=0.5,
                )
                for i in range(batch_images.shape[0])
            ]
        )
        return jnp.mean(result) * regularization_strength

        # param_grad, _, _, _ = jax.vjp(
        #     model.apply, params, batch_images, False, key
        # )[1](jnp.ones((batch_labels.shape[0], 10), dtype=jnp.float64))
        # loss += jnp.array(
        #     jax.tree.flatten(jax.tree.map(jnp.mean, param_grad))[0]
        # ).sum()

    # Compute loss and gradients
    (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)

    if regularization_strength > 0:
        regu_loss, regu_grads = jax.value_and_grad(extra_loss_fn)(params)

        # Combine the gradients: Project the regu_grad into the halfspace of grads
        def combine_gradients(loss_grad, regu_grad):
            _regu_grad_norm = jnp.linalg.norm(regu_grad)
            _loss_grad_norm = jnp.linalg.norm(loss_grad)

            normed_regu_grad = regu_grad / (_regu_grad_norm + 1e-10)
            normed_loss_grad = loss_grad / (_loss_grad_norm + 1e-10)

            inner_product = jnp.sum(normed_loss_grad * normed_regu_grad)
            # jax.debug.print("nner prod:{}", inner_product)

            return jax.lax.cond(
                pred=inner_product > 0,
                true_fun=lambda _: loss_grad + normed_regu_grad * _loss_grad_norm,
                false_fun=lambda _: loss_grad,
                operand=inner_product,
            )

        final_grads = jax.tree.map(combine_gradients, grads, regu_grads)
    else:
        final_grads = grads

    # Update parameters
    updates, opt_state = tx.update(final_grads, opt_state)
    params = optax.apply_updates(params, updates)

    return params, opt_state, loss


@partial(jax.jit, static_argnums=(0,))
def compute_accuracy(model, params, images, labels):
    """JIT-compiled accuracy computation"""
    logits = model.apply(params, images, training=False)
    predictions = jnp.argmax(logits, axis=-1)
    return jnp.mean(predictions == labels)


def main():
    key = jax.random.PRNGKey(0)

    REGULARIZATION_STRENGTH = 1.0

    print("Loading MNIST dataset...")
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    X, y = mnist.data, mnist.target

    # Convert and normalize
    X = jnp.float64(X) / 255.0
    y = jnp.int32(y.astype(int))

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Reshape images to (batch_size, height, width, channels)
    X_train = X_train.reshape(-1, 28, 28, 1)
    X_test = X_test.reshape(-1, 28, 28, 1)

    # Downsample images to target size using average pooling
    def downsample_batch(images, target_size=(4, 4)):
        # Compute pooling window size and strides to achieve target size
        input_h, input_w = images.shape[1:3]
        pool_size = (input_h // target_size[0], input_w // target_size[1])
        strides = pool_size  # Use same strides as pool size for non-overlapping windows

        return jax.lax.reduce_window(
            images,
            init_value=0.0,
            computation=jax.lax.add,
            window_dimensions=(1, *pool_size, 1),
            window_strides=(1, *strides, 1),
            padding="VALID",
        ) / (pool_size[0] * pool_size[1])

    X_train = downsample_batch(X_train)
    X_test = downsample_batch(X_test)

    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    print(f"Regularization strength: {REGULARIZATION_STRENGTH}")

    # Initialize model and optimizer
    model = Net(tiny=True)
    rngs = {"params": key, "dropout": key}
    params = model.init(rngs, X_train[0:1])

    # Use a more efficient optimizer configuration
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),  # Gradient clipping
        optax.adam(learning_rate=0.001),
    )
    opt_state = tx.init(params)

    num_epochs = 5
    batch_size = 8

    # Pre-compute number of batches
    num_batches = len(X_train) // batch_size

    # Training loop with fully JIT-compiled training step
    for epoch in range(num_epochs):
        epoch_losses = []

        # Shuffle data at the beginning of each epoch
        key, shuffle_key = jax.random.split(key)
        indices = jax.random.permutation(shuffle_key, len(X_train))
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]

        for i in range(0, len(X_train), batch_size):
            batch_images = X_train_shuffled[i : i + batch_size]
            batch_labels = y_train_shuffled[i : i + batch_size]

            key, subkey = jax.random.split(key)

            # Single JIT-compiled training step
            params, opt_state, loss = train_step(
                model,
                tx,
                REGULARIZATION_STRENGTH,
                params,
                opt_state,
                batch_images,
                batch_labels,
                subkey,
            )

            epoch_losses.append(loss)

            if i % (batch_size * 50) == 0:  # Reduced logging frequency
                print(
                    f"Epoch {epoch + 1}, Batch {i//batch_size}/{num_batches}, Loss: {loss:.4f}"
                )

        # Print epoch summary
        avg_loss = jnp.mean(jnp.array(epoch_losses))
        print(f"Epoch {epoch + 1} completed. Average loss: {avg_loss:.4f}")

    # Evaluation
    print("Computing test accuracy...")
    test_accuracy = compute_accuracy(model, params, X_test, y_test)
    print(f"Test accuracy: {test_accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()
