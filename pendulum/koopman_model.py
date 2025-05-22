import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Callable


class MLP(eqx.Module):
    layers: list
    activation: Callable = eqx.static_field()

    def __init__(
        self,
        in_size,
        out_size,
        width_size=64,
        depth=2,
        key=None,
        activation=jax.nn.relu,
    ):
        if key is None:
            key = jax.random.PRNGKey(0)
        keys = jax.random.split(key, depth + 1)
        self.layers = [eqx.nn.Linear(in_size, width_size, key=keys[0])]
        for i in range(depth - 1):
            self.layers.append(eqx.nn.Linear(width_size, width_size, key=keys[i + 1]))
        self.layers.append(eqx.nn.Linear(width_size, out_size, key=keys[-1]))
        self.activation = activation

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        return self.layers[-1](x)


class KoopmanModel(eqx.Module):
    encoder: MLP
    decoder: MLP
    latent_dim: int

    def __init__(self, input_dim, latent_dim, key):
        enc_key, dec_key = jax.random.split(key)
        self.encoder = MLP(input_dim, latent_dim, key=enc_key)
        self.decoder = MLP(latent_dim, input_dim, key=dec_key)
        self.latent_dim = latent_dim

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)
