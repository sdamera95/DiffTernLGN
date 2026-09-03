from __future__ import annotations
import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np


class PSTLayer(eqx.Module):
    weights: jax.Array       # (n, 9) float32
    connections: jax.Array   # (n, 2) int32
    width: int = eqx.field(static=True)

    def __init__(
        self,
        key: jax.Array,
        connections: np.ndarray | jax.Array,
        sigma: float = 0.45,
    ) -> None:
        n = connections.shape[0]
        self.connections = jnp.asarray(connections, dtype=jnp.int32)
        self.width = n
        self.weights = jax.random.normal(key, shape=(n, 9)) * sigma

    def __call__(self, prev_outputs: jax.Array) -> jax.Array:
        a = prev_outputs[self.connections[:, 0]]
        b = prev_outputs[self.connections[:, 1]]
        a2 = a * a
        b2 = b * b
        monomials = jnp.stack([
            jnp.ones_like(a), a, b, a * b,
            a2, b2, a2 * b, a * b2, a2 * b2,
        ], axis=-1)
        raw = jnp.sum(self.weights * monomials, axis=-1)
        return jnp.clip(raw, -1.0, 1.0)

    @property
    def num_neurons(self) -> int:
        return self.width
