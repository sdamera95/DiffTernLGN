from __future__ import annotations
import jax
import jax.numpy as jnp
import equinox as eqx
from pst_dtlgn.core.constants import V, V_INV


class PSTNeuron(eqx.Module):
    weights: jax.Array  # (9,)

    def __call__(self, a: jax.Array, b: jax.Array) -> jax.Array:
        return evaluate_polynomial(self.weights, a, b)


def evaluate_polynomial(weights: jax.Array, a: jax.Array, b: jax.Array) -> jax.Array:
    w = weights
    a2 = a * a
    b2 = b * b
    return (w[0] + w[1] * a + w[2] * b + w[3] * a * b
            + w[4] * a2 + w[5] * b2 + w[6] * a2 * b
            + w[7] * a * b2 + w[8] * a2 * b2)


def compute_truth_table(neuron: PSTNeuron) -> jax.Array:
    return V @ neuron.weights


def truth_table_from_weights(weights: jax.Array) -> jax.Array:
    return V @ weights


def from_truth_table(truth_table: jax.Array) -> PSTNeuron:
    weights = V_INV @ jnp.asarray(truth_table, dtype=jnp.float32)
    return PSTNeuron(weights=weights)


def init_neuron(key: jax.Array, sigma: float = 0.45) -> PSTNeuron:
    weights = jax.random.normal(key, shape=(9,)) * sigma
    return PSTNeuron(weights=weights)
