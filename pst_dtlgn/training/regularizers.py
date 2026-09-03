from __future__ import annotations
import jax.numpy as jnp
from pst_dtlgn.core.constants import V


def dist_to_ternary_sq(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.minimum(jnp.minimum((x + 1) ** 2, x ** 2), (x - 1) ** 2)


def ternary_commitment_loss(model) -> jnp.ndarray:
    total_loss = jnp.float32(0.0)
    total_entries = 0
    for layer in model.layers:
        tt = layer.weights @ V.T
        loss = dist_to_ternary_sq(tt)
        total_loss = total_loss + jnp.sum(loss)
        total_entries += tt.shape[0] * 9
    return total_loss / total_entries


def fourier_sparsity_loss(model) -> jnp.ndarray:
    from pst_dtlgn.core.fourier import BIVARIATE_BASIS, NORMS_SQUARED
    total_l1 = jnp.float32(0.0)
    total_neurons = 0
    for layer in model.layers:
        tt = layer.weights @ V.T
        inner = tt @ BIVARIATE_BASIS.T / 9.0
        coeffs = inner / NORMS_SQUARED[None, :]
        l1 = jnp.sum(jnp.abs(coeffs), axis=-1)
        total_l1 = total_l1 + jnp.sum(l1)
        total_neurons += layer.width
    return total_l1 / total_neurons


def combined_regularizer(model, alpha: float = 1.0, beta: float = 0.0) -> jnp.ndarray:
    loss = alpha * ternary_commitment_loss(model)
    if beta > 0:
        loss = loss + beta * fourier_sparsity_loss(model)
    return loss
