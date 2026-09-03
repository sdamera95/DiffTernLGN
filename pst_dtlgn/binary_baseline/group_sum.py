from __future__ import annotations
import jax
import jax.numpy as jnp
import equinox as eqx


class GroupSum(eqx.Module):
    k: int = eqx.field(static=True)
    tau: float = eqx.field(static=True)

    def __init__(self, k: int, tau: float = 10.0) -> None:
        self.k = k
        self.tau = tau

    def __call__(self, x: jax.Array) -> jax.Array:
        assert x.shape[-1] % self.k == 0, (
            f"Last dimension {x.shape[-1]} must be divisible by k={self.k}"
        )
        grouped = x.reshape(*x.shape[:-1], self.k, x.shape[-1] // self.k)
        return grouped.sum(-1) / self.tau
