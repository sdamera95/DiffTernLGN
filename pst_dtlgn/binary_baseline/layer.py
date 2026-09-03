from __future__ import annotations
import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
from pst_dtlgn.binary_baseline.gates import (
    bin_op_soft, bin_op_hard, PASS_THROUGH_A, DEFAULT_PASS_VALUE,
)

_VALID_INITS = ("pass_through_a", "randn")


class BinaryLayer(eqx.Module):
    logits: jax.Array       # (n, 16) float32
    connections: jax.Array  # (n, 2) int32
    width: int = eqx.field(static=True)

    def __init__(
        self,
        connections: np.ndarray | jax.Array,
        *, init: str,
        key: jax.Array | None = None,
    ) -> None:
        if init not in _VALID_INITS:
            raise ValueError(f"init must be one of {_VALID_INITS}, got {init!r}")
        n = connections.shape[0]
        self.width = n
        self.connections = jnp.asarray(connections, dtype=jnp.int32)
        if init == "pass_through_a":
            self.logits = jnp.zeros((n, 16)).at[:, PASS_THROUGH_A].set(DEFAULT_PASS_VALUE)
        elif init == "randn":
            if key is None:
                raise ValueError("key is required when init='randn'")
            self.logits = jax.random.normal(key, (n, 16))

    def soft(self, prev_outputs: jax.Array) -> jax.Array:
        a = prev_outputs[self.connections[:, 0]]
        b = prev_outputs[self.connections[:, 1]]
        return bin_op_soft(a, b, self.logits)

    def hard(self, prev_outputs: jax.Array) -> jax.Array:
        a = prev_outputs[self.connections[:, 0]]
        b = prev_outputs[self.connections[:, 1]]
        return bin_op_hard(a, b, self.logits)
