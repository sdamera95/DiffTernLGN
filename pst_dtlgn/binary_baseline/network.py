from __future__ import annotations
import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
from pst_dtlgn.binary_baseline.layer import BinaryLayer
from pst_dtlgn.binary_baseline.gates import GATE_NAMES_BIN
from pst_dtlgn.network.topology import Topology


class BinaryDLGN(eqx.Module):
    layers: list[BinaryLayer]
    input_dim: int = eqx.field(static=True)
    output_dim: int = eqx.field(static=True)
    depth: int = eqx.field(static=True)

    def __init__(self, key: jax.Array, topology: Topology, *, init: str) -> None:
        self.input_dim = topology.input_dim
        self.output_dim = topology.layer_widths[-1]
        self.depth = len(topology.layer_widths)
        layers = []
        for l_idx in range(self.depth):
            key, subkey = jax.random.split(key)
            layers.append(BinaryLayer(
                topology.connections[l_idx], init=init, key=subkey,
            ))
        self.layers = layers

    def __call__(self, x: jax.Array) -> jax.Array:
        h = x
        for layer in self.layers:
            h = layer.soft(h)
        return h

    def hard_forward(self, x: jax.Array) -> jax.Array:
        h = x
        for layer in self.layers:
            h = layer.hard(h)
        return h

    def gate_census(self) -> dict[str, int]:
        all_g = jnp.concatenate([
            jnp.argmax(layer.logits, axis=-1) for layer in self.layers
        ])
        counts = np.bincount(np.array(all_g), minlength=16)
        return {n: int(c) for n, c in zip(GATE_NAMES_BIN, counts) if c > 0}

    def commitment_score(self) -> float:
        n_committed = 0
        n_total = 0
        for layer in self.layers:
            probs = jax.nn.softmax(layer.logits, axis=-1)
            max_probs = jnp.max(probs, axis=-1)
            n_committed += int(jnp.sum(max_probs > 0.99))
            n_total += layer.width
        return n_committed / max(n_total, 1)

    @property
    def total_neurons(self) -> int:
        return sum(layer.width for layer in self.layers)

    @property
    def layer_widths(self) -> list[int]:
        return [layer.width for layer in self.layers]
