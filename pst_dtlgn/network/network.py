from __future__ import annotations
from typing import Optional
import jax
import jax.numpy as jnp
import equinox as eqx
from pst_dtlgn.network.layer import PSTLayer
from pst_dtlgn.network.topology import Topology


class PolynomialNetwork(eqx.Module):
    layers: list[PSTLayer]
    input_dim: int = eqx.field(static=True)
    output_dim: int = eqx.field(static=True)
    depth: int = eqx.field(static=True)

    def __init__(
        self,
        key: jax.Array,
        topology: Topology,
        sigma: float = 0.45,
    ) -> None:
        self.input_dim = topology.input_dim
        self.output_dim = topology.layer_widths[-1]
        self.depth = len(topology.layer_widths)
        layers = []
        for l in range(self.depth):
            key, subkey = jax.random.split(key)
            layer = PSTLayer(key=subkey, connections=topology.connections[l], sigma=sigma)
            layers.append(layer)
        self.layers = layers

    def __call__(self, x: jax.Array) -> jax.Array:
        h = x
        for layer in self.layers:
            h = layer(h)
        return h

    @property
    def total_neurons(self) -> int:
        return sum(layer.width for layer in self.layers)

    @property
    def layer_widths(self) -> list[int]:
        return [layer.width for layer in self.layers]


def init_network(
    key: jax.Array,
    topology: Topology,
    sigma: float = 0.45,
) -> PolynomialNetwork:
    return PolynomialNetwork(key=key, topology=topology, sigma=sigma)
