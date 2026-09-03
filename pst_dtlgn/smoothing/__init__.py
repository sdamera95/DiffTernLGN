from __future__ import annotations

import jax
import jax.numpy as jnp
import equinox as eqx

from pst_dtlgn.core.constants import V, V_INV
from pst_dtlgn.network.layer import PSTLayer
from pst_dtlgn.network.topology import Topology
from pst_dtlgn.smoothing.base import SmoothingStrategy, IdentitySmoothing
from pst_dtlgn.smoothing.compose import ComposedSmoothing
from pst_dtlgn.smoothing.schedules import (
    ste_strength_schedule,
    gumbel_temperature_schedule,
    gaussian_sigma_schedule,
    moreau_gamma_schedule,
)


# Precompute transposed matrices (used in every forward pass).
_VT = V.T        # (9, 9): coefficients -> truth table (batched)
_V_INV_T = V_INV.T  # (9, 9): truth table -> coefficients (batched)


def smoothed_forward(
    weights: jax.Array,
    connections: jax.Array,
    prev_outputs: jax.Array,
    strategy: SmoothingStrategy,
    step: jax.Array,
    total_steps: int,
    key: jax.Array | None,
) -> jax.Array:
    # 1. Gather inputs
    a = prev_outputs[connections[:, 0]]  # (n,)
    b = prev_outputs[connections[:, 1]]  # (n,)

    # 2-4. Vandermonde smoothing pipeline
    tt_soft = weights @ _VT                 # (n, 9)
    tt_smoothed = strategy.smooth_truth_table(
        tt_soft, step, total_steps, key
    )
    w_smoothed = tt_smoothed @ _V_INV_T     # (n, 9)

    # 5. Polynomial evaluation with smoothed coefficients
    a2 = a * a
    b2 = b * b
    monomials = jnp.stack([
        jnp.ones_like(a),   # 1
        a,                   # x
        b,                   # y
        a * b,               # xy
        a2,                  # x^2
        b2,                  # y^2
        a2 * b,              # x^2 y
        a * b2,              # x y^2
        a2 * b2,             # x^2 y^2
    ], axis=-1)  # (n, 9)

    raw = jnp.sum(w_smoothed * monomials, axis=-1)  # (n,)

    # 6. Clip
    return jnp.clip(raw, -1.0, 1.0)


class SmoothedPSTLayer(eqx.Module):
    layer: PSTLayer
    strategy: SmoothingStrategy

    @property
    def weights(self) -> jax.Array:
        return self.layer.weights

    @property
    def connections(self) -> jax.Array:
        return self.layer.connections

    @property
    def width(self) -> int:
        return self.layer.width

    def __call__(
        self,
        prev_outputs: jax.Array,
        step: jax.Array,
        total_steps: int,
        key: jax.Array | None = None,
    ) -> jax.Array:
        return smoothed_forward(
            self.layer.weights, self.layer.connections,
            prev_outputs, self.strategy, step, total_steps, key,
        )


class SmoothedPSTNetwork(eqx.Module):
    layers: list[PSTLayer]
    strategy: SmoothingStrategy
    input_dim: int = eqx.field(static=True)
    output_dim: int = eqx.field(static=True)
    depth: int = eqx.field(static=True)

    def __init__(
        self,
        key: jax.Array,
        topology: Topology,
        strategy: SmoothingStrategy,
        sigma: float = 0.45,
    ) -> None:
        self.input_dim = topology.input_dim
        self.output_dim = topology.layer_widths[-1]
        self.depth = len(topology.layer_widths)
        self.strategy = strategy

        layers = []
        for l in range(self.depth):
            key, subkey = jax.random.split(key)
            layer = PSTLayer(
                key=subkey,
                connections=topology.connections[l],
                sigma=sigma,
            )
            layers.append(layer)
        self.layers = layers

    def __call__(
        self,
        x: jax.Array,
        step: jax.Array = jnp.float32(0.0),
        total_steps: int = 1,
        key: jax.Array | None = None,
    ) -> jax.Array:
        h = x
        for layer in self.layers:
            if key is not None:
                key, subkey = jax.random.split(key)
            else:
                subkey = None
            h = smoothed_forward(
                layer.weights, layer.connections, h,
                self.strategy, step, total_steps, subkey,
            )
        return h

    def soft_forward(self, x: jax.Array) -> jax.Array:
        h = x
        for layer in self.layers:
            h = layer(h)
        return h

    @classmethod
    def wrap(
        cls,
        network: eqx.Module,
        strategy: SmoothingStrategy,
    ) -> SmoothedPSTNetwork:
        obj = object.__new__(cls)
        object.__setattr__(obj, 'layers', list(network.layers))
        object.__setattr__(obj, 'strategy', strategy)
        object.__setattr__(obj, 'input_dim', network.input_dim)
        object.__setattr__(obj, 'output_dim', network.output_dim)
        object.__setattr__(obj, 'depth', network.depth)
        return obj

    @property
    def total_neurons(self) -> int:
        return sum(layer.width for layer in self.layers)

    @property
    def layer_widths(self) -> list[int]:
        return [layer.width for layer in self.layers]
