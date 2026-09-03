from __future__ import annotations
import math
from typing import NamedTuple
import jax
import jax.numpy as jnp
import numpy as np


class Topology(NamedTuple):
    connections: list[np.ndarray]
    input_dim: int
    layer_widths: list[int]


def random_sparse(
    key: jax.Array,
    input_dim: int,
    layer_widths: list[int],
) -> Topology:
    connections = []
    for l, width in enumerate(layer_widths):
        key, subkey = jax.random.split(key)
        prev_width = input_dim if l == 0 else layer_widths[l - 1]
        conn = jax.random.randint(
            subkey, shape=(width, 2), minval=0, maxval=prev_width
        )
        connections.append(np.array(conn))
    return Topology(connections=connections, input_dim=input_dim,
                    layer_widths=layer_widths)


def butterfly(n: int) -> Topology:
    if n < 2 or (n & (n - 1)) != 0:
        raise ValueError(f"n must be a power of 2, got {n}")
    depth = int(math.log2(n))
    layer_widths = [n] * depth
    connections = []
    for l in range(depth):
        conn = np.zeros((n, 2), dtype=np.int64)
        stride = 1 << l
        for i in range(n):
            conn[i, 0] = i
            conn[i, 1] = i ^ stride
        connections.append(conn)
    return Topology(connections=connections, input_dim=n,
                    layer_widths=layer_widths)


def dilated_temporal(width: int, depth: int) -> Topology:
    layer_widths = [width] * depth
    connections = []
    for l in range(depth):
        conn = np.zeros((width, 2), dtype=np.int64)
        dilation = 1 << l
        for i in range(width):
            conn[i, 0] = i
            conn[i, 1] = (i + dilation) % width
        connections.append(conn)
    return Topology(connections=connections, input_dim=width,
                    layer_widths=layer_widths)


def validate_topology(topology: Topology) -> bool:
    for l, conn in enumerate(topology.connections):
        prev_width = (topology.input_dim if l == 0
                      else topology.layer_widths[l - 1])
        width = topology.layer_widths[l]
        if conn.shape != (width, 2):
            raise ValueError(
                f"Layer {l}: expected connections shape ({width}, 2), "
                f"got {conn.shape}"
            )
        if np.any(conn < 0) or np.any(conn >= prev_width):
            bad_idx = np.argwhere((conn < 0) | (conn >= prev_width))
            raise ValueError(
                f"Layer {l}: connection indices out of range [0, {prev_width}): "
                f"bad positions {bad_idx.tolist()}"
            )
    return True
