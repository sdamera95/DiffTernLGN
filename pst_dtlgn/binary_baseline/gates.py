from __future__ import annotations
import jax
import jax.numpy as jnp
import numpy as np

NUMBER_OF_GATES = 16

GATE_NAMES_BIN = [
    "FALSE", "AND", "A AND ~B", "A",
    "~A AND B", "B", "XOR", "OR",
    "NOR", "XNOR", "~B", "A OR ~B",
    "~A", "~A OR B", "NAND", "TRUE",
]

PASS_THROUGH_A = 3
PASS_THROUGH_B = 5
DEFAULT_PASS_VALUE = 10.0

BIN_TRUTH_TABLES = np.array([
    [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 1, 1],
    [0, 1, 0, 0], [0, 1, 0, 1], [0, 1, 1, 0], [0, 1, 1, 1],
    [1, 0, 0, 0], [1, 0, 0, 1], [1, 0, 1, 0], [1, 0, 1, 1],
    [1, 1, 0, 0], [1, 1, 0, 1], [1, 1, 1, 0], [1, 1, 1, 1],
], dtype=np.int8)


def bin_op_all(a: jax.Array, b: jax.Array) -> jax.Array:
    return jnp.stack([
        jnp.zeros_like(a),          # 0: FALSE
        a * b,                       # 1: AND
        a - a * b,                   # 2: A AND ~B
        a + jnp.zeros_like(b),      # 3: A
        b - a * b,                   # 4: ~A AND B
        b + jnp.zeros_like(a),      # 5: B
        a + b - 2 * a * b,          # 6: XOR
        a + b - a * b,              # 7: OR
        1 - (a + b - a * b),        # 8: NOR
        1 - (a + b - 2 * a * b),    # 9: XNOR
        1 - b + jnp.zeros_like(a),  # 10: ~B
        1 - b + a * b,              # 11: A OR ~B
        1 - a + jnp.zeros_like(b),  # 12: ~A
        1 - a + a * b,              # 13: ~A OR B
        1 - a * b,                  # 14: NAND
        jnp.ones_like(a),           # 15: TRUE
    ], axis=-1)


def bin_op_soft(a: jax.Array, b: jax.Array, logits: jax.Array) -> jax.Array:
    return jnp.sum(
        bin_op_all(a, b) * jax.nn.softmax(logits, axis=-1), axis=-1,
    )


def bin_op_hard(a: jax.Array, b: jax.Array, logits: jax.Array) -> jax.Array:
    return jnp.sum(
        bin_op_all(a, b) * jax.nn.one_hot(
            jnp.argmax(logits, axis=-1), 16,
        ), axis=-1,
    )
