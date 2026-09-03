import jax.numpy as jnp

TRUE = 1
FALSE = -1
UNKNOWN = 0
TERNARY_VALUES = jnp.array([-1, 0, 1])

# Row ordering: (-1,-1), (-1,0), (-1,1), (0,-1), ..., (1,1)
EVAL_POINTS = jnp.array([
    [-1, -1], [-1,  0], [-1,  1],
    [ 0, -1], [ 0,  0], [ 0,  1],
    [ 1, -1], [ 1,  0], [ 1,  1],
], dtype=jnp.float32)

NUM_EVAL_POINTS = 9
NUM_TERNARY_GATES = 3**9  # 19683

MONOMIAL_LABELS = ["1", "x", "y", "xy", "x\u00b2", "y\u00b2", "x\u00b2y", "xy\u00b2", "x\u00b2y\u00b2"]

# Vandermonde matrix: maps monomial coefficients to truth table values
V = jnp.array([
    [ 1, -1, -1,  1,  1,  1, -1, -1,  1],
    [ 1, -1,  0,  0,  1,  0,  0,  0,  0],
    [ 1, -1,  1, -1,  1,  1,  1, -1,  1],
    [ 1,  0, -1,  0,  0,  1,  0,  0,  0],
    [ 1,  0,  0,  0,  0,  0,  0,  0,  0],
    [ 1,  0,  1,  0,  0,  1,  0,  0,  0],
    [ 1,  1, -1, -1,  1,  1, -1,  1,  1],
    [ 1,  1,  0,  0,  1,  0,  0,  0,  0],
    [ 1,  1,  1,  1,  1,  1,  1,  1,  1],
], dtype=jnp.float32)

# Exact inverse: w = V_INV @ truth_table
V_INV = jnp.array([
    [ 0.0,   0.0,   0.0,   0.0,   1.0,   0.0,   0.0,   0.0,   0.0],
    [ 0.0,  -0.5,   0.0,   0.0,   0.0,   0.0,   0.0,   0.5,   0.0],
    [ 0.0,   0.0,   0.0,  -0.5,   0.0,   0.5,   0.0,   0.0,   0.0],
    [ 0.25,  0.0,  -0.25,  0.0,   0.0,   0.0,  -0.25,  0.0,   0.25],
    [ 0.0,   0.5,   0.0,   0.0,  -1.0,   0.0,   0.0,   0.5,   0.0],
    [ 0.0,   0.0,   0.0,   0.5,  -1.0,   0.5,   0.0,   0.0,   0.0],
    [-0.25,  0.0,   0.25,  0.5,   0.0,  -0.5,  -0.25,  0.0,   0.25],
    [-0.25,  0.5,  -0.25,  0.0,   0.0,   0.0,   0.25, -0.5,   0.25],
    [ 0.25, -0.5,   0.25, -0.5,   1.0,  -0.5,   0.25, -0.5,   0.25],
], dtype=jnp.float32)

V_INV_INT = jnp.array([
    [ 0,  0,  0,  0,  4,  0,  0,  0,  0],
    [ 0, -2,  0,  0,  0,  0,  0,  2,  0],
    [ 0,  0,  0, -2,  0,  2,  0,  0,  0],
    [ 1,  0, -1,  0,  0,  0, -1,  0,  1],
    [ 0,  2,  0,  0, -4,  0,  0,  2,  0],
    [ 0,  0,  0,  2, -4,  2,  0,  0,  0],
    [-1,  0,  1,  2,  0, -2, -1,  0,  1],
    [-1,  2, -1,  0,  0,  0,  1, -2,  1],
    [ 1, -2,  1, -2,  4, -2,  1, -2,  1],
], dtype=jnp.int32)

# Self-test: V @ V_INV == I
_identity_check = V @ V_INV
_expected_identity = jnp.eye(9, dtype=jnp.float32)
_max_error = jnp.max(jnp.abs(_identity_check - _expected_identity))
assert float(_max_error) < 1e-6, f"Vandermonde self-test failed: {float(_max_error):.2e}"

_int_check = jnp.max(jnp.abs(V_INV_INT - 4 * V_INV))
assert float(_int_check) < 1e-6, f"V_INV_INT != 4 * V_INV: {float(_int_check):.2e}"
