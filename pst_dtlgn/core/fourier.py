from __future__ import annotations
import jax.numpy as jnp

# Univariate basis at {-1, 0, 1}
PHI_VALUES = jnp.array([
    [1.0,    1.0,    1.0],
    [-1.0,   0.0,    1.0],
    [1/3,   -2/3,    1/3],
], dtype=jnp.float32)


def _build_bivariate_basis() -> jnp.ndarray:
    basis = jnp.zeros((9, 9), dtype=jnp.float32)
    for i in range(3):
        for j in range(3):
            k = i * 3 + j
            outer = jnp.outer(PHI_VALUES[i], PHI_VALUES[j])
            basis = basis.at[k].set(outer.ravel())
    return basis


BIVARIATE_BASIS = _build_bivariate_basis()
NORMS_SQUARED = jnp.sum(BIVARIATE_BASIS ** 2, axis=1) / 9.0

FOURIER_LABELS = [
    "c_00", "c_01", "c_02",
    "c_10", "c_11", "c_12",
    "c_20", "c_21", "c_22",
]


def fourier_coefficients(truth_table: jnp.ndarray) -> jnp.ndarray:
    tt = jnp.asarray(truth_table, dtype=jnp.float32)
    inner_products = BIVARIATE_BASIS @ tt / 9.0
    return inner_products / NORMS_SQUARED


def fourier_reconstruct(coefficients: jnp.ndarray) -> jnp.ndarray:
    c = jnp.asarray(coefficients, dtype=jnp.float32)
    return c @ BIVARIATE_BASIS  # (9,)


def fourier_sparsity(truth_table: jnp.ndarray, threshold: float = 1e-6) -> int:
    coeffs = fourier_coefficients(truth_table)
    return int(jnp.sum(jnp.abs(coeffs) > threshold))


def fourier_l1_norm(truth_table: jnp.ndarray) -> float:
    coeffs = fourier_coefficients(truth_table)
    return float(jnp.sum(jnp.abs(coeffs)))


def fourier_energy_breakdown(truth_table: jnp.ndarray) -> dict[str, float]:
    coeffs = fourier_coefficients(truth_table)
    energies = coeffs ** 2 * NORMS_SQUARED
    total = float(jnp.sum(energies))
    if total < 1e-12:
        return {"constant": 0.0, "linear": 0.0, "quadratic": 0.0, "cubic": 0.0, "quartic": 0.0}
    return {
        "constant": float(energies[0]) / total,
        "linear": float(energies[1] + energies[3]) / total,
        "quadratic": float(energies[2] + energies[4] + energies[6]) / total,
        "cubic": float(energies[5] + energies[7]) / total,
        "quartic": float(energies[8]) / total,
    }
