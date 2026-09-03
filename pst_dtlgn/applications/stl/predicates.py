from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np


def predicate_soft(
    feature: jnp.ndarray,
    theta_low: float,
    theta_high: float,
) -> jnp.ndarray:
    theta_mid = (theta_high + theta_low) / 2.0
    half_width = (theta_high - theta_low) / 2.0
    return jnp.clip((feature - theta_mid) / half_width, -1.0, 1.0)


def predicate_hard(
    feature: jnp.ndarray,
    theta_low: float,
    theta_high: float,
) -> jnp.ndarray:
    return jnp.where(
        feature > theta_high,
        1,
        jnp.where(feature < theta_low, -1, 0),
    ).astype(jnp.int8)


@dataclass
class Predicate:
    name: str
    feature_fn: Callable[[jnp.ndarray], jnp.ndarray]
    theta_low: float
    theta_high: float
    description: str

    def __post_init__(self):
        if self.theta_low >= self.theta_high:
            raise ValueError(
                f"Predicate '{self.name}': theta_low ({self.theta_low}) "
                f"must be < theta_high ({self.theta_high})"
            )

    def soft(self, obs: jnp.ndarray) -> jnp.ndarray:
        return predicate_soft(
            self.feature_fn(obs), self.theta_low, self.theta_high
        )

    def hard(self, obs: jnp.ndarray) -> jnp.ndarray:
        return predicate_hard(
            self.feature_fn(obs), self.theta_low, self.theta_high
        )


class PredicateModule:
    def __init__(self, predicates: list[Predicate]) -> None:
        if not predicates:
            raise ValueError("PredicateModule requires at least one predicate")
        self.predicates = predicates
        self.num_predicates = len(predicates)

    def evaluate_soft(self, obs: jnp.ndarray) -> jnp.ndarray:
        return jnp.stack([p.soft(obs) for p in self.predicates])

    def evaluate_hard(self, obs: jnp.ndarray) -> jnp.ndarray:
        return jnp.stack([p.hard(obs) for p in self.predicates])

    def evaluate_soft_batch(self, obs_batch: jnp.ndarray) -> jnp.ndarray:
        return jax.vmap(self.evaluate_soft)(obs_batch)

    def evaluate_hard_batch(self, obs_batch: jnp.ndarray) -> jnp.ndarray:
        return jax.vmap(self.evaluate_hard)(obs_batch)

    @property
    def names(self) -> list[str]:
        return [p.name for p in self.predicates]


def calibrate_thresholds(
    feature_values: np.ndarray,
    low_pct: float = 30.0,
    high_pct: float = 70.0,
) -> tuple[float, float]:
    if low_pct >= high_pct:
        raise ValueError(
            f"low_pct ({low_pct}) must be < high_pct ({high_pct})"
        )

    vals = np.asarray(feature_values).ravel()
    if len(vals) == 0:
        raise ValueError("feature_values is empty")

    theta_low = float(np.percentile(vals, low_pct))
    theta_high = float(np.percentile(vals, high_pct))

    if theta_low >= theta_high:
        raise ValueError(
            f"Calibrated thresholds are degenerate: "
            f"theta_low={theta_low}, theta_high={theta_high}. "
            f"Feature distribution may have insufficient variance."
        )

    return theta_low, theta_high


def report_label_balance(
    predicate_values: np.ndarray,
    predicate_names: list[str] | None = None,
) -> dict:
    vals = np.asarray(predicate_values)
    if vals.ndim == 1:
        vals = vals[:, None]

    flat = vals.reshape(-1, vals.shape[-1])
    P = flat.shape[-1]

    if predicate_names is None:
        predicate_names = [f"pred_{i}" for i in range(P)]

    fractions = np.zeros((P, 3), dtype=np.float64)
    warnings = []
    n = flat.shape[0]

    for i in range(P):
        col = flat[:, i]
        frac_neg = float(np.mean(col == -1))
        frac_zero = float(np.mean(col == 0))
        frac_pos = float(np.mean(col == 1))
        fractions[i] = [frac_neg, frac_zero, frac_pos]

        for label, frac, name in [
            (-1, frac_neg, "FALSE"),
            (0, frac_zero, "UNKNOWN"),
            (1, frac_pos, "TRUE"),
        ]:
            if frac < 0.10:
                warnings.append(
                    f"{predicate_names[i]}: {name} has only "
                    f"{frac:.1%} representation (< 10%)"
                )

    return {
        "fractions": fractions,
        "predicate_names": predicate_names,
        "num_samples": n,
        "warnings": warnings,
    }
