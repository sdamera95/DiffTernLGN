from __future__ import annotations

import jax
import jax.numpy as jnp
import equinox as eqx

from pst_dtlgn.smoothing.base import SmoothingStrategy
from pst_dtlgn.smoothing.schedules import gaussian_sigma_schedule


class GaussianSmoothing(SmoothingStrategy):
    sigma_max: float = eqx.field(static=True, default=0.3)
    decay_gamma: float = eqx.field(static=True, default=1.5)

    def smooth_truth_table(
        self,
        tt_soft: jax.Array,
        step: jax.Array,
        total_steps: int,
        key: jax.Array | None,
    ) -> jax.Array:
        sigma = gaussian_sigma_schedule(
            step, total_steps, self.sigma_max, self.decay_gamma,
        )
        noise = jax.random.normal(key, tt_soft.shape)
        return tt_soft + sigma * noise

    def schedule_params(self, step: jax.Array, total_steps: int) -> dict:
        sigma = gaussian_sigma_schedule(
            step, total_steps, self.sigma_max, self.decay_gamma,
        )
        return {"gaussian_sigma": sigma}
