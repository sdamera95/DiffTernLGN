from __future__ import annotations

import jax
import jax.numpy as jnp
import equinox as eqx

from pst_dtlgn.smoothing.base import SmoothingStrategy
from pst_dtlgn.smoothing.schedules import ste_strength_schedule


def round_ternary_ste(tt: jax.Array) -> jax.Array:
    rounded = jnp.where(tt > 0.5, 1.0, jnp.where(tt < -0.5, -1.0, 0.0))
    return tt + jax.lax.stop_gradient(rounded - tt)


class STESmoothing(SmoothingStrategy):
    activation_frac: float = eqx.field(static=True, default=0.3)
    use_annealed_blend: bool = eqx.field(static=True, default=True)

    def smooth_truth_table(
        self,
        tt_soft: jax.Array,
        step: jax.Array,
        total_steps: int,
        key: jax.Array | None = None,
    ) -> jax.Array:
        q = ste_strength_schedule(step, total_steps, self.activation_frac)

        tt_hard = round_ternary_ste(tt_soft)

        if self.use_annealed_blend:
            # Continuous blend: gradients flow through both paths
            return (1.0 - q) * tt_soft + q * tt_hard
        else:
            # Hard switch: either fully soft or fully STE
            return jnp.where(q > 0.5, tt_hard, tt_soft)

    def schedule_params(self, step: jax.Array, total_steps: int) -> dict:
        q = ste_strength_schedule(step, total_steps, self.activation_frac)
        return {"ste_strength": q}
