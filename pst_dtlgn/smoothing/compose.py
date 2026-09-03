from __future__ import annotations

import jax
import jax.numpy as jnp
import equinox as eqx

from pst_dtlgn.smoothing.base import SmoothingStrategy


class ComposedSmoothing(SmoothingStrategy):
    strategies: list[SmoothingStrategy]

    def smooth_truth_table(
        self,
        tt_soft: jax.Array,
        step: jax.Array,
        total_steps: int,
        key: jax.Array | None,
    ) -> jax.Array:
        tt = tt_soft
        for strategy in self.strategies:
            if key is not None:
                key, subkey = jax.random.split(key)
            else:
                subkey = None
            tt = strategy.smooth_truth_table(tt, step, total_steps, subkey)
        return tt

    def schedule_params(self, step: jax.Array, total_steps: int) -> dict:
        params: dict = {}
        for i, strategy in enumerate(self.strategies):
            sub_params = strategy.schedule_params(step, total_steps)
            for k, v in sub_params.items():
                params[f"s{i}_{k}"] = v
        return params
