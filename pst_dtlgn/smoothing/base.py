from __future__ import annotations

import abc

import jax
import equinox as eqx


class SmoothingStrategy(eqx.Module):
    @abc.abstractmethod
    def smooth_truth_table(
        self,
        tt_soft: jax.Array,
        step: jax.Array,
        total_steps: int,
        key: jax.Array | None,
    ) -> jax.Array:
        ...

    @abc.abstractmethod
    def schedule_params(self, step: jax.Array, total_steps: int) -> dict:
        ...


class IdentitySmoothing(SmoothingStrategy):
    def smooth_truth_table(
        self,
        tt_soft: jax.Array,
        step: jax.Array,
        total_steps: int,
        key: jax.Array | None = None,
    ) -> jax.Array:
        return tt_soft

    def schedule_params(self, step: jax.Array, total_steps: int) -> dict:
        return {}
