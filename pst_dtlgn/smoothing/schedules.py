from __future__ import annotations

import jax
import jax.numpy as jnp


def ste_strength_schedule(
    step: jax.Array,
    total_steps: int,
    activation_frac: float = 0.3,
) -> jax.Array:
    frac = step / jnp.maximum(total_steps - 1, 1)
    ramp_start = activation_frac
    ramp_end = 0.8
    q = jnp.clip((frac - ramp_start) / (ramp_end - ramp_start), 0.0, 1.0)
    return q


def gumbel_temperature_schedule(
    step: jax.Array,
    total_steps: int,
    tau_init: float = 1.0,
    tau_final: float = 0.05,
) -> jax.Array:
    frac = step / jnp.maximum(total_steps - 1, 1)
    tau = tau_final + 0.5 * (tau_init - tau_final) * (1.0 + jnp.cos(jnp.pi * frac))
    return tau


def gaussian_sigma_schedule(
    step: jax.Array,
    total_steps: int,
    sigma_max: float = 0.3,
    decay_gamma: float = 1.5,
) -> jax.Array:
    frac = step / jnp.maximum(total_steps - 1, 1)
    sigma = sigma_max * jnp.power(jnp.maximum(1.0 - frac, 0.0), decay_gamma)
    return sigma


def moreau_gamma_schedule(
    step: jax.Array,
    total_steps: int,
    gamma_init: float = 0.1,
    gamma_final: float = 100.0,
) -> jax.Array:
    frac = step / jnp.maximum(total_steps - 1, 1)
    log_ratio = jnp.log(gamma_final / gamma_init)
    gamma = gamma_init * jnp.exp(log_ratio * frac)
    return gamma
