from __future__ import annotations


def lambda_schedule(
    step: int | float,
    total_steps: int | float,
    lambda_max: float,
    gamma: float = 2.0,
) -> float:
    if total_steps <= 0:
        return lambda_max
    progress = min(max(step / total_steps, 0.0), 1.0)
    return lambda_max * (progress ** gamma)


def sigma_schedule(
    step: int | float,
    total_steps: int | float,
    sigma_max: float = 0.3,
    sigma_min: float = 0.0,
) -> float:
    if total_steps <= 0:
        return sigma_min
    decay_end = 0.7 * total_steps
    if step >= decay_end:
        return sigma_min
    progress = step / decay_end
    return sigma_max * (1.0 - progress) + sigma_min * progress
