from __future__ import annotations

from typing import Callable, Optional

import jax
import jax.numpy as jnp
import equinox as eqx
import optax

from pst_dtlgn.training.regularizers import ternary_commitment_loss
from pst_dtlgn.training.schedulers import lambda_schedule


def train_step_smoothed(
    model,
    opt_state: optax.OptState,
    optimizer: optax.GradientTransformation,
    batch_x: jax.Array,
    batch_y: jax.Array,
    lambda_reg: jax.Array,
    step: jax.Array,
    total_steps: int,
    key: jax.Array,
    loss_fn: Optional[Callable] = None,
) -> tuple:
    if loss_fn is None:
        loss_fn = lambda pred, tgt: jnp.mean((pred - tgt) ** 2)

    def total_loss_fn(net):
        # Per-sample keys for stochastic smoothing (S3 Gaussian).
        # For deterministic strategies (S1 STE) the key is unused
        # inside the network but splitting it is harmless.
        batch_size = batch_x.shape[0]
        keys = jax.random.split(key, batch_size)

        preds = jax.vmap(
            lambda x, k: net(x, step, total_steps, k)
        )(batch_x, keys)  # (batch, output_dim)

        task_loss = loss_fn(preds, batch_y)
        reg_loss = ternary_commitment_loss(net)
        total = task_loss + lambda_reg * reg_loss
        return total, {"task_loss": task_loss, "reg_loss": reg_loss}

    (loss, metrics), grads = eqx.filter_value_and_grad(
        total_loss_fn, has_aux=True
    )(model)

    updates, new_opt_state = optimizer.update(
        grads, opt_state, eqx.filter(model, eqx.is_inexact_array)
    )
    model = eqx.apply_updates(model, updates)

    metrics["total_loss"] = loss
    return model, new_opt_state, metrics


def train_smoothed(
    model,
    optimizer: optax.GradientTransformation,
    train_x: jax.Array,
    train_y: jax.Array,
    total_steps: int,
    lambda_max: float = 1.0,
    lambda_gamma: float = 2.0,
    batch_size: Optional[int] = None,
    loss_fn: Optional[Callable] = None,
    key: Optional[jax.Array] = None,
    log_every: int = 100,
    callback: Optional[Callable] = None,
) -> tuple:
    if key is None:
        key = jax.random.PRNGKey(0)

    N = train_x.shape[0]
    if batch_size is None:
        batch_size = N

    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    history = []

    # JIT the train step.
    # total_steps is static (captured by closure); step and key are traced.
    @eqx.filter_jit
    def jit_step(m, ost, bx, by, lam, step_val, step_key):
        return train_step_smoothed(
            m, ost, optimizer, bx, by, lam, step_val, total_steps,
            step_key, loss_fn,
        )

    for step in range(total_steps):
        lam = lambda_schedule(step, total_steps, lambda_max, lambda_gamma)

        # 3-way split: batch sampling + step noise + continuation.
        key, batch_key, step_key = jax.random.split(key, 3)

        # Sample batch
        if batch_size < N:
            idx = jax.random.randint(batch_key, (batch_size,), 0, N)
            bx = train_x[idx]
            by = train_y[idx]
        else:
            bx = train_x
            by = train_y

        # HOTFIX: wrap lam and step as JAX scalars to prevent per-value
        # XLA recompilation (same pattern as trainer.py line 134).
        model, opt_state, metrics = jit_step(
            model, opt_state, bx, by,
            jnp.float32(lam), jnp.float32(step), step_key,
        )

        if (step % log_every == 0) or (step == total_steps - 1):
            metrics_log = {
                "step": step,
                "lambda": lam,
                **{k: float(v) for k, v in metrics.items()},
            }
            history.append(metrics_log)
            if callback is not None:
                callback(step, model, metrics_log)

    return model, history
