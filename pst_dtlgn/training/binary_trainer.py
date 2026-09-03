from __future__ import annotations
from typing import Callable, Optional
import jax
import jax.numpy as jnp
import equinox as eqx
import optax


def binary_cross_entropy_loss(logits: jax.Array, targets: jax.Array) -> jax.Array:
    return jnp.mean(
        optax.softmax_cross_entropy_with_integer_labels(logits, targets)
    )


def train_step_binary(
    model,
    opt_state: optax.OptState,
    optimizer: optax.GradientTransformation,
    batch_x: jax.Array,
    batch_y: jax.Array,
    loss_fn: Optional[Callable] = None,
    group_sum=None,
) -> tuple:
    if loss_fn is None:
        if group_sum is not None:
            loss_fn = binary_cross_entropy_loss
        else:
            loss_fn = lambda pred, tgt: jnp.mean((pred - tgt) ** 2)

    def total_loss_fn(net):
        preds = jax.vmap(net)(batch_x)
        if group_sum is not None:
            preds = jax.vmap(group_sum)(preds)
        task_loss = loss_fn(preds, batch_y)
        return task_loss, {"task_loss": task_loss}

    (loss, metrics), grads = eqx.filter_value_and_grad(
        total_loss_fn, has_aux=True
    )(model)
    updates, new_opt_state = optimizer.update(
        grads, opt_state, eqx.filter(model, eqx.is_inexact_array)
    )
    model = eqx.apply_updates(model, updates)
    metrics["total_loss"] = loss
    return model, new_opt_state, metrics


def train_binary(
    model,
    optimizer: optax.GradientTransformation,
    train_x: jax.Array,
    train_y: jax.Array,
    total_steps: int,
    batch_size: Optional[int] = None,
    loss_fn: Optional[Callable] = None,
    group_sum=None,
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

    @eqx.filter_jit
    def jit_step(m, ost, bx, by):
        return train_step_binary(m, ost, optimizer, bx, by, loss_fn, group_sum)

    for step in range(total_steps):
        if batch_size < N:
            key, subkey = jax.random.split(key)
            idx = jax.random.randint(subkey, (batch_size,), 0, N)
            bx = train_x[idx]
            by = train_y[idx]
        else:
            bx = train_x
            by = train_y
        model, opt_state, metrics = jit_step(model, opt_state, bx, by)
        if (step % log_every == 0) or (step == total_steps - 1):
            metrics_log = {
                "step": step,
                **{k: float(v) for k, v in metrics.items()},
            }
            history.append(metrics_log)
            if callback is not None:
                callback(step, model, metrics_log)
    return model, history
