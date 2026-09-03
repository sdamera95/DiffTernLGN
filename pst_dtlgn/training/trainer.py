from __future__ import annotations
from typing import Callable, Optional
import math
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from pst_dtlgn.training.regularizers import ternary_commitment_loss
from pst_dtlgn.training.schedulers import lambda_schedule, sigma_schedule


def train_step(
    model,
    opt_state: optax.OptState,
    optimizer: optax.GradientTransformation,
    batch_x: jnp.ndarray,
    batch_y: jnp.ndarray,
    lambda_reg: float,
    loss_fn: Optional[Callable] = None,
) -> tuple:
    if loss_fn is None:
        loss_fn = lambda pred, tgt: jnp.mean((pred - tgt) ** 2)

    def total_loss_fn(net):
        preds = jax.vmap(net)(batch_x)
        task_loss = loss_fn(preds, batch_y)
        reg_loss = ternary_commitment_loss(net)
        total = task_loss + lambda_reg * reg_loss
        return total, {"task_loss": task_loss, "reg_loss": reg_loss}

    (loss, metrics), grads = eqx.filter_value_and_grad(total_loss_fn, has_aux=True)(model)
    updates, new_opt_state = optimizer.update(
        grads, opt_state, eqx.filter(model, eqx.is_inexact_array)
    )
    model = eqx.apply_updates(model, updates)
    metrics["total_loss"] = loss
    return model, new_opt_state, metrics


def train(
    model,
    optimizer: optax.GradientTransformation,
    train_x: jnp.ndarray,
    train_y: jnp.ndarray,
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

    @eqx.filter_jit
    def jit_step(m, ost, bx, by, lam):
        return train_step(m, ost, optimizer, bx, by, lam, loss_fn)

    for step in range(total_steps):
        lam = lambda_schedule(step, total_steps, lambda_max, lambda_gamma)
        if batch_size < N:
            key, subkey = jax.random.split(key)
            idx = jax.random.randint(subkey, (batch_size,), 0, N)
            bx = train_x[idx]
            by = train_y[idx]
        else:
            bx = train_x
            by = train_y
        model, opt_state, metrics = jit_step(model, opt_state, bx, by, jnp.float32(lam))
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


# ------------------------------------------------------------------
# ternary multiclass loss
def make_ternary_loss(group_sum) -> Callable:
    """Build the ternary multiclass loss closure.

    vmaps ``group_sum`` (a GroupSum module) over the per-neuron predictions to
    produce class logits, then returns the mean
    ``optax.softmax_cross_entropy_with_integer_labels`` against the integer
    labels (flattened with ``.ravel()``). Shared by all CIFAR notebooks.
    """
    def loss_fn(preds, targets):
        logits = jax.vmap(group_sum)(preds)
        return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(
            logits, targets.astype(jnp.int32).ravel()))
    return loss_fn


# ------------------------------------------------------------------
# cosine-warmup training loop
def train_cosine_schedule(
    model,
    optimizer: optax.GradientTransformation,
    train_x: jnp.ndarray,
    train_y: jnp.ndarray,
    total_steps: int,
    lambda_max: float,
    batch_size: Optional[int] = None,
    loss_fn: Optional[Callable] = None,
    key: Optional[jax.Array] = None,
    log_every: int = 100,
) -> tuple:
    """Variant of ``train()`` with a cosine warmup for the regularizer weight.

    Identical to ``train()`` except the commitment-regularizer weight ramps as
    ``lambda(t) = lambda_max * 0.5 * (1 - cos(pi * t / T))`` instead of the
    polynomial ``lambda_schedule``.
    """
    if key is None:
        key = jax.random.PRNGKey(0)
    N = train_x.shape[0]
    if batch_size is None:
        batch_size = N
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    history = []

    @eqx.filter_jit
    def jit_step(m, ost, bx, by, lam):
        return train_step(m, ost, optimizer, bx, by, lam, loss_fn)

    for step in range(total_steps):
        progress = step / total_steps
        lam = lambda_max * 0.5 * (1.0 - math.cos(math.pi * progress))

        if batch_size < N:
            key, subkey = jax.random.split(key)
            idx = jax.random.randint(subkey, (batch_size,), 0, N)
            bx = train_x[idx]
            by = train_y[idx]
        else:
            bx = train_x
            by = train_y

        model, opt_state, metrics = jit_step(
            model, opt_state, bx, by, jnp.float32(lam)
        )

        if (step % log_every == 0) or (step == total_steps - 1):
            history.append({
                "step": step,
                "lambda": lam,
                **{k: float(v) for k, v in metrics.items()},
            })
    return model, history
