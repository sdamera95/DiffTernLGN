from __future__ import annotations
from typing import Callable
import numpy as np
import jax
import jax.numpy as jnp


# ------------------------------------------------------------------
# soft-forward selection
def get_soft_fn(model) -> Callable:
    """Return the soft-forward callable for a model.

    SmoothedPSTNetwork exposes ``soft_forward`` (bypasses smoothing); plain
    PolynomialNetwork / BinaryDLGN have no such attribute, so the model itself
    is returned unchanged.
    """
    return getattr(model, "soft_forward", model)


# ------------------------------------------------------------------
# multiclass soft / circuit evaluation
def evaluate_soft(model, x, group_sum, batch_size=256) -> np.ndarray:
    """Batched soft forward pass -> argmax class predictions over the full set.

    Runs ``get_soft_fn(model)`` then ``group_sum`` per batch and takes the
    argmax. ``x`` may be a numpy or jax array.
    """
    forward_fn = get_soft_fn(model)
    N = len(x)
    all_preds = []
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch = jnp.array(x[start:end])
        raw = jax.vmap(forward_fn)(batch)
        logits = jax.vmap(group_sum)(raw)
        preds = jnp.argmax(logits, axis=-1)
        all_preds.append(np.array(preds))
    return np.concatenate(all_preds)


def evaluate_circuit(circuit, x, n_classes, max_samples=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-sample discrete circuit evaluation (07-family 3-tuple form).

    Reshapes the discrete output into ``n_classes`` groups and sums each group
    to a score. Returns ``(preds, margins, unknown_fracs)`` where margins are
    top1 minus top2 score and unknown_fracs is the mean fraction of zero
    outputs per sample.
    """
    x_np = np.array(x, dtype=np.int8)
    N = len(x_np) if max_samples is None else min(len(x_np), max_samples)
    preds = np.zeros(N, dtype=np.int32)
    margins = np.zeros(N, dtype=np.float32)
    unknown_fracs = np.zeros(N, dtype=np.float32)

    for i in range(N):
        out = circuit(x_np[i])
        out_f = np.array(out, dtype=np.float32)
        neurons_per_group = len(out_f) // n_classes
        scores = out_f.reshape(n_classes, neurons_per_group).sum(axis=-1)

        sorted_scores = np.sort(scores)[::-1]
        preds[i] = int(np.argmax(scores))
        margins[i] = sorted_scores[0] - sorted_scores[1]
        unknown_fracs[i] = float(np.mean(out_f == 0))

    return preds, margins, unknown_fracs


def evaluate_circuit_multiclass(circuit, x_np, n_classes, max_samples=None, unknown_val=0) -> tuple[np.ndarray, np.ndarray]:
    """Per-sample discrete circuit evaluation (09-family 2-tuple form).

    Like :func:`evaluate_circuit` but without margins. Returns
    ``(preds, unknown_fracs)`` where unknown_fracs is the mean fraction of
    outputs equal to ``unknown_val`` per sample (skipped if ``unknown_val`` is
    None).
    """
    N = len(x_np) if max_samples is None else min(len(x_np), max_samples)
    preds = np.zeros(N, dtype=np.int32)
    unknown_fracs = np.zeros(N, dtype=np.float32)
    for i in range(N):
        out = circuit(x_np[i])
        out_f = np.array(out, dtype=np.float32)
        neurons_per_class = len(out_f) // n_classes
        scores = out_f.reshape(n_classes, neurons_per_class).sum(axis=-1)
        preds[i] = int(np.argmax(scores))
        if unknown_val is not None:
            unknown_fracs[i] = float(np.mean(out_f == unknown_val))
    return preds, unknown_fracs


def accuracy_vs_coverage(preds, labels, margins, n_thresholds=200) -> tuple[np.ndarray, np.ndarray]:
    """Selective-classification curve over margin thresholds.

    Sweeps thresholds from 0 to ``max(margins)`` and, on the retained
    ``margins >= t`` subset, records coverage and accuracy, stopping once
    coverage drops below 1%. Returns ``(coverages, accuracies)``.
    """
    thresholds = np.linspace(0, np.max(margins), n_thresholds)
    coverages, accuracies = [], []
    for t in thresholds:
        mask = margins >= t
        cov = mask.mean()
        if cov < 0.01:
            break
        acc = np.mean(preds[mask] == labels[mask])
        coverages.append(float(cov))
        accuracies.append(float(acc))
    return np.array(coverages), np.array(accuracies)


# ------------------------------------------------------------------
# binary evaluation
def evaluate_binary_soft(model, x) -> np.ndarray:
    """Binary DLGN soft predictions thresholded at 0.5 -> {0, 1}."""
    preds = np.array(jax.vmap(model)(x))
    return (preds[:, 0] > 0.5).astype(np.int32)


def evaluate_binary_circuit(circuit, x) -> np.ndarray:
    """Hardened binary circuit predictions -> {0, 1} (takes output column 0)."""
    x_np = np.array(x, dtype=np.int8)
    preds = np.array([circuit(x_np[i]) for i in range(len(x_np))])
    return preds[:, 0].astype(np.int32)


# ------------------------------------------------------------------
# ternary evaluation
def evaluate_ternary_soft(model, x) -> np.ndarray:
    """Ternary soft predictions -> {1, 0, -1=UNKNOWN} by sign of output column 0.

    Uses ``soft_forward`` if available (SmoothedPSTNetwork).
    """
    forward_fn = getattr(model, "soft_forward", model)
    preds = np.array(jax.vmap(forward_fn)(x))
    return np.where(preds[:, 0] > 0, 1, np.where(preds[:, 0] < 0, 0, -1)).astype(np.int32)


def evaluate_ternary_circuit(circuit, x) -> np.ndarray:
    """Hardened ternary circuit predictions -> {1, 0, -1=UNKNOWN}.

    Remaps the raw discrete output column 0 ({1, -1, 0}) to {1, 0, -1}.
    """
    x_np = np.array(x, dtype=np.int8)
    preds = np.array([circuit(x_np[i]) for i in range(len(x_np))])
    raw = preds[:, 0]
    return np.where(raw == 1, 1, np.where(raw == -1, 0, -1)).astype(np.int32)


def accuracy_excluding_unknown(preds, targets) -> tuple[float, int]:
    """Accuracy over non-UNKNOWN (``!= -1``) predictions.

    Returns ``(accuracy, n_decided)``; accuracy is 0.0 when no sample is decided.
    """
    mask = preds != -1
    n_decided = int(mask.sum())
    if n_decided == 0:
        return 0.0, 0
    return float(np.mean(preds[mask] == targets[mask])), n_decided


# ------------------------------------------------------------------
# 2D decision-boundary grids
def predict_grid_ternary(circuit, grid_points, pipeline) -> np.ndarray:
    """Hardened ternary circuit predictions on 2D grid points -> {-1, 0, +1}.

    Encodes ``grid_points`` through ``pipeline`` in ``mode="hard"`` and runs the
    circuit per point, returning output column 0 for contour plotting.
    """
    enc = pipeline.transform(grid_points, mode="hard")
    x_np = np.array(enc, dtype=np.int8)
    return np.array([circuit(x_np[i])[0] for i in range(len(x_np))], dtype=np.float32)


def predict_grid_binary(circuit, grid_points, pipeline) -> np.ndarray:
    """Hardened binary circuit predictions on 2D grid points -> {0, 1}.

    Encodes ``grid_points`` through ``pipeline`` in ``mode="hard"`` and runs the
    circuit per point, returning output column 0 for contour plotting.
    """
    enc = pipeline.transform(grid_points, mode="hard")
    x_np = np.array(enc, dtype=np.int8)
    return np.array([circuit(x_np[i])[0] for i in range(len(x_np))], dtype=np.float32)
