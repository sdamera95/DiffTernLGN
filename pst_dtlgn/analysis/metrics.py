from __future__ import annotations
import numpy as np
import jax
import jax.numpy as jnp
from pst_dtlgn.network.inference import LearnedCircuitBase
from pst_dtlgn.network.harden import LearnedCircuit, round_ternary


def hardening_error_report(disc_result: dict) -> dict:
    all_errors = []
    per_layer_errors = []
    for layer_results in disc_result["per_layer"]:
        layer_errs = [nr["hardening_error"] for nr in layer_results]
        per_layer_errors.append(layer_errs)
        all_errors.extend(layer_errs)
    all_errors = np.array(all_errors)
    total = len(all_errors)
    percentiles = {}
    for p in [25, 50, 75, 90, 95, 99]:
        percentiles[p] = float(np.percentile(all_errors, p))
    thresholds = {0.01: 0, 0.05: 0, 0.1: 0, 0.2: 0, 0.5: 0}
    for th in thresholds:
        thresholds[th] = int(np.sum(all_errors < th))
    return {
        "mean_error": float(np.mean(all_errors)),
        "max_error": float(np.max(all_errors)),
        "min_error": float(np.min(all_errors)),
        "median_error": float(np.median(all_errors)),
        "per_layer_mean": [float(np.mean(le)) if le else 0.0 for le in per_layer_errors],
        "per_layer_max": [float(np.max(le)) if le else 0.0 for le in per_layer_errors],
        "error_percentiles": percentiles,
        "neurons_below_threshold": thresholds,
        "total_neurons": total,
    }


def hard_soft_agreement(
    model, hard_model: LearnedCircuitBase, inputs: np.ndarray,
) -> dict:
    N = inputs.shape[0]
    soft_outputs = []
    hard_outputs = []
    for i in range(N):
        x_soft = jnp.asarray(inputs[i], dtype=jnp.float32)
        soft_out = np.asarray(model(x_soft))
        soft_outputs.append(soft_out)
        x_hard = np.asarray(inputs[i], dtype=np.int8)
        hard_out = np.asarray(hard_model(x_hard))
        hard_outputs.append(hard_out)
    soft_arr = np.array(soft_outputs, dtype=np.float64)
    hard_arr = np.array(hard_outputs, dtype=np.float64)
    soft_rounded = hard_model.round_fn(soft_arr)
    hard_rounded = hard_arr.astype(np.int8)
    exact = np.mean(soft_rounded == hard_rounded)
    abs_diff = np.abs(soft_arr - hard_arr)
    mean_diff = float(np.mean(abs_diff))
    max_diff = float(np.max(abs_diff))
    output_dim = soft_arr.shape[1]
    per_output = np.array([
        float(np.mean(soft_rounded[:, d] == hard_rounded[:, d]))
        for d in range(output_dim)
    ])
    soft_sign = np.sign(soft_arr)
    hard_sign = np.sign(hard_arr)
    nonzero_mask = (soft_sign != 0) & (hard_sign != 0)
    if np.any(nonzero_mask):
        sign_agree = float(np.mean(soft_sign[nonzero_mask] == hard_sign[nonzero_mask]))
    else:
        sign_agree = 1.0
    return {
        "exact_agreement": float(exact),
        "mean_abs_diff": mean_diff,
        "max_abs_diff": max_diff,
        "per_output_agreement": per_output,
        "sign_agreement": sign_agree,
        "num_samples": N,
    }


def gate_diversity(disc_result: dict) -> dict:
    counts = disc_result["gate_census"]
    total = disc_result["total_neurons"]
    n_unique = len(counts)
    num_gates = disc_result.get("num_gates", 19683)
    if total == 0:
        return {
            "unique_gates": 0, "total_neurons": 0, "utilization": 0.0,
            "effective_diversity": 0.0, "max_diversity": np.log2(num_gates),
            "gini_coefficient": 0.0,
        }
    freq = np.array(list(counts.values()), dtype=np.float64)
    probs = freq / total
    entropy = float(-np.sum(probs * np.log(probs + 1e-12)))
    effective_div = float(np.exp(entropy))
    freq_sorted = np.sort(freq)
    n = len(freq_sorted)
    if n == 1:
        gini = 0.0
    else:
        index = np.arange(1, n + 1)
        gini = float(
            (2 * np.sum(index * freq_sorted) - (n + 1) * np.sum(freq_sorted))
            / (n * np.sum(freq_sorted))
        )
    return {
        "unique_gates": n_unique, "total_neurons": total,
        "utilization": n_unique / total,
        "effective_diversity": effective_div,
        "max_diversity": np.log2(num_gates),
        "gini_coefficient": gini,
    }


def functional_redundancy(disc_result: dict) -> dict:
    counts = disc_result["gate_census"]
    total = disc_result["total_neurons"]
    n_unique = len(counts)
    if total == 0:
        return {
            "total_neurons": 0, "unique_gates": 0, "redundancy_ratio": 0.0,
            "max_copies": 0, "max_copies_gate": -1,
            "singletons": 0, "singleton_fraction": 0.0,
        }
    max_gate = max(counts, key=counts.get)
    max_count = counts[max_gate]
    singletons = sum(1 for c in counts.values() if c == 1)
    return {
        "total_neurons": total, "unique_gates": n_unique,
        "redundancy_ratio": 1.0 - n_unique / total,
        "max_copies": max_count, "max_copies_gate": max_gate,
        "singletons": singletons,
        "singleton_fraction": singletons / n_unique if n_unique > 0 else 0.0,
    }


def format_metrics_report(
    disc_result: dict, model=None,
    hard_model: LearnedCircuitBase | None = None,
    test_inputs: np.ndarray | None = None,
) -> str:
    err = hardening_error_report(disc_result)
    div = gate_diversity(disc_result)
    red = functional_redundancy(disc_result)
    lines = []
    lines.append("=" * 60)
    lines.append("HARDENING METRICS REPORT")
    lines.append("=" * 60)
    lines.append("\n--- Hardening Error ---")
    lines.append(f"  Mean:     {err['mean_error']:.6f}")
    lines.append(f"  Median:   {err['median_error']:.6f}")
    lines.append(f"  Max:      {err['max_error']:.6f}")
    lines.append(f"  P90:      {err['error_percentiles'][90]:.6f}")
    lines.append(f"  P99:      {err['error_percentiles'][99]:.6f}")
    lines.append(f"  Below 0.01: {err['neurons_below_threshold'][0.01]}/{err['total_neurons']}")
    lines.append(f"  Below 0.1:  {err['neurons_below_threshold'][0.1]}/{err['total_neurons']}")
    lines.append("\n  Per-layer mean error:")
    for l, (mean_e, max_e) in enumerate(zip(err["per_layer_mean"], err["per_layer_max"])):
        lines.append(f"    Layer {l}: mean={mean_e:.6f}, max={max_e:.6f}")
    lines.append("\n--- Gate Diversity ---")
    lines.append(f"  Unique gates:         {div['unique_gates']}")
    lines.append(f"  Total neurons:        {div['total_neurons']}")
    lines.append(f"  Utilization:          {div['utilization']:.1%}")
    lines.append(f"  Effective diversity:  {div['effective_diversity']:.1f}")
    lines.append(f"  Gini coefficient:     {div['gini_coefficient']:.3f}")
    lines.append("\n--- Functional Redundancy ---")
    lines.append(f"  Redundancy ratio:     {red['redundancy_ratio']:.1%}")
    lines.append(f"  Max copies of 1 gate: {red['max_copies']}")
    lines.append(f"  Singleton gates:      {red['singletons']} ({red['singleton_fraction']:.1%})")
    if model is not None and hard_model is not None and test_inputs is not None:
        agree = hard_soft_agreement(model, hard_model, test_inputs)
        lines.append("\n--- Hard-Soft Agreement ---")
        lines.append(f"  Exact agreement:  {agree['exact_agreement']:.1%}")
        lines.append(f"  Sign agreement:   {agree['sign_agreement']:.1%}")
        lines.append(f"  Mean |diff|:      {agree['mean_abs_diff']:.6f}")
        lines.append(f"  Max |diff|:       {agree['max_abs_diff']:.6f}")
        lines.append(f"  Test samples:     {agree['num_samples']}")
    lines.append("\n" + "=" * 60)
    return "\n".join(lines)
