from __future__ import annotations
import numpy as np
import jax.numpy as jnp
from pst_dtlgn.core.constants import V
from pst_dtlgn.core.fourier import (
    fourier_coefficients, fourier_sparsity, fourier_l1_norm,
    fourier_energy_breakdown, FOURIER_LABELS,
)


def neuron_spectral_analysis(weights: np.ndarray) -> dict:
    w = jnp.asarray(weights, dtype=jnp.float32)
    tt = V @ w
    coeffs = fourier_coefficients(tt)
    sparsity = fourier_sparsity(tt)
    l1 = fourier_l1_norm(tt)
    energy = fourier_energy_breakdown(tt)
    abs_coeffs = jnp.abs(coeffs)
    dominant_idx = int(jnp.argmax(abs_coeffs))
    return {
        "truth_table": np.asarray(tt),
        "fourier_coeffs": np.asarray(coeffs),
        "fourier_labels": FOURIER_LABELS,
        "sparsity": sparsity,
        "l1_norm": l1,
        "energy_breakdown": energy,
        "dominant_term": FOURIER_LABELS[dominant_idx],
    }


def network_spectral_summary(model) -> dict:
    sparsities = []
    l1_norms = []
    energies = {"constant": [], "linear": [], "quadratic": [], "cubic": [], "quartic": []}
    dominant_counts: dict[str, int] = {}
    for layer in model.layers:
        W = np.asarray(layer.weights)
        for j in range(W.shape[0]):
            analysis = neuron_spectral_analysis(W[j])
            sparsities.append(analysis["sparsity"])
            l1_norms.append(analysis["l1_norm"])
            for deg, val in analysis["energy_breakdown"].items():
                energies[deg].append(val)
            dom = analysis["dominant_term"]
            dominant_counts[dom] = dominant_counts.get(dom, 0) + 1
    sparsities = np.array(sparsities)
    l1_norms = np.array(l1_norms)
    sparsity_hist: dict[int, int] = {}
    for s in sparsities:
        sparsity_hist[int(s)] = sparsity_hist.get(int(s), 0) + 1
    mean_energy = {}
    for deg in energies:
        vals = np.array(energies[deg])
        mean_energy[deg] = float(np.mean(vals)) if len(vals) > 0 else 0.0
    return {
        "mean_sparsity": float(np.mean(sparsities)),
        "std_sparsity": float(np.std(sparsities)),
        "mean_l1_norm": float(np.mean(l1_norms)),
        "std_l1_norm": float(np.std(l1_norms)),
        "mean_energy": mean_energy,
        "total_neurons": len(sparsities),
        "sparsity_histogram": dict(sorted(sparsity_hist.items())),
        "dominant_term_counts": dict(sorted(dominant_counts.items(), key=lambda x: -x[1])),
    }


def layer_spectral_report(model) -> list[dict]:
    reports = []
    for l_idx, layer in enumerate(model.layers):
        W = np.asarray(layer.weights)
        n = W.shape[0]
        sparsities = []
        l1_norms = []
        energies = {"constant": [], "linear": [], "quadratic": [], "cubic": [], "quartic": []}
        for j in range(n):
            analysis = neuron_spectral_analysis(W[j])
            sparsities.append(analysis["sparsity"])
            l1_norms.append(analysis["l1_norm"])
            for deg, val in analysis["energy_breakdown"].items():
                energies[deg].append(val)
        mean_energy = {}
        for deg in energies:
            vals = np.array(energies[deg])
            mean_energy[deg] = float(np.mean(vals)) if len(vals) > 0 else 0.0
        reports.append({
            "layer_idx": l_idx, "width": n,
            "mean_sparsity": float(np.mean(sparsities)),
            "mean_l1_norm": float(np.mean(l1_norms)),
            "mean_energy": mean_energy,
        })
    return reports


def format_spectral_report(model) -> str:
    summary = network_spectral_summary(model)
    layer_reports = layer_spectral_report(model)
    lines = []
    lines.append("=" * 60)
    lines.append("FOURIER SPECTRAL ANALYSIS")
    lines.append("=" * 60)
    lines.append(f"Total neurons:     {summary['total_neurons']}")
    lines.append(f"Mean sparsity:     {summary['mean_sparsity']:.1f} "
                 f"± {summary['std_sparsity']:.1f}")
    lines.append(f"Mean L1 norm:      {summary['mean_l1_norm']:.3f} "
                 f"± {summary['std_l1_norm']:.3f}")
    lines.append("")
    lines.append("Mean energy by polynomial degree:")
    for deg, val in summary["mean_energy"].items():
        bar = "█" * int(val * 40)
        lines.append(f"  {deg:10s}  {val:6.1%}  {bar}")
    lines.append("")
    lines.append("Sparsity distribution (nonzero Fourier coefficients):")
    for s, count in sorted(summary["sparsity_histogram"].items()):
        frac = count / summary["total_neurons"]
        lines.append(f"  {s} terms:  {count:4d}  ({frac:.1%})")
    lines.append("")
    lines.append("Dominant Fourier term distribution:")
    for label, count in list(summary["dominant_term_counts"].items())[:9]:
        frac = count / summary["total_neurons"]
        lines.append(f"  {label:6s}  {count:4d}  ({frac:.1%})")
    lines.append("")
    lines.append("Per-layer spectral summary:")
    lines.append(f"  {'Layer':>5s}  {'Width':>5s}  {'Sparsity':>8s}  "
                 f"{'L1 norm':>8s}  {'Linear':>7s}  {'Quad':>6s}")
    lines.append(f"  {'-'*5}  {'-'*5}  {'-'*8}  {'-'*8}  {'-'*7}  {'-'*6}")
    for lr in layer_reports:
        lines.append(
            f"  {lr['layer_idx']:5d}  {lr['width']:5d}  "
            f"{lr['mean_sparsity']:8.1f}  {lr['mean_l1_norm']:8.3f}  "
            f"{lr['mean_energy']['linear']:7.1%}  "
            f"{lr['mean_energy']['quadratic']:6.1%}"
        )
    lines.append("=" * 60)
    return "\n".join(lines)
