from __future__ import annotations
from collections import Counter
import numpy as np
from pst_dtlgn.core.gate_library import GateLibrary, CURATED_GATES


def gate_census(disc_result: dict) -> dict:
    counts = dict(disc_result["gate_census"])
    total = disc_result["total_neurons"]
    sorted_gates = sorted(counts.items(), key=lambda x: -x[1])
    if total > 0:
        probs = np.array([c / total for _, c in sorted_gates])
        entropy = float(-np.sum(probs * np.log2(probs + 1e-12)))
    else:
        entropy = 0.0
    return {
        "gate_counts": counts, "unique_gates": len(counts),
        "total_neurons": total, "top_gates": sorted_gates, "entropy": entropy,
    }


def curated_coverage_report(disc_result: dict, gate_library: GateLibrary) -> dict:
    curated_count = 0
    curated_h1 = 0
    curated_breakdown: dict[str, int] = {}
    hamming_dist: dict[int, int] = {}
    total = 0
    for layer_results in disc_result["per_layer"]:
        for neuron_result in layer_results:
            total += 1
            hd = neuron_result["hamming_to_curated"]
            hamming_dist[hd] = hamming_dist.get(hd, 0) + 1
            if hd == 0 and neuron_result["curated_name"] is not None:
                curated_count += 1
                name = neuron_result["curated_name"]
                curated_breakdown[name] = curated_breakdown.get(name, 0) + 1
            if hd <= 1:
                curated_h1 += 1
    total = max(total, 1)
    return {
        "curated_coverage": curated_count / total,
        "curated_coverage_h1": curated_h1 / total,
        "curated_breakdown": curated_breakdown,
        "hamming_distribution": dict(sorted(hamming_dist.items())),
        "uncurated_fraction": 1.0 - curated_count / total,
        "total_neurons": total,
    }


def layer_gate_report(disc_result: dict) -> list[dict]:
    reports = []
    for l_idx, layer_results in enumerate(disc_result["per_layer"]):
        n = len(layer_results)
        if n == 0:
            continue
        gate_counts: dict[int, int] = {}
        curated_count = 0
        total_error = 0.0
        for nr in layer_results:
            gidx = nr["gate_index"]
            gate_counts[gidx] = gate_counts.get(gidx, 0) + 1
            total_error += nr["hardening_error"]
            if nr["hamming_to_curated"] == 0 and nr["curated_name"] is not None:
                curated_count += 1
        sorted_gates = sorted(gate_counts.items(), key=lambda x: -x[1])[:5]
        gate_names = []
        for gidx, _ in sorted_gates:
            name = None
            for nr in layer_results:
                if nr["gate_index"] == gidx:
                    if nr["hamming_to_curated"] == 0:
                        name = nr["curated_name"]
                    break
            gate_names.append(name or "uncurated")
        reports.append({
            "layer_idx": l_idx, "width": n,
            "unique_gates": len(gate_counts),
            "mean_hardening_error": total_error / n,
            "curated_count": curated_count,
            "curated_fraction": curated_count / n,
            "top_gates": sorted_gates, "gate_names": gate_names,
        })
    return reports


def format_census_table(disc_result: dict, gate_library: GateLibrary, max_rows: int = 20) -> str:
    census = gate_census(disc_result)
    curated = curated_coverage_report(disc_result, gate_library)
    lines = []
    lines.append("=" * 60)
    lines.append("GATE CENSUS REPORT")
    lines.append("=" * 60)
    lines.append(f"Total neurons:     {census['total_neurons']}")
    lines.append(f"Unique gates:      {census['unique_gates']}")
    lines.append(f"Gate entropy:      {census['entropy']:.2f} bits")
    lines.append(f"Curated coverage:  {curated['curated_coverage']:.1%}")
    lines.append(f"Curated (H≤1):    {curated['curated_coverage_h1']:.1%}")
    lines.append("")
    if curated["curated_breakdown"]:
        lines.append("Curated gate breakdown:")
        for name, count in sorted(curated["curated_breakdown"].items(), key=lambda x: -x[1]):
            frac = count / census["total_neurons"]
            lines.append(f"  {name:16s}  {count:4d}  ({frac:.1%})")
        lines.append("")
    lines.append("Hamming distance to nearest curated gate:")
    for hd, count in sorted(curated["hamming_distribution"].items()):
        frac = count / curated["total_neurons"]
        lines.append(f"  H={hd}:  {count:4d}  ({frac:.1%})")
    lines.append("")
    lines.append(f"Top {min(max_rows, len(census['top_gates']))} gates by frequency:")
    lines.append(f"  {'Gate Index':>12s}  {'Count':>6s}  {'Frac':>7s}  {'Name':s}")
    lines.append(f"  {'-'*12}  {'-'*6}  {'-'*7}  {'-'*16}")
    for gidx, count in census["top_gates"][:max_rows]:
        frac = count / census["total_neurons"]
        info = gate_library.lookup_by_index(gidx)
        name = info["curated_name"] or "-"
        lines.append(f"  {gidx:12d}  {count:6d}  {frac:7.1%}  {name}")
    lines.append("=" * 60)
    return "\n".join(lines)
