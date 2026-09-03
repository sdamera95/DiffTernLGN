from __future__ import annotations
import time
from typing import Callable
import jax
import jax.numpy as jnp
import numpy as np
from pst_dtlgn.core.constants import V
from pst_dtlgn.core.gate_library import GateLibrary
from pst_dtlgn.network.harden import (
    harden_layers, round_ternary,
    _hard_layer_forward, _hardened_first_layer_forward,
)
from pst_dtlgn.binary_baseline.harden import (
    BinaryLearnedCircuit, round_binary,
)


class LayerwiseHardenedNetwork:

    def __init__(self, soft_model, disc_result: dict, hard_layers: int) -> None:
        self.soft_model = soft_model
        self.truth_tables = disc_result["truth_tables"]
        self.hardened_weights = disc_result.get("hardened_weights", None)
        self.connections_hard = disc_result["connections"]
        self.hard_layers = hard_layers
        self.depth = len(soft_model.layers)
        if hard_layers < 0 or hard_layers > self.depth:
            raise ValueError(f"hard_layers must be in [0, {self.depth}], got {hard_layers}")

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self.hard_layers == 0:
            return np.asarray(self.soft_model(jnp.asarray(x, dtype=jnp.float32)))
        if self.hard_layers == self.depth:
            h = np.asarray(x, dtype=np.float64)
            if self.hardened_weights is not None:
                h = _hardened_first_layer_forward(
                    self.hardened_weights[0], self.connections_hard[0], h)
            else:
                h = round_ternary(h)
                h = _hard_layer_forward(self.truth_tables[0], self.connections_hard[0], h)
            for tt, conn in zip(self.truth_tables[1:], self.connections_hard[1:]):
                h = _hard_layer_forward(tt, conn, h)
            return h
        h = np.asarray(x, dtype=np.float64)
        if self.hardened_weights is not None:
            h = _hardened_first_layer_forward(
                self.hardened_weights[0], self.connections_hard[0], h)
        else:
            h = round_ternary(h)
            h = _hard_layer_forward(self.truth_tables[0], self.connections_hard[0], h)
        for l in range(1, self.hard_layers):
            h = _hard_layer_forward(self.truth_tables[l], self.connections_hard[l], h)
        h_float = jnp.asarray(h, dtype=jnp.float32)
        for l in range(self.hard_layers, self.depth):
            h_float = self.soft_model.layers[l](h_float)
        return np.asarray(h_float)

    @property
    def label(self) -> str:
        if self.hard_layers == 0:
            return "Fully Soft"
        if self.hard_layers == self.depth:
            return "Fully Hard"
        return f"Hard L0-{self.hard_layers - 1} / Soft L{self.hard_layers}-{self.depth - 1}"


class BinaryLayerwiseHardenedNetwork:

    def __init__(self, soft_model, harden_result: dict, hard_layers: int) -> None:
        self.soft_model = soft_model
        self.harden_result = harden_result
        self.hard_layers = hard_layers
        self.depth = len(soft_model.layers)
        if hard_layers < 0 or hard_layers > self.depth:
            raise ValueError(f"hard_layers must be in [0, {self.depth}], got {hard_layers}")

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self.hard_layers == 0:
            return np.asarray(self.soft_model(jnp.asarray(x, dtype=jnp.float32)))
        if self.hard_layers == self.depth:
            circuit = BinaryLearnedCircuit(self.harden_result)
            x_bin = round_binary(np.asarray(x))
            return circuit(x_bin)
        h = jnp.asarray(round_binary(np.asarray(x)), dtype=jnp.float32)
        for l in range(self.hard_layers):
            h = self.soft_model.layers[l].hard(h)
            h = jnp.where(h > 0.5, 1.0, 0.0)
        for l in range(self.hard_layers, self.depth):
            h = self.soft_model.layers[l].soft(h)
        return np.asarray(h)

    @property
    def label(self) -> str:
        if self.hard_layers == 0:
            return "Fully Soft"
        if self.hard_layers == self.depth:
            return "Fully Hard"
        return f"Hard L0-{self.hard_layers - 1} / Soft L{self.hard_layers}-{self.depth - 1}"


# --- Ternary benchmarking ---

def benchmark_inference(soft_model, disc_result: dict, test_inputs: np.ndarray,
                        n_warmup: int = 3, n_repeats: int = 10) -> list[dict]:
    depth = len(soft_model.layers)
    inputs = np.asarray(test_inputs)
    N = inputs.shape[0]
    results = []
    soft_time = None
    for level in range(depth + 1):
        model = LayerwiseHardenedNetwork(soft_model, disc_result, level)
        for _ in range(n_warmup):
            for i in range(min(5, N)):
                _ = model(inputs[i])
        times = []
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            for i in range(N):
                _ = model(inputs[i])
            t1 = time.perf_counter()
            times.append(t1 - t0)
        times = np.array(times)
        mean_total = float(np.mean(times))
        std_total = float(np.std(times))
        mean_per_sample = mean_total / max(N, 1) * 1e6
        std_per_sample = std_total / max(N, 1) * 1e6
        if level == 0:
            soft_time = mean_per_sample
        speedup = soft_time / mean_per_sample if mean_per_sample > 0 else float("inf")
        results.append({
            "level": level, "hard_layers": level, "total_layers": depth,
            "label": model.label,
            "mean_time_per_sample_us": mean_per_sample,
            "std_time_per_sample_us": std_per_sample,
            "total_time_ms": mean_total * 1e3,
            "speedup_vs_soft": speedup, "n_samples": N, "n_repeats": n_repeats,
        })
    return results


def benchmark_accuracy(soft_model, disc_result: dict, test_inputs: np.ndarray,
                       test_labels: np.ndarray, threshold: float = 0.3) -> list[dict]:
    depth = len(soft_model.layers)
    inputs = np.asarray(test_inputs)
    labels = np.asarray(test_labels).ravel()
    N_labels = len(labels)

    def _get_verdict(model, level_idx):
        raw_list = [model(inputs[i]) for i in range(inputs.shape[0])]
        raw_arr = np.array(raw_list)
        if raw_arr.ndim > 1 and raw_arr.shape[1] > 1:
            raw_per_sample = raw_arr[:, 0]
        else:
            raw_per_sample = raw_arr.ravel()
        raw_per_sample = raw_per_sample.astype(np.float32)
        if level_idx == depth:
            preds = np.where(raw_per_sample > 0.5, 1, np.where(raw_per_sample < -0.5, -1, 0)).astype(np.int8)
        else:
            preds = np.where(raw_per_sample > threshold, 1, np.where(raw_per_sample < -threshold, -1, 0)).astype(np.int8)
        return raw_per_sample, preds

    soft_model_wrap = LayerwiseHardenedNetwork(soft_model, disc_result, 0)
    soft_raw, soft_preds = _get_verdict(soft_model_wrap, 0)
    results = []
    for level in range(depth + 1):
        model = LayerwiseHardenedNetwork(soft_model, disc_result, level)
        raw_per_sample, preds = _get_verdict(model, level)
        n = min(len(preds), N_labels)
        preds = preds[:n]; raw_sample = raw_per_sample[:n]
        labs = labels[:n]; soft_p = soft_preds[:n]
        accuracy = float(np.mean(preds == labs))
        agreement = float(np.mean(preds == soft_p))
        per_class = {}
        for val, name in [(-1, "FALSE"), (0, "UNKNOWN"), (1, "TRUE")]:
            mask = labs == val
            if mask.sum() > 0:
                per_class[name] = float(np.mean(preds[mask] == val))
            else:
                per_class[name] = float("nan")
        mean_abs = float(np.mean(np.abs(raw_sample)))
        results.append({
            "level": level, "hard_layers": level, "total_layers": depth,
            "label": model.label, "accuracy": accuracy,
            "agreement_with_soft": agreement, "per_class_accuracy": per_class,
            "mean_abs_output": mean_abs,
        })
    return results


def benchmark_full(soft_model, disc_result: dict, test_inputs: np.ndarray,
                   test_labels: np.ndarray, threshold: float = 0.3,
                   n_warmup: int = 3, n_repeats: int = 10,
                   timing_inputs: np.ndarray | None = None) -> dict:
    speed_inputs = timing_inputs if timing_inputs is not None else test_inputs
    speed_results = benchmark_inference(soft_model, disc_result, speed_inputs, n_warmup=n_warmup, n_repeats=n_repeats)
    accuracy_results = benchmark_accuracy(soft_model, disc_result, test_inputs, test_labels, threshold=threshold)
    levels = [{**s, **a} for s, a in zip(speed_results, accuracy_results)]
    lines = []
    lines.append("=" * 80)
    lines.append("HARDENING LEVEL BENCHMARK: ACCURACY vs SPEED")
    lines.append("=" * 80)
    lines.append(f"{'Level':>5s}  {'Mode':<35s}  {'Accuracy':>8s}  {'Agree/Soft':>10s}  {'us/sample':>10s}  {'Speedup':>7s}")
    lines.append("-" * 80)
    for lvl in levels:
        lines.append(f"{lvl['level']:>5d}  {lvl['label']:<35s}  {lvl['accuracy']:>8.1%}  {lvl['agreement_with_soft']:>10.1%}  {lvl['mean_time_per_sample_us']:>10.1f}  {lvl['speedup_vs_soft']:>6.1f}x")
    lines.append("-" * 80)
    soft = levels[0]; hard = levels[-1]
    acc_drop = soft["accuracy"] - hard["accuracy"]
    speedup = hard["speedup_vs_soft"]
    lines.append(f"\nSoft -> Hard: accuracy change = {acc_drop:+.1%}, speedup = {speedup:.1f}x")
    if len(levels) > 2:
        soft_acc = soft["accuracy"]
        candidates = [l for l in levels[1:-1] if l["accuracy"] >= 0.9 * soft_acc]
        if candidates:
            best = max(candidates, key=lambda l: l["speedup_vs_soft"])
            lines.append(f"Best intermediate: {best['label']} — accuracy={best['accuracy']:.1%}, speedup={best['speedup_vs_soft']:.1f}x")
    lines.append("=" * 80)
    summary = "\n".join(lines)
    depth = levels[0]["total_layers"]
    return {"levels": levels, "summary": summary, "depth": depth}


def format_benchmark_table(results: dict) -> str:
    return results["summary"]


# --- Binary benchmarking ---

def benchmark_binary_inference(soft_model, harden_result: dict, test_inputs: np.ndarray,
                               n_warmup: int = 3, n_repeats: int = 10) -> list[dict]:
    depth = len(soft_model.layers)
    inputs = np.asarray(test_inputs)
    N = inputs.shape[0]
    results = []
    soft_time = None
    for level in range(depth + 1):
        model = BinaryLayerwiseHardenedNetwork(soft_model, harden_result, level)
        for _ in range(n_warmup):
            for i in range(min(5, N)):
                _ = model(inputs[i])
        times = []
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            for i in range(N):
                _ = model(inputs[i])
            t1 = time.perf_counter()
            times.append(t1 - t0)
        times = np.array(times)
        mean_total = float(np.mean(times))
        std_total = float(np.std(times))
        mean_per_sample = mean_total / max(N, 1) * 1e6
        std_per_sample = std_total / max(N, 1) * 1e6
        if level == 0:
            soft_time = mean_per_sample
        speedup = soft_time / mean_per_sample if mean_per_sample > 0 else float("inf")
        results.append({
            "level": level, "hard_layers": level, "total_layers": depth,
            "label": model.label,
            "mean_time_per_sample_us": mean_per_sample,
            "std_time_per_sample_us": std_per_sample,
            "total_time_ms": mean_total * 1e3,
            "speedup_vs_soft": speedup, "n_samples": N, "n_repeats": n_repeats,
        })
    return results


def benchmark_binary_accuracy(soft_model, harden_result: dict, test_inputs: np.ndarray,
                              test_labels: np.ndarray, threshold: float = 0.5) -> list[dict]:
    depth = len(soft_model.layers)
    inputs = np.asarray(test_inputs)
    labels = np.asarray(test_labels).ravel()
    N_labels = len(labels)

    def _get_verdict(model, level_idx):
        raw_list = [model(inputs[i]) for i in range(inputs.shape[0])]
        raw_arr = np.array(raw_list)
        if raw_arr.ndim > 1 and raw_arr.shape[1] > 1:
            raw_per_sample = raw_arr[:, 0]
        else:
            raw_per_sample = raw_arr.ravel()
        raw_per_sample = raw_per_sample.astype(np.float32)
        preds = np.where(raw_per_sample > threshold, 1, 0).astype(np.int8)
        return raw_per_sample, preds

    soft_model_wrap = BinaryLayerwiseHardenedNetwork(soft_model, harden_result, 0)
    soft_raw, soft_preds = _get_verdict(soft_model_wrap, 0)
    results = []
    for level in range(depth + 1):
        model = BinaryLayerwiseHardenedNetwork(soft_model, harden_result, level)
        raw_per_sample, preds = _get_verdict(model, level)
        n = min(len(preds), N_labels)
        preds = preds[:n]; raw_sample = raw_per_sample[:n]
        labs = labels[:n]; soft_p = soft_preds[:n]
        accuracy = float(np.mean(preds == labs))
        agreement = float(np.mean(preds == soft_p))
        per_class = {}
        for val, name in [(0, "FALSE"), (1, "TRUE")]:
            mask = labs == val
            if mask.sum() > 0:
                per_class[name] = float(np.mean(preds[mask] == val))
            else:
                per_class[name] = float("nan")
        mean_abs = float(np.mean(np.abs(raw_sample)))
        results.append({
            "level": level, "hard_layers": level, "total_layers": depth,
            "label": model.label, "accuracy": accuracy,
            "agreement_with_soft": agreement, "per_class_accuracy": per_class,
            "mean_abs_output": mean_abs,
        })
    return results


def benchmark_binary_full(soft_model, harden_result: dict, test_inputs: np.ndarray,
                          test_labels: np.ndarray, threshold: float = 0.5,
                          n_warmup: int = 3, n_repeats: int = 10,
                          timing_inputs: np.ndarray | None = None) -> dict:
    speed_inputs = timing_inputs if timing_inputs is not None else test_inputs
    speed_results = benchmark_binary_inference(soft_model, harden_result, speed_inputs, n_warmup=n_warmup, n_repeats=n_repeats)
    accuracy_results = benchmark_binary_accuracy(soft_model, harden_result, test_inputs, test_labels, threshold=threshold)
    levels = [{**s, **a} for s, a in zip(speed_results, accuracy_results)]
    lines = []
    lines.append("=" * 80)
    lines.append("BINARY DLGN HARDENING LEVEL BENCHMARK: ACCURACY vs SPEED")
    lines.append("=" * 80)
    lines.append(f"{'Level':>5s}  {'Mode':<35s}  {'Accuracy':>8s}  {'Agree/Soft':>10s}  {'us/sample':>10s}  {'Speedup':>7s}")
    lines.append("-" * 80)
    for lvl in levels:
        lines.append(f"{lvl['level']:>5d}  {lvl['label']:<35s}  {lvl['accuracy']:>8.1%}  {lvl['agreement_with_soft']:>10.1%}  {lvl['mean_time_per_sample_us']:>10.1f}  {lvl['speedup_vs_soft']:>6.1f}x")
    lines.append("-" * 80)
    soft = levels[0]; hard = levels[-1]
    acc_drop = soft["accuracy"] - hard["accuracy"]
    speedup = hard["speedup_vs_soft"]
    lines.append(f"\nSoft -> Hard: accuracy change = {acc_drop:+.1%}, speedup = {speedup:.1f}x")
    if len(levels) > 2:
        soft_acc = soft["accuracy"]
        candidates = [l for l in levels[1:-1] if l["accuracy"] >= 0.9 * soft_acc]
        if candidates:
            best = max(candidates, key=lambda l: l["speedup_vs_soft"])
            lines.append(f"Best intermediate: {best['label']} — accuracy={best['accuracy']:.1%}, speedup={best['speedup_vs_soft']:.1f}x")
    lines.append("=" * 80)
    summary = "\n".join(lines)
    depth = levels[0]["total_layers"]
    return {"levels": levels, "summary": summary, "depth": depth}


def format_binary_benchmark_table(results: dict) -> str:
    return results["summary"]
