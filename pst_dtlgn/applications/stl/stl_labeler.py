from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


# Pointwise boolean combinators

def stl_not(signal: np.ndarray) -> np.ndarray:
    return -signal


def stl_and(signal_a: np.ndarray, signal_b: np.ndarray) -> np.ndarray:
    return np.minimum(signal_a, signal_b)


def stl_or(signal_a: np.ndarray, signal_b: np.ndarray) -> np.ndarray:
    return np.maximum(signal_a, signal_b)


def stl_implies(signal_a: np.ndarray, signal_b: np.ndarray) -> np.ndarray:
    return np.maximum(-signal_a, signal_b)


# Temporal operators (offline / omniscient)

def stl_always(signal: np.ndarray, horizon: int | None = None) -> np.ndarray:
    T = len(signal)
    if T == 0:
        return np.array([], dtype=np.float64)

    if horizon is None:
        # Unbounded: suffix minimum
        result = np.empty(T, dtype=np.float64)
        result[-1] = signal[-1]
        for t in range(T - 2, -1, -1):
            result[t] = min(signal[t], result[t + 1])
        return result

    window = horizon + 1
    result = np.empty(T, dtype=np.float64)
    for t in range(T):
        end = min(t + window, T)
        result[t] = np.min(signal[t:end])
    return result


def stl_eventually(signal: np.ndarray, horizon: int | None = None) -> np.ndarray:
    T = len(signal)
    if T == 0:
        return np.array([], dtype=np.float64)

    if horizon is None:
        # Unbounded: suffix maximum
        result = np.empty(T, dtype=np.float64)
        result[-1] = signal[-1]
        for t in range(T - 2, -1, -1):
            result[t] = max(signal[t], result[t + 1])
        return result

    window = horizon + 1
    result = np.empty(T, dtype=np.float64)
    for t in range(T):
        end = min(t + window, T)
        result[t] = np.max(signal[t:end])
    return result


def stl_until(
    signal_a: np.ndarray,
    signal_b: np.ndarray,
    horizon: int | None = None,
) -> np.ndarray:
    T = len(signal_a)
    if T == 0:
        return np.array([], dtype=np.float64)

    result = np.empty(T, dtype=np.float64)

    for t in range(T):
        H = T - t if horizon is None else min(horizon + 1, T - t)
        best = -np.inf
        running_min_a = np.inf
        for dt in range(H):
            t_prime = t + dt
            if dt > 0:
                running_min_a = min(running_min_a, signal_a[t_prime - 1])
            else:
                running_min_a = np.inf  # vacuously true for dt=0
            candidate = min(signal_b[t_prime], running_min_a)
            best = max(best, candidate)
        result[t] = best

    return result


# Online (causal) temporal operators

def stl_always_online(signal: np.ndarray, horizon: int | None = None) -> np.ndarray:
    T = len(signal)
    if T == 0:
        return np.array([], dtype=np.float64)

    result = np.empty(T, dtype=np.float64)

    if horizon is None:
        running_min = signal[0]
        for t in range(T):
            running_min = min(running_min, signal[t])
            result[t] = running_min
    else:
        for t in range(T):
            start = max(0, t - horizon)
            result[t] = np.min(signal[start:t + 1])

    return result


def stl_eventually_online(signal: np.ndarray, horizon: int | None = None) -> np.ndarray:
    T = len(signal)
    if T == 0:
        return np.array([], dtype=np.float64)

    result = np.empty(T, dtype=np.float64)

    if horizon is None:
        running_max = signal[0]
        for t in range(T):
            running_max = max(running_max, signal[t])
            result[t] = running_max
    else:
        for t in range(T):
            start = max(0, t - horizon)
            result[t] = np.max(signal[start:t + 1])

    return result


# Ternary thresholding

def ternary_label(
    robustness: np.ndarray,
    delta: float = 0.1,
) -> np.ndarray:
    return np.where(
        robustness > delta, 1,
        np.where(robustness < -delta, -1, 0),
    ).astype(np.int8)


# STL Specification dataclasses

@dataclass
class STLSpec:
    name: str
    formula: str
    eval_fn: Callable[[np.ndarray], np.ndarray]
    pattern: str
    delta: float = 0.1

    def label_offline(self, predicates: np.ndarray) -> np.ndarray:
        robustness = self.eval_fn(predicates)
        return ternary_label(robustness, self.delta)

    def robustness(self, predicates: np.ndarray) -> np.ndarray:
        return self.eval_fn(predicates)


@dataclass
class OnlineSTLSpec:
    name: str
    formula: str
    eval_fn_online: Callable[[np.ndarray], np.ndarray]
    pattern: str
    delta: float = 0.1

    def label_online(self, predicates: np.ndarray) -> np.ndarray:
        robustness = self.eval_fn_online(predicates)
        return ternary_label(robustness, self.delta)


# Episode labeling

def label_episode(
    predicates: np.ndarray,
    spec: STLSpec,
    mode: str = "offline",
) -> dict:
    if mode == "offline":
        robustness = spec.eval_fn(predicates)
    elif mode == "online":
        if not hasattr(spec, "eval_fn_online"):
            raise ValueError(
                f"Spec '{spec.name}' does not have an online evaluator. "
                "Use an OnlineSTLSpec or provide eval_fn_online."
            )
        robustness = spec.eval_fn_online(predicates)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'offline' or 'online'.")

    labels = ternary_label(robustness, spec.delta)

    return {
        "robustness": robustness,
        "labels": labels,
        "spec_name": spec.name,
    }


def label_episodes_batch(
    episodes_predicates: list[np.ndarray],
    spec: STLSpec,
    mode: str = "offline",
) -> list[dict]:
    return [label_episode(ep, spec, mode) for ep in episodes_predicates]
