from __future__ import annotations
import jax.numpy as jnp
import numpy as np
from pst_dtlgn.core.constants import V, V_INV
from pst_dtlgn.core.gate_library import GateLibrary
from pst_dtlgn.network.inference import LearnedCircuitBase


def round_ternary(x: np.ndarray | jnp.ndarray) -> np.ndarray:
    x = np.asarray(x)
    return np.where(x > 0.5, 1, np.where(x < -0.5, -1, 0)).astype(np.int8)


def harden_neuron(
    weights: np.ndarray | jnp.ndarray,
    gate_library: GateLibrary,
) -> dict:
    w = np.asarray(weights, dtype=np.float32)
    V_np = np.asarray(V, dtype=np.float32)
    V_INV_np = np.asarray(V_INV, dtype=np.float32)
    tt_continuous = V_np @ w
    tt_discrete = round_ternary(tt_continuous)
    hardened_weights = V_INV_np @ tt_discrete.astype(np.float32)
    gate_idx = gate_library.lookup_by_truth_table(tt_discrete)
    curated_name, hamming_dist = gate_library.nearest_curated(tt_discrete)
    hard_error = float(np.mean((tt_continuous - tt_discrete.astype(np.float32)) ** 2))
    return {
        "gate_index": gate_idx,
        "truth_table": tt_discrete,
        "hardened_weights": hardened_weights,
        "curated_name": curated_name,
        "hamming_to_curated": hamming_dist,
        "hardening_error": hard_error,
        "continuous_truth_table": tt_continuous,
    }


def harden_layers(layers: list, gate_library: GateLibrary) -> dict:
    per_layer = []
    all_truth_tables = []
    all_hardened_weights = []
    all_connections = []
    total_error = 0.0
    total_neurons = 0
    gate_census: dict[int, int] = {}
    n_curated = 0
    n_curated_h1 = 0

    for layer in layers:
        layer_results = []
        n = layer.weights.shape[0]
        layer_tt = np.zeros((n, 9), dtype=np.int8)
        layer_hw = np.zeros((n, 9), dtype=np.float32)
        for j in range(n):
            result = harden_neuron(np.asarray(layer.weights[j]), gate_library)
            layer_results.append(result)
            layer_tt[j] = result["truth_table"]
            layer_hw[j] = result["hardened_weights"]
            total_error += result["hardening_error"]
            total_neurons += 1
            gidx = result["gate_index"]
            gate_census[gidx] = gate_census.get(gidx, 0) + 1
            if result["curated_name"] is not None and result["hamming_to_curated"] == 0:
                n_curated += 1
            if result["hamming_to_curated"] <= 1:
                n_curated_h1 += 1
        per_layer.append(layer_results)
        all_truth_tables.append(layer_tt)
        all_hardened_weights.append(layer_hw)
        all_connections.append(np.asarray(layer.connections, dtype=np.int64))

    mean_error = total_error / max(total_neurons, 1)
    curated_cov = n_curated / max(total_neurons, 1)
    curated_cov_h1 = n_curated_h1 / max(total_neurons, 1)
    return {
        "per_layer": per_layer,
        "truth_tables": all_truth_tables,
        "hardened_weights": all_hardened_weights,
        "connections": all_connections,
        "mean_error": mean_error,
        "gate_census": gate_census,
        "curated_coverage": curated_cov,
        "curated_coverage_h1": curated_cov_h1,
        "total_neurons": total_neurons,
    }


def harden_network(network, gate_library: GateLibrary) -> dict:
    result = harden_layers(network.layers, gate_library)
    result["input_dim"] = network.input_dim
    result["output_dim"] = network.output_dim
    result["num_gates"] = 19683
    return result


def harden_network_fast(network, gate_library: GateLibrary | None = None) -> dict:
    """Vectorized equivalent of :func:`harden_network`.

    Replaces the per-neuron Python loop in :func:`harden_layers` with batched
    numpy linear algebra. Produces truth tables and hardened weights that are
    bit-identical to :func:`harden_network`; returns the circuit-ready summary
    (``truth_tables``, ``hardened_weights``, ``connections``, dims, plus a
    vectorized ``gate_census`` / ``mean_error``) but NOT the per-neuron result
    dicts that the slow path builds for downstream analysis. The optional
    ``gate_library`` argument is accepted for drop-in compatibility and unused:
    the gate index is the base-3 encoding of the truth table (matching
    ``GateLibrary.lookup_by_truth_table``).
    """
    V_np = np.asarray(V, dtype=np.float32)
    V_INV_np = np.asarray(V_INV, dtype=np.float32)
    powers = (3 ** np.arange(8, -1, -1)).astype(np.int64)  # [3^8, ..., 3^0]

    truth_tables, hardened_weights, connections = [], [], []
    total_error = 0.0
    total_neurons = 0
    gate_census: dict[int, int] = {}
    for layer in network.layers:
        W = np.asarray(layer.weights, dtype=np.float32)          # (n, 9)
        tt_cont = W @ V_np.T                                      # (n, 9)
        tt_disc = round_ternary(tt_cont)                          # (n, 9) int8
        hw = tt_disc.astype(np.float32) @ V_INV_np.T             # (n, 9)
        truth_tables.append(tt_disc)
        hardened_weights.append(hw)
        connections.append(np.asarray(layer.connections, dtype=np.int64))
        total_error += float(np.sum(
            np.mean((tt_cont - tt_disc.astype(np.float32)) ** 2, axis=1)
        ))
        total_neurons += tt_disc.shape[0]
        idx = ((tt_disc.astype(np.int64) + 1) * powers[None, :]).sum(axis=1)
        u, c = np.unique(idx, return_counts=True)
        for gi, cnt in zip(u.tolist(), c.tolist()):
            gate_census[gi] = gate_census.get(gi, 0) + cnt

    return {
        "truth_tables": truth_tables,
        "hardened_weights": hardened_weights,
        "connections": connections,
        "mean_error": total_error / max(total_neurons, 1),
        "gate_census": gate_census,
        "total_neurons": total_neurons,
        "input_dim": network.input_dim,
        "output_dim": network.output_dim,
        "num_gates": 19683,
    }


def _hardened_first_layer_forward(
    hardened_weights: np.ndarray,
    connections: np.ndarray,
    prev_outputs: np.ndarray,
) -> np.ndarray:
    a = prev_outputs[connections[:, 0]].astype(np.float64)
    b = prev_outputs[connections[:, 1]].astype(np.float64)
    a2 = a * a
    b2 = b * b
    monomials = np.stack([
        np.ones_like(a), a, b, a * b,
        a2, b2, a2 * b, a * b2, a2 * b2,
    ], axis=-1)
    W = hardened_weights.astype(np.float64)
    raw = np.sum(W * monomials, axis=-1)
    clipped = np.clip(raw, -1.0, 1.0)
    return round_ternary(clipped)


def _polynomial_layer_forward(
    hardened_weights: np.ndarray,
    connections: np.ndarray,
    prev_outputs: np.ndarray,
) -> np.ndarray:
    a = prev_outputs[connections[:, 0]].astype(np.float64)
    b = prev_outputs[connections[:, 1]].astype(np.float64)
    a2 = a * a
    b2 = b * b
    monomials = np.stack([
        np.ones_like(a), a, b, a * b,
        a2, b2, a2 * b, a * b2, a2 * b2,
    ], axis=-1)
    W = hardened_weights.astype(np.float64)
    raw = np.sum(W * monomials, axis=-1)
    return np.clip(raw, -1.0, 1.0)


def _hard_layer_forward(
    truth_tables: np.ndarray,
    connections: np.ndarray,
    prev_outputs: np.ndarray,
) -> np.ndarray:
    n = truth_tables.shape[0]
    a = prev_outputs[connections[:, 0]]
    b = prev_outputs[connections[:, 1]]
    idx = (a.astype(np.int32) + 1) * 3 + (b.astype(np.int32) + 1)
    out = np.array([truth_tables[j, idx[j]] for j in range(n)], dtype=np.int8)
    return out


class TernaryLearnedCircuit(LearnedCircuitBase):

    def __init__(self, harden_result: dict) -> None:
        self.truth_tables = harden_result["truth_tables"]
        self.hardened_weights = harden_result["hardened_weights"]
        self.connections = harden_result["connections"]
        self.input_dim = harden_result["input_dim"]
        self.output_dim = harden_result["output_dim"]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x)
        # Check BEFORE int8 cast to catch non-ternary floats
        if not np.all(np.isin(x_arr, [-1, 0, 1])):
            raise ValueError(
                "TernaryLearnedCircuit requires ternary inputs in "
                "{-1, 0, 1}. For real-valued data, use TernaryEncoder "
                "to convert inputs to balanced ternary representation "
                "first."
            )
        h = x_arr.astype(np.int8)
        for tt, conn in zip(self.truth_tables, self.connections):
            h = _hard_layer_forward(tt, conn, h)
        return h

    @property
    def signal_values(self) -> tuple[int, ...]:
        return (-1, 0, 1)

    @staticmethod
    def round_fn(x: np.ndarray) -> np.ndarray:
        return round_ternary(x)

    @property
    def num_gates(self) -> int:
        return 19683


LearnedCircuit = TernaryLearnedCircuit


class TernaryLearnedPolynomial:

    def __init__(self, harden_result: dict) -> None:
        self.hardened_weights = harden_result["hardened_weights"]
        self.connections = harden_result["connections"]
        self.input_dim = harden_result["input_dim"]
        self.output_dim = harden_result["output_dim"]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        h = np.asarray(x, dtype=np.float64)
        for hw, conn in zip(self.hardened_weights, self.connections):
            h = _polynomial_layer_forward(hw, conn, h)
        return h


LearnedPolynomialNetwork = TernaryLearnedPolynomial
