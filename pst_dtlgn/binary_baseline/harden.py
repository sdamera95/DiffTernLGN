from __future__ import annotations
import jax
import jax.numpy as jnp
import numpy as np
from pst_dtlgn.network.inference import LearnedCircuitBase
from pst_dtlgn.binary_baseline.gates import (
    BIN_TRUTH_TABLES, GATE_NAMES_BIN, NUMBER_OF_GATES,
)


def round_binary(x: np.ndarray) -> np.ndarray:
    return np.where(np.asarray(x) > 0.5, 1, 0).astype(np.int8)


class BinaryLearnedCircuit(LearnedCircuitBase):

    def __init__(self, harden_result: dict) -> None:
        self.gate_indices = harden_result["gate_indices"]
        self.connections = harden_result["connections"]
        self.input_dim = harden_result["input_dim"]
        self.output_dim = harden_result["output_dim"]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x)
        if not np.all(np.isin(x_arr, [0, 1])):
            raise ValueError(
                "BinaryLearnedCircuit requires binary inputs in {0, 1}. "
                "Use round_binary() to convert continuous inputs first."
            )
        h = x_arr.astype(np.int8)
        for gate_idx, conn in zip(self.gate_indices, self.connections):
            a = h[conn[:, 0]]
            b = h[conn[:, 1]]
            idx = a.astype(np.int32) * 2 + b.astype(np.int32)
            n = len(gate_idx)
            out = np.array(
                [BIN_TRUTH_TABLES[gate_idx[j], idx[j]] for j in range(n)],
                dtype=np.int8,
            )
            h = out
        return h

    @property
    def signal_values(self) -> tuple[int, ...]:
        return (0, 1)

    @staticmethod
    def round_fn(x: np.ndarray) -> np.ndarray:
        return round_binary(x)

    @property
    def num_gates(self) -> int:
        return NUMBER_OF_GATES


def harden_binary_network(network) -> dict:
    gate_indices = []
    connections = []
    gate_counts_int: dict[int, int] = {}
    gate_counts_named: dict[str, int] = {}
    per_layer = []
    total_neurons = 0
    for layer in network.layers:
        selected = np.array(jnp.argmax(layer.logits, axis=-1))
        gate_indices.append(selected)
        connections.append(np.array(layer.connections))
        total_neurons += len(selected)
        layer_counts: dict[str, int] = {}
        for g in selected:
            g_int = int(g)
            name = GATE_NAMES_BIN[g_int]
            layer_counts[name] = layer_counts.get(name, 0) + 1
            gate_counts_int[g_int] = gate_counts_int.get(g_int, 0) + 1
            gate_counts_named[name] = gate_counts_named.get(name, 0) + 1
        probs = np.array(jax.nn.softmax(layer.logits, axis=-1))
        max_probs = np.max(probs, axis=-1)
        per_layer.append({
            "gate_counts": layer_counts,
            "mean_max_prob": float(np.mean(max_probs)),
            "min_max_prob": float(np.min(max_probs)),
        })
    return {
        "gate_indices": gate_indices, "connections": connections,
        "input_dim": network.input_dim, "output_dim": network.output_dim,
        "gate_census": gate_counts_int, "gate_census_named": gate_counts_named,
        "commitment_score": network.commitment_score(),
        "total_neurons": total_neurons,
        "num_gates": NUMBER_OF_GATES, "per_layer": per_layer,
    }
