from __future__ import annotations
import itertools
from pathlib import Path
from typing import Optional
import numpy as np
import jax.numpy as jnp
from pst_dtlgn.core.constants import V_INV, NUM_TERNARY_GATES

CURATED_GATES: dict[str, list[int]] = {
    "AND":           [-1, -1, -1, -1,  0,  0, -1,  0,  1],
    "OR":            [-1,  0,  1,  0,  0,  1,  1,  1,  1],
    "NAND":          [ 1,  1,  1,  1,  0,  0,  1,  0, -1],
    "NOR":           [ 1,  0, -1,  0,  0, -1, -1, -1, -1],
    "XOR":           [-1,  0,  1,  0, -1,  0,  1,  0, -1],
    "XNOR":          [ 1,  0, -1,  0,  1,  0, -1,  0,  1],
    "IMPLIES":       [ 1,  1,  1,  0,  0,  1, -1,  0,  1],
    "SEQUENCE":      [-1, -1, -1,  0,  0,  0, -1,  0,  1],
    "SELECTOR":      [-1,  0,  1,  0,  0,  0,  1,  1,  1],
    "SEQUENCE_REV":  [-1,  0, -1, -1,  0,  0, -1,  0,  1],
    "SELECTOR_REV":  [-1,  0,  1,  0,  0,  1,  1,  0,  1],
    "AND_NOT_B":     [-1, -1, -1,  0,  0, -1,  1,  0, -1],
    "NOT_A_AND_B":   [-1,  0,  1, -1,  0,  0, -1, -1, -1],
    "OR_NOT_B":      [ 1,  0, -1,  1,  0,  0,  1,  1,  1],
    "MAJORITY":      [-1, -1,  0, -1,  0,  1,  0,  1,  1],
}


def _tt_to_canonical_index(tt: tuple[int, ...]) -> int:
    idx = 0
    for v in tt:
        idx = idx * 3 + (v + 1)
    return idx


def _canonical_index_to_tt(idx: int) -> tuple[int, ...]:
    digits = []
    for _ in range(9):
        digits.append(idx % 3 - 1)
        idx //= 3
    return tuple(reversed(digits))


def _hamming_distance(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.sum(a != b))


class GateLibrary:
    def __init__(self) -> None:
        self._build()

    def _build(self) -> None:
        all_tts = list(itertools.product([-1, 0, 1], repeat=9))
        self.truth_tables = np.array(all_tts, dtype=np.int8)
        v_inv_np = np.array(V_INV, dtype=np.float32)
        self.coefficients = self.truth_tables.astype(np.float32) @ v_inv_np.T

        self._tt_to_idx: dict[tuple[int, ...], int] = {}
        for i, tt in enumerate(all_tts):
            self._tt_to_idx[tt] = i

        self._compute_properties()
        self._map_curated_gates()

    def _compute_properties(self) -> None:
        tt = self.truth_tables
        swap_idx = [0, 3, 6, 1, 4, 7, 2, 5, 8]
        tt_swapped = tt[:, swap_idx]
        neg_idx = [8, 7, 6, 5, 4, 3, 2, 1, 0]
        tt_negated = tt[:, neg_idx]

        symmetric = np.all(tt == tt_swapped, axis=1)
        self_dual = np.all(tt == -tt_negated, axis=1)

        mono_a = (
            (tt[:, 3] >= tt[:, 0]) & (tt[:, 6] >= tt[:, 3]) &
            (tt[:, 4] >= tt[:, 1]) & (tt[:, 7] >= tt[:, 4]) &
            (tt[:, 5] >= tt[:, 2]) & (tt[:, 8] >= tt[:, 5])
        )
        mono_b = (
            (tt[:, 1] >= tt[:, 0]) & (tt[:, 2] >= tt[:, 1]) &
            (tt[:, 4] >= tt[:, 3]) & (tt[:, 5] >= tt[:, 4]) &
            (tt[:, 7] >= tt[:, 6]) & (tt[:, 8] >= tt[:, 7])
        )
        constant = np.all(tt == tt[:, :1], axis=1)

        a_vals = np.array([-1, -1, -1, 0, 0, 0, 1, 1, 1], dtype=np.int8)
        pass_a = np.all(tt == a_vals[np.newaxis, :], axis=1)
        b_vals = np.array([-1, 0, 1, -1, 0, 1, -1, 0, 1], dtype=np.int8)
        pass_b = np.all(tt == b_vals[np.newaxis, :], axis=1)

        self.properties = {
            "symmetric": symmetric, "self_dual": self_dual,
            "monotone_a": mono_a, "monotone_b": mono_b,
            "constant": constant, "pass_through_a": pass_a, "pass_through_b": pass_b,
        }

    def _map_curated_gates(self) -> None:
        self.curated_names: list[Optional[str]] = [None] * NUM_TERNARY_GATES
        self.curated_indices: dict[str, int] = {}
        self._curated_tt_arrays: dict[str, np.ndarray] = {}

        for name, tt_list in CURATED_GATES.items():
            tt_tuple = tuple(tt_list)
            idx = self._tt_to_idx.get(tt_tuple)
            if idx is not None:
                self.curated_names[idx] = name
                self.curated_indices[name] = idx
                self._curated_tt_arrays[name] = np.array(tt_list, dtype=np.int8)

    def lookup_by_truth_table(self, tt) -> int:
        return self._tt_to_idx[tuple(int(v) for v in tt)]

    def lookup_by_index(self, idx: int) -> dict:
        result = {
            "truth_table": self.truth_tables[idx],
            "coefficients": self.coefficients[idx],
            "curated_name": self.curated_names[idx],
        }
        for prop_name, prop_arr in self.properties.items():
            result[prop_name] = bool(prop_arr[idx])
        return result

    def nearest_curated(self, tt) -> tuple[str, int]:
        tt_np = np.array(tt, dtype=np.int8)
        best_name = ""
        best_dist = 10
        for name, curated_tt in self._curated_tt_arrays.items():
            dist = _hamming_distance(tt_np, curated_tt)
            if dist < best_dist:
                best_dist = dist
                best_name = name
                if dist == 0:
                    break
        return best_name, best_dist

    def save(self, path: str | Path) -> None:
        data = {
            "truth_tables": self.truth_tables,
            "coefficients": self.coefficients,
        }
        for prop_name, prop_arr in self.properties.items():
            data[f"prop_{prop_name}"] = prop_arr
        np.savez(str(path), **data)

    @classmethod
    def load(cls, path: str | Path) -> GateLibrary:
        lib = cls.__new__(cls)
        data = np.load(str(path))

        lib.truth_tables = data["truth_tables"]
        lib.coefficients = data["coefficients"]

        # Rebuild hash map from loaded truth tables
        lib._tt_to_idx = {}
        for i in range(NUM_TERNARY_GATES):
            tt_tuple = tuple(int(v) for v in lib.truth_tables[i])
            lib._tt_to_idx[tt_tuple] = i

        lib.properties = {}
        for key in data.files:
            if key.startswith("prop_"):
                prop_name = key[5:]
                lib.properties[prop_name] = data[key]

        lib._map_curated_gates()
        return lib
