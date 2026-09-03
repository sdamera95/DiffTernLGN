from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pst_dtlgn.applications.stl.predicates import PredicateModule, calibrate_thresholds
from pst_dtlgn.applications.stl.stl_labeler import STLSpec, label_episode
from pst_dtlgn.applications.stl.dataset_adapter import DatasetAdapter, Episode


def evaluate_predicates_episode(
    episode: Episode,
    pred_module: PredicateModule,
    mode: str = "soft",
) -> np.ndarray:
    T = len(episode.actions)
    P = pred_module.num_predicates

    pred_values = np.zeros((T, P), dtype=np.float32)
    for t in range(T):
        obs = episode.observations[t]
        if mode == "soft":
            pred_values[t] = pred_module.evaluate_soft(obs)
        else:
            pred_values[t] = pred_module.evaluate_hard(obs)

    return pred_values


def calibrate_predicates_from_episodes(
    episodes: list[Episode],
    pred_module: PredicateModule,
    low_pct: float = 30.0,
    high_pct: float = 70.0,
    max_episodes: int | None = None,
) -> list[tuple[float, float]]:
    eps = episodes[:max_episodes] if max_episodes else episodes

    feature_values = [[] for _ in range(pred_module.num_predicates)]

    for ep in eps:
        T = len(ep.actions)
        for t in range(T):
            obs = ep.observations[t]
            for i, pred in enumerate(pred_module.predicates):
                fval = float(pred.feature_fn(obs))
                feature_values[i].append(fval)

    thresholds = []
    for i in range(pred_module.num_predicates):
        vals = np.array(feature_values[i])
        theta_low, theta_high = calibrate_thresholds(vals, low_pct, high_pct)
        thresholds.append((theta_low, theta_high))

    return thresholds


def build_dataset_from_episodes(
    episodes: list[Episode],
    pred_module: PredicateModule,
    stl_specs: list[STLSpec],
    mode: str = "soft",
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    seed: int = 42,
) -> dict:
    N = len(episodes)
    if N == 0:
        raise ValueError("No episodes provided")

    rng = np.random.RandomState(seed)
    indices = rng.permutation(N)

    n_train = int(N * train_frac)
    n_val = int(N * val_frac)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    pred_series = []
    for ep in episodes:
        pred_series.append(evaluate_predicates_episode(ep, pred_module, mode))

    result = {}
    for spec in stl_specs:
        labels = []
        for ps in pred_series:
            ep_result = label_episode(ps, spec, mode="offline")
            labels.append(ep_result["labels"])

        def gather(idx_set):
            x = [pred_series[i] for i in idx_set]
            y = [labels[i] for i in idx_set]
            return x, y

        train_x, train_y = gather(train_idx)
        val_x, val_y = gather(val_idx)
        test_x, test_y = gather(test_idx)

        result[spec.name] = {
            "train_x": train_x,
            "train_y": train_y,
            "val_x": val_x,
            "val_y": val_y,
            "test_x": test_x,
            "test_y": test_y,
        }

    result["metadata"] = {
        "num_episodes": N,
        "num_train": len(train_idx),
        "num_val": len(val_idx),
        "num_test": len(test_idx),
        "num_predicates": pred_module.num_predicates,
        "predicate_names": pred_module.names,
        "spec_names": [s.name for s in stl_specs],
        "mode": mode,
        "seed": seed,
    }

    return result


def build_dataset(
    minari_id: str,
    pred_module: PredicateModule,
    stl_specs: list[STLSpec],
    max_episodes: int | None = None,
    mode: str = "soft",
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    seed: int = 42,
) -> dict:
    adapter = DatasetAdapter(minari_id)
    episodes = adapter.get_episodes(max_episodes=max_episodes)

    return build_dataset_from_episodes(
        episodes, pred_module, stl_specs,
        mode=mode, train_frac=train_frac, val_frac=val_frac, seed=seed,
    )


def report_dataset_stats(dataset: dict) -> dict:
    report = {}
    meta = dataset["metadata"]

    for spec_name in meta["spec_names"]:
        spec_data = dataset[spec_name]

        def dist_of(labels_list):
            if not labels_list:
                return {-1: 0.0, 0: 0.0, 1: 0.0}
            all_labels = np.concatenate(labels_list)
            n = len(all_labels)
            return {
                -1: float(np.mean(all_labels == -1)),
                0: float(np.mean(all_labels == 0)),
                1: float(np.mean(all_labels == 1)),
            }

        def total_steps(labels_list):
            return sum(len(l) for l in labels_list)

        report[spec_name] = {
            "train_dist": dist_of(spec_data["train_y"]),
            "val_dist": dist_of(spec_data["val_y"]),
            "test_dist": dist_of(spec_data["test_y"]),
            "total_train_steps": total_steps(spec_data["train_y"]),
            "total_val_steps": total_steps(spec_data["val_y"]),
            "total_test_steps": total_steps(spec_data["test_y"]),
        }

    report["metadata"] = meta
    return report


def pad_episodes(
    x_list: list[np.ndarray],
    y_list: list[np.ndarray],
    max_len: int | None = None,
    pad_value_x: float = 0.0,
    pad_value_y: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not x_list:
        raise ValueError("Empty episode list")

    N = len(x_list)
    P = x_list[0].shape[1]
    lengths = [len(x) for x in x_list]
    T_max = max_len if max_len is not None else max(lengths)

    x_padded = np.full((N, T_max, P), pad_value_x, dtype=np.float32)
    y_padded = np.full((N, T_max), pad_value_y, dtype=np.int8)
    mask = np.zeros((N, T_max), dtype=bool)

    for i in range(N):
        T_i = min(lengths[i], T_max)
        x_padded[i, :T_i] = x_list[i][:T_i]
        y_padded[i, :T_i] = y_list[i][:T_i]
        mask[i, :T_i] = True

    return x_padded, y_padded, mask
